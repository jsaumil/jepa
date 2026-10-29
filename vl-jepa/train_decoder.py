import os
import sys
import time
import math
import json
import csv
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from vl_jepa import VLJEPA
from prepare import DeepFakeDataset, collate_fn, QUERY


def parse_args():
    parser = argparse.ArgumentParser(description="Train MLP decoder on top of frozen VL-JEPA encoder")

    # data
    parser.add_argument("--train_videos", type=str, required=True)
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_videos", type=str, default=None)
    parser.add_argument("--val_csv", type=str, default=None)
    parser.add_argument("--val_split", type=float, default=0.2)

    # encoder (frozen, loaded from checkpoint)
    parser.add_argument("--encoder_ckpt", type=str, default="checkpoints/best.pt")
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--vocab_size", type=int, default=50244)
    parser.add_argument("--small_model", action="store_true", default=False)

    # decoder
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)

    # training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="amp")

    # data params
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--max_frames", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)

    # checkpointing (separate dir so the encoder's best.pt is never overwritten)
    parser.add_argument("--save_dir", type=str, default="checkpoints_decoder")
    parser.add_argument("--save_every", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None)

    # logging
    parser.add_argument("--print_every", type=int, default=1)

    return parser.parse_args()


def get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_metrics(preds, labels):
    # positive class = 1 (fake)
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    n = tp + fp + fn + tn
    acc = (tp + tn) / max(n, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return {
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "n_samples": n, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


@torch.no_grad()
def evaluate(model, dataloader, device, args):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        x = batch["x"].to(device)
        query = batch["query"].to(device)
        labels = batch["label"].to(device)

        with autocast(device_type="cuda", enabled=args.amp):
            out = model(x, query, labels=labels)
        logits = out["logits"]

        total_loss += out["loss"].item()
        n_batches += 1
        all_preds.append(logits.argmax(dim=-1).cpu())
        all_labels.append(labels.cpu())

    metrics = {"loss": total_loss / max(n_batches, 1)}
    if all_preds:
        metrics.update(compute_metrics(torch.cat(all_preds), torch.cat(all_labels)))
    return metrics


def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, args):
    model.train()  # encoder stays in eval mode (see VLJEPA.train)
    decoder = model.decoder
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []
    optimizer.zero_grad()

    for i, batch in enumerate(dataloader):
        x = batch["x"].to(device)
        query = batch["query"].to(device)
        labels = batch["label"].to(device)

        with autocast(device_type="cuda", enabled=args.amp):
            out = model(x, query, labels=labels)
        logits = out["logits"]
        loss = out["loss"]

        if not torch.isfinite(loss):
            raise RuntimeError(f"Training diverged: non-finite loss at batch {i + 1}")

        loss_scaled = loss / args.grad_accum
        if scaler:
            scaler.scale(loss_scaled).backward()
        else:
            loss_scaled.backward()

        if (i + 1) % args.grad_accum == 0 or (i + 1) == len(dataloader):
            if scaler:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(decoder.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(decoder.parameters(), args.grad_clip)
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item()
        n_batches += 1
        all_preds.append(logits.detach().argmax(dim=-1).cpu())
        all_labels.append(labels.cpu())

    metrics = {
        "loss": total_loss / max(n_batches, 1),
    }
    if all_preds:
        metrics.update(compute_metrics(torch.cat(all_preds), torch.cat(all_labels)))
    return metrics


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, loss, path, args):
    # full VL-JEPA (encoder + MLP) so the checkpoint is usable on its own for inference
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "loss": loss,
        "encoder_ckpt": args.encoder_ckpt,
        "model_config": {
            "embed_dim": args.embed_dim,
            "vocab_size": args.vocab_size,
            "small_model": args.small_model,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
        },
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt["epoch"], ckpt["loss"]


def make_loader(dataset, args, shuffle):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Query: \"{QUERY}\"")
    args.amp = args.amp and device.type == "cuda"

    # -- data (same split as train.py: seed 42, so val stays unseen by the encoder) --
    use_split = not (args.val_videos and args.val_csv)

    full_dataset = DeepFakeDataset(
        videos_dir=args.train_videos,
        csv_dir=args.train_csv,
        img_size=args.img_size,
        max_frames=args.max_frames,
        is_train=True,
    )
    if use_split:
        n = len(full_dataset)
        n_val = int(n * args.val_split)
        n_train = n - n_val

        indices = list(range(n))
        random.seed(42)
        random.shuffle(indices)
        train_dataset = Subset(full_dataset, indices[:n_train])
        val_dataset = Subset(full_dataset, indices[n_train:])
        print(f"Train: {n_train} samples | Val: {n_val} samples (split={args.val_split})")
    else:
        train_dataset = full_dataset
        val_dataset = DeepFakeDataset(
            videos_dir=args.val_videos,
            csv_dir=args.val_csv,
            img_size=args.img_size,
            max_frames=args.max_frames,
            is_train=False,
        )
        print(f"Train: {len(train_dataset)} samples | Val: {len(val_dataset)} samples (separate)")

    train_loader = make_loader(train_dataset, args, shuffle=True)
    val_loader = make_loader(val_dataset, args, shuffle=False)
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    if len(train_loader) == 0:
        raise ValueError(
            "No training batches were created. Reduce --batch_size or provide more data."
        )

    # -- full VL-JEPA model: frozen encoder from best.pt + trainable MLP --
    model = VLJEPA(
        embed_dim=args.embed_dim,
        vocab_size=args.vocab_size,
        small_model=args.small_model,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )
    model.load_encoder(args.encoder_ckpt)
    model.freeze_encoder()
    model = model.to(device)

    n_enc = sum(p.numel() for p in model.encoder.parameters()) / 1e6
    n_enc_trainable = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    n_dec = sum(p.numel() for p in model.decoder.parameters()) / 1e6
    print(f"Encoder params: {n_enc:.2f}M (frozen, trainable={n_enc_trainable})")
    print(f"Decoder params: {n_dec:.3f}M (trainable)")

    # -- optimizer: decoder params only --
    optimizer = torch.optim.AdamW(
        model.decoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    updates_per_epoch = math.ceil(len(train_loader) / max(args.grad_accum, 1))
    total_steps = updates_per_epoch * args.epochs
    warmup_steps = updates_per_epoch * args.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = GradScaler("cuda", enabled=True) if args.amp else None

    # -- resume --
    start_epoch = 0
    best_val_loss = float("inf")
    best_val_f1 = -1.0
    if args.resume and os.path.exists(args.resume):
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)
        start_epoch += 1
        print(f"Resumed from {args.resume}, epoch {start_epoch}")

    # -- save dir --
    os.makedirs(args.save_dir, exist_ok=True)

    metrics_path_json = os.path.join(args.save_dir, "metrics.json")
    metrics_path_csv = os.path.join(args.save_dir, "metrics.csv")
    history = []
    if args.resume and os.path.exists(metrics_path_json):
        try:
            with open(metrics_path_json, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    if history:
        best_val_loss = min(row.get("val_loss", float("inf")) for row in history)
        best_val_f1 = max(row.get("val_f1", -1.0) for row in history)

    def save_metrics():
        with open(metrics_path_json, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
        if history:
            keys = list(history[0].keys())
            with open(metrics_path_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=keys)
                w.writeheader()
                for row in history:
                    w.writerow(row)

    # -- training loop --
    print(f"\nStarting decoder training for {args.epochs} epochs")
    print("-" * 60)

    train_loss = float("nan")
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scheduler, scaler, device, args
        )
        val_metrics = evaluate(model, val_loader, device, args)
        train_loss = train_metrics["loss"]
        val_loss = val_metrics["loss"]

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        log_row = {
            "epoch": epoch + 1,
            "train_loss": float(train_loss),
            "train_accuracy": float(train_metrics.get("accuracy", 0.0)),
            "train_precision": float(train_metrics.get("precision", 0.0)),
            "train_recall": float(train_metrics.get("recall", 0.0)),
            "train_f1": float(train_metrics.get("f1", 0.0)),
            "train_n_samples": int(train_metrics.get("n_samples", 0)),
            "val_loss": float(val_loss),
            "val_accuracy": float(val_metrics.get("accuracy", 0.0)),
            "val_precision": float(val_metrics.get("precision", 0.0)),
            "val_recall": float(val_metrics.get("recall", 0.0)),
            "val_f1": float(val_metrics.get("f1", 0.0)),
            "val_n_samples": int(val_metrics.get("n_samples", 0)),
            "lr": float(lr_now),
            "elapsed_s": float(elapsed),
        }
        history.append(log_row)
        save_metrics()

        if (epoch + 1) % args.print_every == 0:
            msg = (
                f"Epoch {epoch+1}/{args.epochs} | "
                f"train_loss: {train_loss:.4f} | "
                f"train_acc: {log_row['train_accuracy']:.4f} | "
                f"train_prec: {log_row['train_precision']:.4f} | "
                f"train_rec: {log_row['train_recall']:.4f} | "
                f"train_f1: {log_row['train_f1']:.4f} || "
                f"val_loss: {val_loss:.4f} | "
                f"val_acc: {log_row['val_accuracy']:.4f} | "
                f"val_prec: {log_row['val_precision']:.4f} | "
                f"val_rec: {log_row['val_recall']:.4f} | "
                f"val_f1: {log_row['val_f1']:.4f} | "
                f"lr: {lr_now:.2e} | {elapsed:.1f}s"
            )
            print(msg)

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch+1}.pt")
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, train_loss, ckpt_path, args
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.save_dir, "best.pt")
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, val_loss, best_path, args
            )
            with open(os.path.join(args.save_dir, "best_metrics.json"), "w") as f:
                json.dump({**log_row, **{f"val_{k}": v for k, v in val_metrics.items()}}, f, indent=2)

        if val_metrics.get("f1", 0.0) > best_val_f1:
            best_val_f1 = val_metrics.get("f1", 0.0)
            best_f1_path = os.path.join(args.save_dir, "best_f1.pt")
            save_checkpoint(
                model, optimizer, scheduler, scaler, epoch, val_loss, best_f1_path, args
            )
            with open(os.path.join(args.save_dir, "best_f1_metrics.json"), "w") as f:
                json.dump({**log_row, **{f"val_{k}": v for k, v in val_metrics.items()}}, f, indent=2)

    save_metrics()

    # -- final save --
    final_path = os.path.join(args.save_dir, "final.pt")
    save_checkpoint(
        model, optimizer, scheduler, scaler, args.epochs - 1, train_loss, final_path, args
    )
    print(f"\nTraining complete. Final VL-JEPA model saved to {final_path}")


if __name__ == "__main__":
    main()
