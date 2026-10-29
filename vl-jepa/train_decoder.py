import os
import sys
import copy
import time
import math
import json
import csv
import random
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
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

    # feature caching: videos go through the frozen encoder once, the MLP
    # then trains on the cached embeddings (no video decoding per epoch)
    parser.add_argument("--batch_size", type=int, default=12,
                        help="videos per batch when running the frozen encoder")
    parser.add_argument("--cache_views", type=int, default=1,
                        help="random frame samplings cached per train video (augmentation)")
    parser.add_argument("--recompute_cache", action="store_true", default=False)

    # training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--decoder_batch_size", type=int, default=256)
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
    parser.add_argument("--log_every_batches", type=int, default=20)

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
def extract_features(model, dataset, device, args, name):
    """Run the frozen x-encoder + predictor over every video once -> ([N, C], [N])."""
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    feats = []
    labels = []
    t0 = time.time()
    for i, batch in enumerate(loader):
        x = batch["x"].to(device, non_blocking=True)
        query = batch["query"].to(device, non_blocking=True)
        with autocast(device_type="cuda", enabled=args.amp):
            y_pred = model.encode(x, query)
        feats.append(y_pred.float().cpu())
        labels.append(batch["label"])

        if (i + 1) % args.log_every_batches == 0 or (i + 1) == len(loader):
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(loader) - i - 1)
            print(f"[cache {name}] batch {i + 1}/{len(loader)} | "
                  f"{elapsed:.0f}s elapsed | eta {eta:.0f}s", flush=True)

    return torch.cat(feats), torch.cat(labels).long()


def build_feature_cache(model, train_dataset, val_dataset, device, args):
    cache_path = os.path.join(args.save_dir, "features_cache.pt")
    meta = {
        "encoder_ckpt": os.path.abspath(args.encoder_ckpt),
        "encoder_mtime": os.path.getmtime(args.encoder_ckpt),
        "n_train": len(train_dataset),
        "n_val": len(val_dataset),
        "cache_views": args.cache_views,
        "max_frames": args.max_frames,
        "img_size": args.img_size,
    }

    if os.path.exists(cache_path) and not args.recompute_cache:
        cache = torch.load(cache_path, map_location="cpu")
        if cache.get("meta") == meta:
            print(f"Loaded cached features from {cache_path}")
            return cache
        print("Feature cache is stale (encoder/data/settings changed), recomputing")

    print(f"\nCaching encoder features ({args.cache_views} train view(s) + val)")
    print("-" * 60)
    train_views = []
    train_labels = None
    for v in range(args.cache_views):
        f, train_labels = extract_features(
            model, train_dataset, device, args, f"train view {v + 1}/{args.cache_views}"
        )
        train_views.append(f)
    val_feats, val_labels = extract_features(model, val_dataset, device, args, "val")

    cache = {
        "meta": meta,
        "train": torch.stack(train_views),  # [V, N, C]
        "train_labels": train_labels,
        "val": val_feats,  # [N, C]
        "val_labels": val_labels,
    }
    torch.save(cache, cache_path)
    print(f"Saved feature cache to {cache_path}")
    return cache


@torch.no_grad()
def evaluate(decoder, feats, labels, args):
    decoder.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []

    for s in range(0, feats.shape[0], args.decoder_batch_size):
        logits = decoder(feats[s: s + args.decoder_batch_size])
        loss = F.cross_entropy(logits, labels[s: s + args.decoder_batch_size])
        total_loss += loss.item()
        n_batches += 1
        all_preds.append(logits.argmax(dim=-1))

    metrics = {"loss": total_loss / max(n_batches, 1)}
    metrics.update(compute_metrics(torch.cat(all_preds).cpu(), labels.cpu()))
    return metrics


def train_one_epoch(decoder, train_views, labels, optimizer, scheduler, args):
    decoder.train()
    V, N, _ = train_views.shape
    device = train_views.device
    # each sample uses a random cached view of its video this epoch
    view_idx = torch.randint(0, V, (N,), device=device)
    feats = train_views[view_idx, torch.arange(N, device=device)]
    perm = torch.randperm(N, device=device)

    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_labels = []

    for s in range(0, N, args.decoder_batch_size):
        idx = perm[s: s + args.decoder_batch_size]
        logits = decoder(feats[idx])
        loss = F.cross_entropy(logits, labels[idx])

        if not torch.isfinite(loss):
            raise RuntimeError(f"Training diverged: non-finite loss at batch {n_batches + 1}")

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(decoder.parameters(), args.grad_clip)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1
        all_preds.append(logits.detach().argmax(dim=-1))
        all_labels.append(labels[idx])

    metrics = {"loss": total_loss / max(n_batches, 1)}
    metrics.update(compute_metrics(torch.cat(all_preds).cpu(), torch.cat(all_labels).cpu()))
    return metrics


def save_checkpoint(model, optimizer, scheduler, epoch, loss, path, args):
    # full VL-JEPA (encoder + MLP) so the checkpoint is usable on its own for inference
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
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


def load_checkpoint(path, model, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    return ckpt["epoch"], ckpt["loss"]


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
        # val uses evenly spaced frames (deterministic), train uses random frames
        val_base = copy.copy(full_dataset)
        val_base.is_train = False
        train_dataset = Subset(full_dataset, indices[:n_train])
        val_dataset = Subset(val_base, indices[n_train:])
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

    if len(train_dataset) == 0:
        raise ValueError("No training samples found.")

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

    os.makedirs(args.save_dir, exist_ok=True)

    # -- encoder features: computed once, reused for every epoch --
    cache = build_feature_cache(model, train_dataset, val_dataset, device, args)
    train_views = cache["train"].to(device)
    train_labels = cache["train_labels"].to(device)
    val_feats = cache["val"].to(device)
    val_labels = cache["val_labels"].to(device)
    print(f"Train features: {tuple(train_views.shape)} | Val features: {tuple(val_feats.shape)}")

    # -- optimizer: decoder params only --
    optimizer = torch.optim.AdamW(
        model.decoder.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    updates_per_epoch = math.ceil(train_views.shape[1] / args.decoder_batch_size)
    total_steps = updates_per_epoch * args.epochs
    warmup_steps = updates_per_epoch * args.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # -- resume --
    start_epoch = 0
    best_val_loss = float("inf")
    best_val_f1 = -1.0
    if args.resume and os.path.exists(args.resume):
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch += 1
        print(f"Resumed from {args.resume}, epoch {start_epoch}")

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
            model.decoder, train_views, train_labels, optimizer, scheduler, args
        )
        val_metrics = evaluate(model.decoder, val_feats, val_labels, args)
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
            print(msg, flush=True)

        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch+1}.pt")
            save_checkpoint(model, optimizer, scheduler, epoch, train_loss, ckpt_path, args)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.save_dir, "best.pt")
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, best_path, args)
            with open(os.path.join(args.save_dir, "best_metrics.json"), "w") as f:
                json.dump({**log_row, **{f"val_{k}": v for k, v in val_metrics.items()}}, f, indent=2)

        if val_metrics.get("f1", 0.0) > best_val_f1:
            best_val_f1 = val_metrics.get("f1", 0.0)
            best_f1_path = os.path.join(args.save_dir, "best_f1.pt")
            save_checkpoint(model, optimizer, scheduler, epoch, val_loss, best_f1_path, args)
            with open(os.path.join(args.save_dir, "best_f1_metrics.json"), "w") as f:
                json.dump({**log_row, **{f"val_{k}": v for k, v in val_metrics.items()}}, f, indent=2)

    save_metrics()

    # -- final save --
    final_path = os.path.join(args.save_dir, "final.pt")
    save_checkpoint(model, optimizer, scheduler, args.epochs - 1, train_loss, final_path, args)
    print(f"\nTraining complete. Final VL-JEPA model saved to {final_path}")


if __name__ == "__main__":
    main()
