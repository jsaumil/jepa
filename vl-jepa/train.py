import os
import sys
import time
import math
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from model import DeepFake
from prepare import create_dataloader, QUERY


def parse_args():
    parser = argparse.ArgumentParser(description="Train VL-JEPA DeepFake model")

    # data
    parser.add_argument("--train_images", type=str, required=True)
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_images", type=str, default=None)
    parser.add_argument("--val_csv", type=str, default=None)

    # model
    parser.add_argument("--embed_dim", type=int, default=768)
    parser.add_argument("--vocab_size", type=int, default=50244)

    # training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true", default=False)

    # data params
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--max_frames", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)

    # checkpointing
    parser.add_argument("--save_dir", type=str, default="checkpoints")
    parser.add_argument("--save_every", type=int, default=10)
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


@torch.no_grad()
def evaluate(model, dataloader, device, max_batches=None):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    for i, batch in enumerate(dataloader):
        if max_batches and i >= max_batches:
            break

        x = batch["x"].to(device)
        query = batch["query"].to(device)
        y = batch["y"].to(device)

        _, loss = model(x, query, y, train=True)
        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def train_one_epoch(model, dataloader, optimizer, scheduler, scaler, device, args):
    model.train()
    total_loss = 0.0
    n_batches = 0

    for i, batch in enumerate(dataloader):
        x = batch["x"].to(device)
        query = batch["query"].to(device)
        y = batch["y"].to(device)

        if args.amp:
            with autocast():
                _, loss = model(x, query, y, train=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            _, loss = model(x, query, y, train=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        optimizer.zero_grad()
        scheduler.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, loss, path):
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,
        "loss": loss,
    }, path)


def load_checkpoint(path, model, optimizer, scheduler, scaler):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    scheduler.load_state_dict(ckpt["scheduler_state_dict"])
    if scaler and ckpt.get("scaler_state_dict"):
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    return ckpt["epoch"], ckpt["loss"]


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Query: \"{QUERY}\"")

    # -- data --
    train_loader, train_dataset = create_dataloader(
        images_dir=args.train_images,
        csv_dir=args.train_csv,
        batch_size=args.batch_size,
        img_size=args.img_size,
        max_frames=args.max_frames,
        num_workers=args.num_workers,
        shuffle=True,
        is_train=True,
    )
    print(f"Train batches: {len(train_loader)}")

    val_loader = None
    if args.val_images and args.val_csv:
        val_loader, _ = create_dataloader(
            images_dir=args.val_images,
            csv_dir=args.val_csv,
            batch_size=args.batch_size,
            img_size=args.img_size,
            max_frames=args.max_frames,
            num_workers=args.num_workers,
            shuffle=False,
            is_train=False,
        )
        print(f"Val batches: {len(val_loader)}")

    # -- model --
    model = DeepFake(
        embed_dim=args.embed_dim,
        vocab_size=args.vocab_size,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Model params: {n_params:.2f}M")

    # -- optimizer --
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    scaler = GradScaler() if args.amp else None

    # -- resume --
    start_epoch = 0
    best_val_loss = float("inf")
    if args.resume and os.path.exists(args.resume):
        start_epoch, _ = load_checkpoint(args.resume, model, optimizer, scheduler, scaler)
        start_epoch += 1
        print(f"Resumed from {args.resume}, epoch {start_epoch}")

    # -- save dir --
    os.makedirs(args.save_dir, exist_ok=True)

    # -- training loop --
    print(f"\nStarting training for {args.epochs} epochs")
    print("-" * 60)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, args)

        val_loss = None
        if val_loader:
            val_loss = evaluate(model, val_loader, device)

        elapsed = time.time() - t0
        lr_now = optimizer.param_groups[0]["lr"]

        if epoch % args.print_every == 0:
            msg = f"Epoch {epoch+1}/{args.epochs} | train_loss: {train_loss:.4f} | lr: {lr_now:.2e} | {elapsed:.1f}s"
            if val_loss is not None:
                msg += f" | val_loss: {val_loss:.4f}"
            print(msg)

        # -- checkpoint --
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch+1}.pt")
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, train_loss, ckpt_path)

        if val_loss is not None and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = os.path.join(args.save_dir, "best.pt")
            save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_loss, best_path)

    # -- final save --
    final_path = os.path.join(args.save_dir, "final.pt")
    save_checkpoint(model, optimizer, scheduler, scaler, args.epochs - 1, train_loss, final_path)
    print(f"\nTraining complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main()
