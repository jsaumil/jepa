import os
import sys
import csv
import json
import time
import zipfile
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from vl_jepa import VLJEPA
from test_prepare import DeepFakeDataset, collate_fn
from train_decoder import compute_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Test a trained VL-JEPA checkpoint on labelled videos")

    # data
    parser.add_argument("--videos", type=str, required=True, help="folder with the test videos")
    parser.add_argument("--csv", type=str, required=True, help="CSV file (or folder of CSVs) with the labels")

    # model: a train_decoder.py checkpoint, which holds the encoder + MLP
    parser.add_argument("--ckpt", type=str, required=True)

    # run params
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--img_size", type=int, default=224)
    parser.add_argument("--max_frames", type=int, default=16)
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="predict fake when prob_fake >= threshold")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no_amp", action="store_false", dest="amp")

    # output
    parser.add_argument("--output_dir", type=str, default="test_results")
    parser.add_argument("--log_every_batches", type=int, default=20)

    return parser.parse_args()


def load_model(path, device):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # torch.load can miss damage inside the archive, so check it first
    if not zipfile.is_zipfile(path):
        raise RuntimeError(f"Checkpoint is corrupted or incomplete (not a valid archive): {path}")
    with zipfile.ZipFile(path) as z:
        bad_entry = z.testzip()
    if bad_entry is not None:
        raise RuntimeError(f"Checkpoint is corrupted (bad checksum in {bad_entry}): {path}")

    ckpt = torch.load(path, map_location="cpu")
    state = ckpt["model_state_dict"]
    if not any(k.startswith("decoder.") for k in state):
        raise RuntimeError(
            f"{path} has no decoder weights. Use a checkpoint from train_decoder.py "
            "(checkpoints_decoder/), not an encoder checkpoint from train.py."
        )

    model = VLJEPA(**ckpt["model_config"])
    model.load_state_dict(state, strict=True)

    bad_params = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    if bad_params:
        raise RuntimeError(f"Checkpoint has nan/inf in params: {bad_params}")

    print(f"Loaded {path} (saved_epoch={ckpt.get('epoch')} saved_loss={ckpt.get('loss')})")
    return model.to(device).eval()


def count_csv_rows(csv_path):
    path = Path(csv_path)
    files = [path] if path.is_file() else sorted(path.glob("*.csv"))
    total = 0
    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            total += sum(1 for _ in csv.DictReader(f))
    return total


def roc_auc(probs, labels):
    # rank-based AUC, tied scores share their average rank
    n_pos = (labels == 1).sum().item()
    n_neg = (labels == 0).sum().item()
    if n_pos == 0 or n_neg == 0:
        return None
    _, inverse, counts = probs.double().unique(return_inverse=True, return_counts=True)
    avg_ranks = counts.cumsum(0).double() - (counts.double() - 1) / 2
    rank_sum = avg_ranks[inverse][labels == 1].sum().item()
    return (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def full_metrics(probs, labels, loss, threshold):
    preds = (probs >= threshold).long()
    metrics = compute_metrics(preds, labels)
    tp, fp, fn, tn = metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]

    specificity = tn / max(tn + fp, 1)
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    metrics.update({
        "specificity": specificity,
        "npv": tn / max(tn + fn, 1),
        "balanced_accuracy": (metrics["recall"] + specificity) / 2,
        "mcc": (tp * tn - fp * fn) / denom if denom > 0 else 0.0,
        "roc_auc": roc_auc(probs, labels),
        "loss": loss,
        "threshold": threshold,
        "n_fake": tp + fn,
        "n_real": tn + fp,
        "n_pred_fake": tp + fp,
        "n_pred_real": tn + fn,
        "majority_baseline_accuracy": max(tp + fn, tn + fp) / max(metrics["n_samples"], 1),
    })
    return metrics


@torch.no_grad()
def run_inference(model, dataset, device, args):
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    all_probs = []
    all_labels = []
    paths = []
    unreadable = []
    total_loss = 0.0
    seen = 0
    t0 = time.time()

    for i, batch in enumerate(loader):
        batch_paths = [s["video_path"] for s in dataset.samples[seen: seen + batch["x"].shape[0]]]
        seen += batch["x"].shape[0]

        # the dataset returns an all-zero clip for a video it can't decode.
        # Real frames are never all zero after normalization, so these are
        # left out instead of being scored as if they were real predictions.
        valid = batch["x"].flatten(1).abs().sum(dim=1) > 0
        unreadable.extend(p for p, ok in zip(batch_paths, valid.tolist()) if not ok)

        if valid.any():
            x = batch["x"][valid].to(device, non_blocking=True)
            query = batch["query"][valid].to(device, non_blocking=True)
            labels = batch["label"][valid].to(device, non_blocking=True)

            # same as training: encoder under autocast, decoder on float features
            with autocast(device_type=device.type, enabled=args.amp):
                y_pred = model.encode(x, query)
            logits = model.decoder(y_pred.float())

            total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
            all_probs.append(logits.softmax(dim=-1)[:, 1].cpu())
            all_labels.append(labels.cpu())
            paths.extend(p for p, ok in zip(batch_paths, valid.tolist()) if ok)

        if (i + 1) % args.log_every_batches == 0 or (i + 1) == len(loader):
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(loader) - i - 1)
            print(f"[test] batch {i + 1}/{len(loader)} | {elapsed:.0f}s elapsed | eta {eta:.0f}s", flush=True)

    if not all_probs:
        raise ValueError("None of the videos could be decoded.")

    labels = torch.cat(all_labels)
    return torch.cat(all_probs), labels, paths, unreadable, total_loss / labels.shape[0]


def print_report(metrics):
    tp, fp, fn, tn = metrics["tp"], metrics["fp"], metrics["fn"], metrics["tn"]
    auc = metrics["roc_auc"]

    print("\nResults (positive class = fake)")
    print("-" * 60)
    print(f"Samples:            {metrics['n_samples']} ({metrics['n_fake']} fake, {metrics['n_real']} real)")
    print(f"Accuracy:           {metrics['accuracy']:.4f}")
    print(f"Precision:          {metrics['precision']:.4f}")
    print(f"Recall:             {metrics['recall']:.4f}")
    print(f"F1:                 {metrics['f1']:.4f}")
    print(f"Specificity:        {metrics['specificity']:.4f}")
    print(f"NPV:                {metrics['npv']:.4f}")
    print(f"Balanced accuracy:  {metrics['balanced_accuracy']:.4f}")
    print(f"MCC:                {metrics['mcc']:.4f}")
    print(f"ROC-AUC:            {'n/a (needs both classes)' if auc is None else f'{auc:.4f}'}")
    print(f"Loss:               {metrics['loss']:.4f}")
    print(f"Majority baseline:  {metrics['majority_baseline_accuracy']:.4f} (accuracy of always guessing the larger class)")

    print("\nConfusion matrix")
    print(f"{'':>14}{'pred fake':>12}{'pred real':>12}")
    print(f"{'actual fake':>14}{tp:>12}{fn:>12}")
    print(f"{'actual real':>14}{fp:>12}{tn:>12}")

    if metrics["n_pred_fake"] == 0 or metrics["n_pred_real"] == 0:
        only = "fake" if metrics["n_pred_real"] == 0 else "real"
        print(f"\nWARNING: the model predicted '{only}' for every video, so it is not "
              "separating the classes, whatever the accuracy says.")


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    args.amp = args.amp and device.type == "cuda"

    # is_train=False: evenly spaced frames, so results are repeatable
    dataset = DeepFakeDataset(
        videos_dir=args.videos,
        csv_dir=args.csv,
        img_size=args.img_size,
        max_frames=args.max_frames,
        is_train=False,
    )
    if len(dataset) == 0:
        raise ValueError(
            "No test samples found. The CSV needs a path column (File Path/file_path/filename) "
            "and a label column (Label/label/class), and the videos must exist."
        )

    n_rows = count_csv_rows(args.csv)
    n_skipped = n_rows - len(dataset)
    if n_skipped:
        print(f"WARNING: {n_skipped} of {n_rows} CSV rows skipped (see the warnings above)")

    model = load_model(args.ckpt, device)

    probs, labels, paths, unreadable, loss = run_inference(model, dataset, device, args)
    if unreadable:
        print(f"WARNING: {len(unreadable)} unreadable video(s) left out of the metrics")

    metrics = full_metrics(probs, labels, loss, args.threshold)
    print_report(metrics)

    os.makedirs(args.output_dir, exist_ok=True)
    metrics_path = os.path.join(args.output_dir, "test_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump({
            **metrics,
            "ckpt": args.ckpt,
            "n_csv_rows_skipped": n_skipped,
            "unreadable_videos": unreadable,
        }, f, indent=2)

    predictions_path = os.path.join(args.output_dir, "test_predictions.csv")
    with open(predictions_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "label", "prediction", "prob_fake", "correct"])
        for path, label, prob in zip(paths, labels.tolist(), probs.tolist()):
            pred = int(prob >= args.threshold)
            writer.writerow([path, label, pred, f"{prob:.6f}", int(pred == label)])

    print(f"\nSaved {metrics_path}")
    print(f"Saved {predictions_path}")


if __name__ == "__main__":
    main()
