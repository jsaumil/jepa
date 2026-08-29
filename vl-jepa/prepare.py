import os
import csv
import torch
import random
import cv2
import tiktoken
import numpy as np
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from typing import Dict, List, Tuple


QUERY = "it is fake or not?"
IMG_SIZE = 224
MAX_FRAMES = 16


class DeepFakeDataset(Dataset):
    def __init__(
        self,
        videos_dir: str,
        csv_dir: str,
        img_size: int = IMG_SIZE,
        max_frames: int = MAX_FRAMES,
        max_query_len: int = 32,
        max_label_len: int = 8,
        is_train: bool = True,
    ):
        self.videos_dir = Path(videos_dir)
        self.csv_dir = Path(csv_dir)
        self.img_size = img_size
        self.max_frames = max_frames
        self.max_query_len = max_query_len
        self.max_label_len = max_label_len
        self.is_train = is_train

        self.enc = tiktoken.get_encoding("gpt2")
        self.query_tokens = torch.tensor(
            self.enc.encode(QUERY), dtype=torch.long
        )

        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self.samples = self._load_csv_data()
        print(f"Loaded {len(self.samples)} samples from {csv_dir}")

    def _load_csv_data(self) -> List[Dict]:
        samples = []
        video_exts = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

        for csv_file in sorted(self.csv_dir.glob("*.csv")):
            with open(csv_file, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    file_path = row.get("File Path") or row.get("file_path") or row.get("filename")
                    label = row.get("Label") or row.get("label") or row.get("class")

                    if file_path is None or label is None:
                        continue

                    file_path = file_path.strip()
                    label = label.strip().upper()

                    video_stem = Path(file_path).stem
                    csv_subdir = Path(file_path).parent

                    video_path = None

                    candidate = self.videos_dir / csv_subdir / f"{video_stem}.mp4"
                    if candidate.exists():
                        video_path = candidate
                    else:
                        for ext in video_exts:
                            candidate = self.videos_dir / f"{video_stem}{ext}"
                            if candidate.exists():
                                video_path = candidate
                                break

                    if video_path is None:
                        for ext in video_exts:
                            matches = list(self.videos_dir.glob(f"**/{video_stem}{ext}"))
                            if matches:
                                video_path = matches[0]
                                break

                    if video_path is None:
                        print(f"Warning: no video found for '{video_stem}', skipping")
                        continue

                    samples.append({
                        "video_path": str(video_path),
                        "label": label,
                    })

        return samples

    def _read_video_frames(self, video_path: str) -> List[torch.Tensor]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Warning: cannot open '{video_path}'")
            return []

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []

        if total_frames <= self.max_frames:
            indices = list(range(total_frames))
        elif self.is_train:
            indices = sorted(random.sample(range(total_frames), self.max_frames))
        else:
            step = total_frames / self.max_frames
            indices = [int(i * step) for i in range(self.max_frames)]

        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = Image.fromarray(frame)
                frame = self.transform(frame)
                frames.append(frame)

        cap.release()
        return frames

    def _tokenize_label(self, label: str) -> torch.Tensor:
        label_upper = label.upper()
        if label_upper in ("FAKE", "1", "TRUE", "YES", "1.0"):
            text = "fake"
        else:
            text = "not fake"
        tokens = self.enc.encode(text)
        return torch.tensor(tokens, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        frames = self._read_video_frames(sample["video_path"])
        if not frames:
            frames = [torch.zeros(3, self.img_size, self.img_size)]

        x = torch.stack(frames, dim=1)  # [C, T, H, W]

        query = self.query_tokens.clone()
        label_tokens = self._tokenize_label(sample["label"])

        return {
            "x": x,
            "query": query,
            "y": label_tokens,
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    max_q = max(item["query"].shape[0] for item in batch)
    max_y = max(item["y"].shape[0] for item in batch)
    max_t = max(item["x"].shape[1] for item in batch)

    queries = []
    labels = []
    images = []

    for item in batch:
        q = item["query"]
        y = item["y"]
        x = item["x"]

        q_padded = torch.zeros(max_q, dtype=torch.long)
        q_padded[: q.shape[0]] = q

        y_padded = torch.zeros(max_y, dtype=torch.long)
        y_padded[: y.shape[0]] = y

        C, T_i, H, W = x.shape
        if T_i < max_t:
            pad = torch.zeros(C, max_t - T_i, H, W)
            x = torch.cat([x, pad], dim=1)

        queries.append(q_padded)
        labels.append(y_padded)
        images.append(x)

    return {
        "x": torch.stack(images),
        "query": torch.stack(queries),
        "y": torch.stack(labels),
    }


def create_dataloader(
    videos_dir: str,
    csv_dir: str,
    batch_size: int = 16,
    img_size: int = IMG_SIZE,
    max_frames: int = MAX_FRAMES,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    max_query_len: int = 32,
    max_label_len: int = 8,
    is_train: bool = True,
) -> Tuple[DataLoader, DeepFakeDataset]:
    dataset = DeepFakeDataset(
        videos_dir=videos_dir,
        csv_dir=csv_dir,
        img_size=img_size,
        max_frames=max_frames,
        max_query_len=max_query_len,
        max_label_len=max_label_len,
        is_train=is_train,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        collate_fn=collate_fn,
    )

    return dataloader, dataset


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare DeepFake dataset")
    parser.add_argument("--videos_dir", type=str, required=True, help="Path to videos folder")
    parser.add_argument("--csv_dir", type=str, required=True, help="Path to CSV folder")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--img_size", type=int, default=IMG_SIZE)
    parser.add_argument("--max_frames", type=int, default=MAX_FRAMES)
    parser.add_argument("--num_workers", type=int, default=4)
    args = parser.parse_args()

    dataloader, dataset = create_dataloader(
        videos_dir=args.videos_dir,
        csv_dir=args.csv_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        max_frames=args.max_frames,
        num_workers=args.num_workers,
    )

    print(f"\nDataset size: {len(dataset)}")
    print(f"Query: \"{QUERY}\" -> tokens: {dataset.query_tokens.tolist()}")
    print(f"Input format: (B, C, T, H, W) = (B, 3, <=16, 224, 224)")

    for batch in dataloader:
        print(f"\nBatch shapes:")
        print(f"  x:     {batch['x'].shape}  # (B, C, T, H, W)")
        print(f"  query: {batch['query'].shape}")
        print(f"  y:     {batch['y'].shape}")
        break
