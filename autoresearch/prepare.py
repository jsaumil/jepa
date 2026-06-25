import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Callable
import torchvision.transforms as T
from utils.patch_embed import PatchEmbed


class BaseImageDataset(Dataset):
    """Base dataset class for image loading with preprocessing."""
    
    def __init__(
        self,
        image_dir: str,
        patch_size: int = 16,
        img_size: int = 224,
        normalize: bool = False,
        extend_transforms: Optional[Callable] = None,
    ):
        self.image_dir = Path(image_dir)
        self.patch_size = patch_size
        self.img_size = img_size
        self.normalize = normalize
        
        # Validate img_size is divisible by patch_size
        assert img_size % patch_size == 0, \
            f"img_size ({img_size}) must be divisible by patch_size ({patch_size})"
        
        # Get all image files
        self.image_paths = self._get_image_paths()
        print(f"Found {len(self.image_paths)} images in {image_dir}")
        
        # Build transforms
        self.transform = self._build_transforms(extend_transforms)
        
        # Patch embedding layer
        self.patch_embed = PatchEmbed()
        
        # Compute patch info
        self.n_patches = (img_size // patch_size) ** 2
        
    def _get_image_paths(self) -> List[Path]:
        """Get all valid image paths from directory."""
        valid_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.webp', '.tiff'}
        image_paths = []
        
        if self.image_dir.is_file():
            # Single file
            if self.image_dir.suffix.lower() in valid_extensions:
                return [self.image_dir]
        
        # Directory - get all images
        for ext in valid_extensions:
            image_paths.extend(self.image_dir.glob(f'*{ext}'))
            image_paths.extend(self.image_dir.glob(f'*{ext.upper()}'))
        
        # Also check subdirectories
        for ext in valid_extensions:
            image_paths.extend(self.image_dir.glob(f'**/*{ext}'))
            image_paths.extend(self.image_dir.glob(f'**/*{ext.upper()}'))
        
        return sorted(set(image_paths))
    
    def _build_transforms(self, extend_transforms: Optional[Callable] = None) -> Callable:
        """Build preprocessing transforms."""
        transforms_list = [
            # CRITICAL: Resize to fixed size divisible by patch_size
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),  # Converts to [0, 1] and [C, H, W]
        ]
        
        # ImageNet normalization (standard for ViT)
        if self.normalize:
            transforms_list.append(
                T.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        
        # Add any extra transforms (augmentations, etc.)
        if extend_transforms is not None:
            transforms_list.append(extend_transforms)
        
        return T.Compose(transforms_list)
    
    def load_image(self, idx: int) -> torch.Tensor:
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image)
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        raise NotImplementedError


class MaskedImageDataset(BaseImageDataset):
    """Dataset for Masked Image Modeling (like MAE)."""
    
    def __init__(
        self,
        image_dir: str,
        patch_size: int = 16,
        img_size: int = 224,
        mask_ratio: float = 0.75,  # 75% masking like MAE
        normalize: bool = True,
        extend_transforms: Optional[Callable] = None,
    ):
        super().__init__(
            image_dir=image_dir,
            patch_size=patch_size,
            img_size=img_size,
            normalize=normalize,
            extend_transforms=extend_transforms,
        )
        
        self.mask_ratio = mask_ratio
        self.n_masked = int(self.n_patches * mask_ratio)
        self.n_visible = self.n_patches - self.n_masked
        
        print(f"Total patches: {self.n_patches}")
        print(f"Masked patches: {self.n_masked} ({mask_ratio*100:.1f}%)")
        print(f"Visible patches: {self.n_visible}")
    
    def generate_mask(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # Random permutation of all patch indices
        permutation = torch.randperm(self.n_patches)
        
        # First n_masked are masked, rest are visible
        masked_indices = permutation[:self.n_masked]
        visible_indices = permutation[self.n_masked:]
        
        return masked_indices, visible_indices
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load and preprocess image
        pixel_values = self.load_image(idx)  # [C, H, W]
        
        # Add batch dimension for patch_embed
        x = pixel_values.unsqueeze(0)  # [1, C, H, W]
        
        # Get patch embeddings
        patch_embeddings = self.patch_embed(x)  # [1, n_patches, embed_dim]
        patch_embeddings = patch_embeddings.squeeze(0)  # [n_patches, embed_dim]
        
        # Generate mask
        masked_indices, visible_indices = self.generate_mask()
        
        # Create boolean mask
        mask = torch.zeros(self.n_patches, dtype=torch.bool)
        mask[masked_indices] = True
        
        return {
            "pixel_values": pixel_values,
            "patch_embeddings": patch_embeddings,
            "masked_indices": masked_indices,
            "visible_indices": visible_indices,
            "mask": mask,
            "image_path": str(self.image_paths[idx]),
        }
    

def create_dataloader(
    image_dir: str,
    batch_size: int = 32,
    patch_size: int = 16,
    img_size: int = 224,
    mask_ratio: float = 0.75,
    num_workers: int = 4,
    shuffle: bool = True,
    pin_memory: bool = True,
    drop_last: bool = True,
    normalize: bool = True,
    extend_transforms: Optional[Callable] = None,
) -> Tuple[DataLoader, MaskedImageDataset]:
    """
    Create dataloader for masked image modeling.
    
    Args:
        image_dir: Path to images
        batch_size: Batch size
        patch_size: Patch size for ViT
        img_size: Resize images to this size
        mask_ratio: Fraction of patches to mask
        num_workers: Number of data loading workers
        shuffle: Whether to shuffle data
        pin_memory: Pin memory for faster GPU transfer
        drop_last: Drop incomplete last batch
        normalize: Use ImageNet normalization
        extend_transforms: Additional transforms
    
    Returns:
        Tuple of (dataloader, dataset)
    """
    dataset = MaskedImageDataset(
        image_dir=image_dir,
        patch_size=patch_size,
        img_size=img_size,
        mask_ratio=mask_ratio,
        normalize=normalize,
        extend_transforms=extend_transforms,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=custom_collate_fn,
    )
    
    return dataloader, dataset


def custom_collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function to handle dict batching."""
    collated = {}
    
    for key in batch[0].keys():
        if key == "image_path":
            # Keep as list of strings
            collated[key] = [item[key] for item in batch]
        else:
            collated[key] = torch.stack([item[key] for item in batch])
    
    return collated