"""AffectNet dataset and data augmentation transforms."""

import os

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


def get_base_transform() -> transforms.Compose:
    """Standard preprocessing transform for validation / inference."""
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def get_augmentation_transforms() -> list[transforms.Compose]:
    """One augmentation transform per expand-factor slot (training only)."""
    base = get_base_transform()
    return [
        transforms.Compose([transforms.RandomHorizontalFlip(p=1.0), base]),
        transforms.Compose([transforms.RandomRotation(10), base]),
        transforms.Compose(
            [transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), base]
        ),
        transforms.Compose([transforms.GaussianBlur(3, sigma=(0.1, 1.0)), base]),
        transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.9, 1.0)), base]),
    ]


class AffectNet(Dataset):
    """AffectNet dataset for multi-task facial expression recognition.

    Each sample returns:
        image   – transformed tensor (C, H, W)
        expression – integer class label (0–7)
        va      – float tensor [valence, arousal]

    The dataset is virtually expanded by ``expand_factor`` via augmentation:
    index ``i`` maps to image ``i // expand_factor`` with augmentation
    ``i % expand_factor``.
    """

    def __init__(
        self,
        img_directory: str,
        ann_directory: str,
        base_transform: transforms.Compose | None = None,
        aug_transforms: list[transforms.Compose] | None = None,
        expand_factor: int = 3,
    ) -> None:
        self.img_directory = img_directory
        self.ann_directory = ann_directory
        self.base_transform = base_transform
        self.aug_transforms = aug_transforms or []
        self.expand_factor = expand_factor

        # Collect sample IDs from annotation file names (*_exp.npy)
        self.indices: list[str] = sorted(
            [f.split("_")[0] for f in os.listdir(ann_directory) if f.endswith("_exp.npy")],
            key=lambda x: int(x),
        )

    def __len__(self) -> int:
        return len(self.indices) * self.expand_factor

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int, torch.Tensor]:
        real_idx = idx // self.expand_factor
        aug_idx = idx % self.expand_factor
        img_id = self.indices[real_idx]

        image = Image.open(os.path.join(self.img_directory, f"{img_id}.jpg")).convert("RGB")

        expression = int(np.load(os.path.join(self.ann_directory, f"{img_id}_exp.npy")))
        valence = float(np.load(os.path.join(self.ann_directory, f"{img_id}_val.npy")))
        arousal = float(np.load(os.path.join(self.ann_directory, f"{img_id}_aro.npy")))

        if self.aug_transforms and aug_idx < len(self.aug_transforms):
            image = self.aug_transforms[aug_idx](image)
        elif self.base_transform:
            image = self.base_transform(image)

        return image, expression, torch.tensor([valence, arousal], dtype=torch.float)
