"""VGGFace2Net — VGG16-based multi-task model for FER."""

import torch
import torch.nn as nn
import torchvision.models as models


class VGGFace2Net(nn.Module):
    """Multi-task model built on a pretrained VGG16 backbone.

    Two task-specific heads are attached after the convolutional feature extractor:
        - ``exp_head``: classifies into ``num_classes`` facial expressions.
        - ``va_head``:  regresses continuous valence and arousal values.

    The backbone is initially frozen; partial unfreezing is handled externally
    by the training loop (after epoch 3, the last 10 feature layers are thawed).
    """

    def __init__(self, num_classes: int = 8) -> None:
        super().__init__()
        backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)
        # Drop the original classifier — we plug our own heads directly on features
        backbone.classifier = nn.Identity()
        self.backbone = backbone

        in_features = 25088  # VGG16 output: 512 channels × 7 × 7

        self.exp_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, num_classes),
        )
        self.va_head = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.ReLU(),
            nn.Dropout(0.6),
            nn.Linear(256, 2),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = self.backbone.features(x)
        feats = feats.view(feats.size(0), -1)
        return self.exp_head(feats), self.va_head(feats)
