from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torchvision.models import (
    VGG16_Weights,
    VGG19_Weights,
    vgg16,
    vgg19,
)


VGG_LAYER_MAP = {
    "vgg19": {
        "conv1_1": 0,
        "conv2_1": 5,
        "conv3_1": 10,
        "conv4_1": 19,
        "conv4_2": 21,
        "conv5_1": 28,
    },
    "vgg16": {
        "conv1_1": 0,
        "conv2_1": 5,
        "conv3_1": 10,
        "conv4_1": 17,
        "conv4_2": 19,
        "conv5_1": 24,
    },
}


@dataclass
class FeatureExtractorConfig:
    architecture: str
    content_layer: str
    style_layers: list[str]


class VGGFeatureExtractor(nn.Module):
    def __init__(self, cfg: FeatureExtractorConfig) -> None:
        super().__init__()
        arch = cfg.architecture.lower()
        if arch not in VGG_LAYER_MAP:
            raise ValueError(f"Unsupported architecture: {cfg.architecture}")

        if arch == "vgg19":
            model = vgg19(weights=VGG19_Weights.DEFAULT).features
        else:
            model = vgg16(weights=VGG16_Weights.DEFAULT).features

        self.features = model.eval()
        for param in self.features.parameters():
            param.requires_grad_(False)

        self.arch = arch
        self.content_layer = cfg.content_layer
        self.style_layers = cfg.style_layers

        self.layer_map = VGG_LAYER_MAP[arch]
        self.capture_layers = set([self.content_layer, *self.style_layers])

        missing = [l for l in self.capture_layers if l not in self.layer_map]
        if missing:
            raise ValueError(f"Invalid layers for {arch}: {missing}")

        self.register_buffer(
            "mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
            persistent=False,
        )

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.normalize(x)
        outputs: dict[str, torch.Tensor] = {}
        inv_layer_map = {idx: name for name, idx in self.layer_map.items()}

        for idx, layer in enumerate(self.features):
            x = layer(x)
            if idx in inv_layer_map:
                name = inv_layer_map[idx]
                if name in self.capture_layers:
                    outputs[name] = x

        return outputs
