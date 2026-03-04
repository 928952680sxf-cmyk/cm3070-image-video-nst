from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms


def load_image(path: str, device: str, max_size: int | None = None) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if max_size is not None:
        w, h = img.size
        scale = max_size / max(w, h)
        if scale < 1.0:
            img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)

    tensor = transforms.ToTensor()(img).unsqueeze(0).to(device)
    return tensor


def resize_like(image: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if image.shape[-2:] == target.shape[-2:]:
        return image
    return torch.nn.functional.interpolate(
        image,
        size=target.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )


def save_image(tensor: torch.Tensor, path: str) -> None:
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    clamped = tensor.detach().cpu().squeeze(0).clamp(0, 1)
    img = transforms.ToPILImage()(clamped)
    img.save(path_obj)
