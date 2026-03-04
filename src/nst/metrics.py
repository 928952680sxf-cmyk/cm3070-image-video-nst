from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim

from .losses import gram_matrix


def tensor_to_np_img(t: torch.Tensor) -> np.ndarray:
    arr = t.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    return np.clip(arr, 0.0, 1.0)


def compute_metrics(
    stylized: torch.Tensor,
    content: torch.Tensor,
    style: torch.Tensor,
    extractor,
    content_layer: str,
    style_layers: list[str],
) -> dict[str, float]:
    with torch.no_grad():
        feat_sty = extractor(stylized)
        feat_style = extractor(style)

    style_dist = 0.0
    for layer in style_layers:
        gs = gram_matrix(feat_sty[layer])
        gt = gram_matrix(feat_style[layer])
        style_dist += torch.mean((gs - gt) ** 2).item()
    style_dist /= len(style_layers)

    content_l2 = torch.mean((stylized - content) ** 2).item()

    s_img = tensor_to_np_img(stylized)
    c_img = tensor_to_np_img(content)
    ssim_score = ssim(c_img, s_img, channel_axis=2, data_range=1.0)

    return {
        "pixel_content_l2": float(content_l2),
        "style_gram_mse": float(style_dist),
        "content_ssim": float(ssim_score),
    }
