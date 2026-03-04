from __future__ import annotations

import torch
import torch.nn.functional as F


def gram_matrix(feat: torch.Tensor) -> torch.Tensor:
    b, c, h, w = feat.shape
    f = feat.view(b, c, h * w)
    gram = torch.bmm(f, f.transpose(1, 2))
    return gram / (c * h * w)


def content_loss(gen: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(gen, target)


def style_loss(
    gen_feats: dict[str, torch.Tensor],
    style_targets: dict[str, torch.Tensor],
    style_layers: list[str],
) -> torch.Tensor:
    loss = torch.tensor(0.0, device=next(iter(gen_feats.values())).device)
    for layer in style_layers:
        loss = loss + F.mse_loss(gram_matrix(gen_feats[layer]), style_targets[layer])
    return loss / len(style_layers)


def total_variation_loss(img: torch.Tensor) -> torch.Tensor:
    diff_h = torch.mean(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    diff_w = torch.mean(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return diff_h + diff_w
