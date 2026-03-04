from __future__ import annotations

import time
from dataclasses import dataclass

import torch
from tqdm import trange

from .losses import content_loss, gram_matrix, style_loss, total_variation_loss


@dataclass
class TransferConfig:
    steps: int
    optimizer: str
    lr: float
    content_weight: float
    style_weight: float
    tv_weight: float
    log_interval: int
    init: str


def _build_targets(extractor, content_img, style_img, content_layer, style_layers):
    with torch.no_grad():
        content_feats = extractor(content_img)
        style_feats = extractor(style_img)

    content_target = content_feats[content_layer].detach()
    style_targets = {layer: gram_matrix(style_feats[layer]).detach() for layer in style_layers}
    return content_target, style_targets


def run_style_transfer(
    extractor,
    content_img: torch.Tensor,
    style_img: torch.Tensor,
    content_layer: str,
    style_layers: list[str],
    cfg: TransferConfig,
    init_image: torch.Tensor | None = None,
) -> tuple[torch.Tensor, list[dict], float]:
    if cfg.init == "content":
        generated = content_img.clone().contiguous().requires_grad_(True)
    elif cfg.init == "noise":
        generated = torch.randn_like(content_img).contiguous().requires_grad_(True)
    elif cfg.init == "tensor":
        if init_image is None:
            raise ValueError("init='tensor' requires init_image")
        generated = init_image.clone().detach().contiguous().requires_grad_(True)
    else:
        raise ValueError(f"Unsupported init: {cfg.init}")

    content_target, style_targets = _build_targets(
        extractor, content_img, style_img, content_layer, style_layers
    )

    optimizer_name = cfg.optimizer.lower()
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam([generated], lr=cfg.lr)
    elif optimizer_name == "lbfgs":
        optimizer = torch.optim.LBFGS([generated], lr=cfg.lr, max_iter=1)
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

    history: list[dict] = []
    start = time.time()

    if optimizer_name == "adam":
        for step in trange(1, cfg.steps + 1, desc="NST(Adam)"):
            optimizer.zero_grad(set_to_none=True)
            gen_feats = extractor(generated.clamp(0, 1))
            c_loss = content_loss(gen_feats[content_layer], content_target)
            s_loss = style_loss(gen_feats, style_targets, style_layers)
            tv_loss = total_variation_loss(generated)
            total = (
                cfg.content_weight * c_loss
                + cfg.style_weight * s_loss
                + cfg.tv_weight * tv_loss
            )
            total.backward()
            optimizer.step()

            if step % cfg.log_interval == 0 or step == 1 or step == cfg.steps:
                history.append(
                    {
                        "step": step,
                        "content_loss": float(c_loss.item()),
                        "style_loss": float(s_loss.item()),
                        "tv_loss": float(tv_loss.item()),
                        "total_loss": float(total.item()),
                    }
                )
    else:
        progress = trange(1, cfg.steps + 1, desc="NST(LBFGS)")
        for step in progress:
            losses: dict[str, torch.Tensor] = {}

            def closure():
                optimizer.zero_grad(set_to_none=True)
                gen_feats = extractor(generated.clamp(0, 1))
                c_loss = content_loss(gen_feats[content_layer], content_target)
                s_loss = style_loss(gen_feats, style_targets, style_layers)
                tv_loss = total_variation_loss(generated)
                total = (
                    cfg.content_weight * c_loss
                    + cfg.style_weight * s_loss
                    + cfg.tv_weight * tv_loss
                )
                total.backward()
                losses["content"] = c_loss.detach()
                losses["style"] = s_loss.detach()
                losses["tv"] = tv_loss.detach()
                losses["total"] = total.detach()
                return total

            optimizer.step(closure)

            if step % cfg.log_interval == 0 or step == 1 or step == cfg.steps:
                history.append(
                    {
                        "step": step,
                        "content_loss": float(losses["content"].item()),
                        "style_loss": float(losses["style"].item()),
                        "tv_loss": float(losses["tv"].item()),
                        "total_loss": float(losses["total"].item()),
                    }
                )

    elapsed = time.time() - start
    return generated.detach().clamp(0, 1), history, elapsed
