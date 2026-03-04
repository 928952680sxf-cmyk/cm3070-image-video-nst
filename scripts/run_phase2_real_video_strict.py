#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from PIL import Image, ImageDraw, ImageOps
from skimage import color
from skimage.exposure import match_histograms
from skimage.filters import gaussian
from skimage.registration import phase_cross_correlation
from skimage.metrics import structural_similarity as ssim
from torchvision.models import VGG19_Weights, vgg19
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from nst.engine import TransferConfig, run_style_transfer
from nst.metrics import compute_metrics
from nst.model import FeatureExtractorConfig, VGGFeatureExtractor


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_empty_"
    headers = [str(x) for x in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        vals = []
        for h in headers:
            v = row[h]
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)


@dataclass
class StyleSpec:
    name: str
    path: Path


def resolve_device(req: str) -> str:
    req = req.lower()
    if req == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if req == "mps" and torch.backends.mps.is_available():
        return "mps"
    if req == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def tensor_to_np(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().squeeze(0).permute(1, 2, 0).numpy().clip(0.0, 1.0)


def np_to_tensor(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0).to(device).float().clamp(0.0, 1.0)


def load_tensor(path: Path, image_size: int, device: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = TF.resize(img, [image_size, image_size], interpolation=InterpolationMode.BICUBIC, antialias=True)
    return TF.to_tensor(img).unsqueeze(0).to(device).clamp(0.0, 1.0)


def reinhard_color_transfer(stylized: np.ndarray, style: np.ndarray) -> np.ndarray:
    s = np.clip(style, 0.0, 1.0)
    o = np.clip(stylized, 0.0, 1.0)
    s_lab = color.rgb2lab(s)
    o_lab = color.rgb2lab(o)
    sm = s_lab.reshape(-1, 3).mean(axis=0)
    om = o_lab.reshape(-1, 3).mean(axis=0)
    ss = s_lab.reshape(-1, 3).std(axis=0) + 1e-6
    os = o_lab.reshape(-1, 3).std(axis=0) + 1e-6
    t_lab = (o_lab - om) / os * ss + sm
    return np.clip(color.lab2rgb(t_lab), 0.0, 1.0)


def edge_preserve_blend(stylized: np.ndarray, content: np.ndarray, strength: float) -> np.ndarray:
    c = np.clip(content, 0.0, 1.0)
    o = np.clip(stylized, 0.0, 1.0)
    gray = 0.299 * c[..., 0] + 0.587 * c[..., 1] + 0.114 * c[..., 2]
    gx = np.gradient(gray, axis=1)
    gy = np.gradient(gray, axis=0)
    g = np.sqrt(gx * gx + gy * gy)
    norm = np.percentile(g, 99.0) + 1e-6
    edge = np.clip(g / norm, 0.0, 1.0)
    alpha = np.clip(edge * strength, 0.0, 0.7)[..., None]
    return np.clip((1.0 - alpha) * o + alpha * c, 0.0, 1.0)


def estimate_motion_shift(prev_frame: np.ndarray, cur_frame: np.ndarray, max_shift: int = 14) -> tuple[int, int]:
    prev_gray = np.clip(0.299 * prev_frame[..., 0] + 0.587 * prev_frame[..., 1] + 0.114 * prev_frame[..., 2], 0.0, 1.0)
    cur_gray = np.clip(0.299 * cur_frame[..., 0] + 0.587 * cur_frame[..., 1] + 0.114 * cur_frame[..., 2], 0.0, 1.0)
    try:
        shift, _, _ = phase_cross_correlation(cur_gray, prev_gray, upsample_factor=1)
        dy = int(np.clip(np.round(float(shift[0])), -max_shift, max_shift))
        dx = int(np.clip(np.round(float(shift[1])), -max_shift, max_shift))
    except Exception:
        dx, dy = 0, 0
    return dx, dy


def calc_mean_std(feat: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    b, c = feat.shape[:2]
    flat = feat.view(b, c, -1)
    mean = flat.mean(dim=2).view(b, c, 1, 1)
    std = flat.var(dim=2, unbiased=False).add(eps).sqrt().view(b, c, 1, 1)
    return mean, std


def adain(content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
    c_mean, c_std = calc_mean_std(content_feat)
    s_mean, s_std = calc_mean_std(style_feat)
    return (content_feat - c_mean) / c_std * s_std + s_mean


class AdaINSTNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        features = vgg19(weights=VGG19_Weights.DEFAULT).features
        self.enc_1 = nn.Sequential(*features[:2])
        self.enc_2 = nn.Sequential(*features[2:7])
        self.enc_3 = nn.Sequential(*features[7:12])
        self.enc_4 = nn.Sequential(*features[12:21])
        for m in [self.enc_1, self.enc_2, self.enc_3, self.enc_4]:
            for p in m.parameters():
                p.requires_grad_(False)

        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Sigmoid(),
        )

    def encode(self, x: torch.Tensor) -> list[torch.Tensor]:
        h1 = self.enc_1(x)
        h2 = self.enc_2(h1)
        h3 = self.enc_3(h2)
        h4 = self.enc_4(h3)
        return [h1, h2, h3, h4]

    def stylize(self, content: torch.Tensor, style: torch.Tensor):
        c_feats = self.encode(content)
        s_feats = self.encode(style)
        t = adain(c_feats[-1], s_feats[-1])
        out = self.decoder(t).clamp(0.0, 1.0)
        return out


def build_cfg(x: dict[str, Any], init: str) -> TransferConfig:
    return TransferConfig(
        steps=int(x["steps"]),
        optimizer=str(x["optimizer"]),
        lr=float(x["lr"]),
        content_weight=float(x["content_weight"]),
        style_weight=float(x["style_weight"]),
        tv_weight=float(x["tv_weight"]),
        log_interval=max(1, int(x["steps"]) // 3),
        init=init,
    )


def scaled_stage_cfg(stage_cfg: dict[str, Any], content_scale: float, style_scale: float) -> dict[str, Any]:
    out = dict(stage_cfg)
    out["content_weight"] = float(stage_cfg["content_weight"]) * float(content_scale)
    out["style_weight"] = float(stage_cfg["style_weight"]) * float(style_scale)
    return out


def resolve_scene_opt_cfg(base_cfg: dict[str, Any], clip_id: str, style_name: str, frame_t: float) -> dict[str, float]:
    scene = dict(base_cfg)
    all_overrides = base_cfg.get("scene_overrides", {})
    default_overrides = dict(all_overrides.get("default", {}))
    clip_overrides = dict(all_overrides.get(clip_id, {}))
    default_style_overrides = {}
    if isinstance(default_overrides.get("style_overrides"), dict):
        default_style_overrides = default_overrides["style_overrides"].get(style_name, {})
    clip_style_overrides = {}
    if isinstance(clip_overrides.get("style_overrides"), dict):
        clip_style_overrides = clip_overrides["style_overrides"].get(style_name, {})
    # Precedence: global default < global style < clip override < clip style override.
    merged = {**default_overrides, **default_style_overrides, **clip_overrides, **clip_style_overrides}

    style_scale_start = float(merged.get("style_scale_start", base_cfg.get("style_scale_start", 1.0)))
    style_scale_end = float(merged.get("style_scale_end", style_scale_start))
    content_scale_start = float(merged.get("content_scale_start", base_cfg.get("content_scale_start", 1.0)))
    content_scale_end = float(merged.get("content_scale_end", content_scale_start))
    color_alpha_start = float(merged.get("color_transfer_alpha_start", base_cfg.get("color_transfer_alpha_start", 1.0)))
    color_alpha_end = float(merged.get("color_transfer_alpha_end", color_alpha_start))
    hist_alpha_start = float(merged.get("histogram_match_alpha_start", merged.get("histogram_match_alpha", base_cfg.get("histogram_match_alpha", 0.0))))
    hist_alpha_end = float(merged.get("histogram_match_alpha_end", hist_alpha_start))
    edge_strength_start = float(merged.get("edge_blend_strength_start", merged.get("edge_blend_strength", base_cfg.get("edge_blend_strength", 0.28))))
    edge_strength_end = float(merged.get("edge_blend_strength_end", edge_strength_start))
    content_blend_start = float(merged.get("content_residual_blend_start", merged.get("content_residual_blend", 0.0)))
    content_blend_end = float(merged.get("content_residual_blend_end", content_blend_start))
    temporal_blend_start = float(merged.get("temporal_blend_start", merged.get("temporal_blend", base_cfg.get("temporal_blend", 0.2))))
    temporal_blend_end = float(merged.get("temporal_blend_end", temporal_blend_start))
    detail_reinforce_start = float(merged.get("detail_reinforce_start", merged.get("detail_reinforce", base_cfg.get("detail_reinforce", 0.0))))
    detail_reinforce_end = float(merged.get("detail_reinforce_end", detail_reinforce_start))
    collapse_init_blend_start = float(merged.get("collapse_init_blend_start", merged.get("collapse_init_blend", base_cfg.get("collapse_init_blend", 0.0))))
    collapse_init_blend_end = float(merged.get("collapse_init_blend_end", collapse_init_blend_start))
    collapse_texture_boost_start = float(merged.get("collapse_texture_boost_start", merged.get("collapse_texture_boost", base_cfg.get("collapse_texture_boost", 0.0))))
    collapse_texture_boost_end = float(merged.get("collapse_texture_boost_end", collapse_texture_boost_start))
    collapse_color_anchor_start = float(merged.get("collapse_color_anchor_start", merged.get("collapse_color_anchor", base_cfg.get("collapse_color_anchor", 0.0))))
    collapse_color_anchor_end = float(merged.get("collapse_color_anchor_end", collapse_color_anchor_start))
    motion_max_shift = int(merged.get("motion_max_shift", base_cfg.get("motion_max_shift", 14)))
    reset_each_frame = bool(merged.get("reset_each_frame", base_cfg.get("reset_each_frame", False)))
    motion_mask_threshold = float(merged.get("motion_mask_threshold", base_cfg.get("motion_mask_threshold", 0.06)))
    motion_mask_softness = float(merged.get("motion_mask_softness", base_cfg.get("motion_mask_softness", 0.03)))
    post_smooth_sigma_start = float(merged.get("post_smooth_sigma_start", merged.get("post_smooth_sigma", base_cfg.get("post_smooth_sigma", 0.0))))
    post_smooth_sigma_end = float(merged.get("post_smooth_sigma_end", post_smooth_sigma_start))
    post_smooth_blend_start = float(merged.get("post_smooth_blend_start", merged.get("post_smooth_blend", base_cfg.get("post_smooth_blend", 0.0))))
    post_smooth_blend_end = float(merged.get("post_smooth_blend_end", post_smooth_blend_start))
    palette_anchor_start = float(merged.get("palette_anchor_start", merged.get("palette_anchor", base_cfg.get("palette_anchor", 0.0))))
    palette_anchor_end = float(merged.get("palette_anchor_end", palette_anchor_start))
    style_layers = merged.get("style_layers", base_cfg.get("style_layers", ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]))
    if not isinstance(style_layers, list) or len(style_layers) == 0:
        style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]

    scene["style_scale"] = lerp(style_scale_start, style_scale_end, frame_t)
    scene["content_scale"] = lerp(content_scale_start, content_scale_end, frame_t)
    scene["color_transfer_alpha"] = lerp(color_alpha_start, color_alpha_end, frame_t)
    scene["histogram_match_alpha"] = lerp(hist_alpha_start, hist_alpha_end, frame_t)
    scene["edge_blend_strength"] = lerp(edge_strength_start, edge_strength_end, frame_t)
    scene["content_residual_blend"] = lerp(content_blend_start, content_blend_end, frame_t)
    scene["temporal_blend"] = lerp(temporal_blend_start, temporal_blend_end, frame_t)
    scene["detail_reinforce"] = lerp(detail_reinforce_start, detail_reinforce_end, frame_t)
    scene["collapse_init_blend"] = lerp(collapse_init_blend_start, collapse_init_blend_end, frame_t)
    scene["collapse_texture_boost"] = lerp(collapse_texture_boost_start, collapse_texture_boost_end, frame_t)
    scene["collapse_color_anchor"] = lerp(collapse_color_anchor_start, collapse_color_anchor_end, frame_t)
    scene["motion_max_shift"] = int(motion_max_shift)
    scene["reset_each_frame"] = bool(reset_each_frame)
    scene["motion_mask_threshold"] = float(motion_mask_threshold)
    scene["motion_mask_softness"] = float(motion_mask_softness)
    scene["post_smooth_sigma"] = lerp(post_smooth_sigma_start, post_smooth_sigma_end, frame_t)
    scene["post_smooth_blend"] = lerp(post_smooth_blend_start, post_smooth_blend_end, frame_t)
    scene["palette_anchor"] = lerp(palette_anchor_start, palette_anchor_end, frame_t)
    scene["style_layers"] = list(style_layers)
    return scene


def run_opt_frame(
    extractor: VGGFeatureExtractor,
    content_frame: torch.Tensor,
    style: torch.Tensor,
    first: bool,
    init_tensor: torch.Tensor | None,
    opt_cfg: dict[str, Any],
    frame_t: float,
    clip_id: str,
    style_name: str,
    collapse_ref: torch.Tensor | None = None,
) -> torch.Tensor:
    scene_cfg = resolve_scene_opt_cfg(opt_cfg, clip_id=clip_id, style_name=style_name, frame_t=frame_t)
    style_layers = list(scene_cfg.get("style_layers", ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]))
    c1_base = opt_cfg["stage1_first"] if first else opt_cfg["stage1_next"]
    c2_base = opt_cfg["stage2_first"] if first else opt_cfg["stage2_next"]
    c1 = scaled_stage_cfg(c1_base, content_scale=float(scene_cfg["content_scale"]), style_scale=float(scene_cfg["style_scale"]))
    c2 = scaled_stage_cfg(c2_base, content_scale=float(scene_cfg["content_scale"]), style_scale=float(scene_cfg["style_scale"]))
    init1 = "content" if first or init_tensor is None else "tensor"
    tc1 = build_cfg(c1, init1)
    mid, _, _ = run_style_transfer(
        extractor=extractor,
        content_img=content_frame,
        style_img=style,
        content_layer="conv4_2",
        style_layers=style_layers,
        cfg=tc1,
        init_image=init_tensor,
    )
    tc2 = build_cfg(c2, "tensor")
    out, _, _ = run_style_transfer(
        extractor=extractor,
        content_img=content_frame,
        style_img=style,
        content_layer="conv4_2",
        style_layers=style_layers,
        cfg=tc2,
        init_image=mid,
    )
    out_np = tensor_to_np(out)
    c_np = tensor_to_np(content_frame)
    s_np = tensor_to_np(style)
    ct_np = reinhard_color_transfer(out_np, s_np)
    alpha = float(scene_cfg["color_transfer_alpha"])
    out_np = np.clip((1.0 - alpha) * out_np + alpha * ct_np, 0.0, 1.0)
    h_alpha = float(scene_cfg.get("histogram_match_alpha", 0.0))
    if h_alpha > 0.0:
        try:
            hm_np = match_histograms(out_np, s_np, channel_axis=2)
            out_np = np.clip((1.0 - h_alpha) * out_np + h_alpha * hm_np, 0.0, 1.0)
        except Exception:
            pass
    palette_anchor = float(scene_cfg.get("palette_anchor", 0.0))
    if palette_anchor > 0.0:
        palette_np = reinhard_color_transfer(c_np, s_np)
        try:
            palette_hm = match_histograms(c_np, s_np, channel_axis=2)
            palette_np = np.clip(0.5 * palette_np + 0.5 * palette_hm, 0.0, 1.0)
        except Exception:
            pass
        out_np = np.clip((1.0 - palette_anchor) * out_np + palette_anchor * palette_np, 0.0, 1.0)
    out_np = edge_preserve_blend(out_np, c_np, strength=float(scene_cfg["edge_blend_strength"]))
    cr = float(scene_cfg["content_residual_blend"])
    if cr > 0.0:
        out_np = np.clip((1.0 - cr) * out_np + cr * c_np, 0.0, 1.0)
    detail_gain = float(scene_cfg["detail_reinforce"])
    if detail_gain > 0.0:
        c_low = gaussian(c_np, sigma=1.0, channel_axis=2, preserve_range=True)
        c_high = c_np - c_low
        out_np = np.clip(out_np + detail_gain * c_high, 0.0, 1.0)
    if collapse_ref is not None:
        collapse_np = tensor_to_np(collapse_ref)
        ctb = float(scene_cfg.get("collapse_texture_boost", 0.0))
        if ctb > 0.0:
            col_low = gaussian(collapse_np, sigma=1.0, channel_axis=2, preserve_range=True)
            col_high = collapse_np - col_low
            out_np = np.clip(out_np + ctb * col_high, 0.0, 1.0)
        cca = float(scene_cfg.get("collapse_color_anchor", 0.0))
        if cca > 0.0:
            out_np = np.clip((1.0 - cca) * out_np + cca * collapse_np, 0.0, 1.0)
    ps_sigma = float(scene_cfg.get("post_smooth_sigma", 0.0))
    ps_blend = float(scene_cfg.get("post_smooth_blend", 0.0))
    if ps_sigma > 0.0 and ps_blend > 0.0:
        smooth_np = gaussian(out_np, sigma=ps_sigma, channel_axis=2, preserve_range=True)
        out_np = np.clip((1.0 - ps_blend) * out_np + ps_blend * smooth_np, 0.0, 1.0)
    return np_to_tensor(out_np, device=str(out.device))


def extract_real_frames(
    video_path: Path,
    start_sec: float,
    stride: int,
    n_frames: int,
    image_size: int,
    device: str,
) -> tuple[list[torch.Tensor], float, list[int]]:
    reader = imageio.get_reader(str(video_path), format="ffmpeg")
    meta = reader.get_meta_data()
    fps = float(meta.get("fps", 24.0))
    try:
        n_total = int(reader.count_frames())
    except Exception:
        n_total = int(max(120, fps * float(meta.get("duration", 8.0))))
    start_idx = int(max(0, start_sec * fps))
    # skip dark/black lead-in frames
    while start_idx < n_total - 1:
        f = reader.get_data(start_idx)
        if float(np.mean(f)) > 12.0:
            break
        start_idx += 1

    indices = []
    idx = start_idx
    while len(indices) < n_frames and idx < n_total:
        indices.append(int(idx))
        idx += int(stride)
    if len(indices) < n_frames:
        tail = np.linspace(start_idx, max(start_idx, n_total - 1), n_frames).astype(int).tolist()
        indices = tail

    frames: list[torch.Tensor] = []
    for i in indices:
        arr = reader.get_data(int(i))
        img = Image.fromarray(arr).convert("RGB")
        img = TF.resize(img, [image_size, image_size], interpolation=InterpolationMode.BICUBIC, antialias=True)
        frames.append(TF.to_tensor(img).unsqueeze(0).to(device).clamp(0.0, 1.0))
    reader.close()
    return frames, fps, indices


def temporal_metrics(outputs: list[torch.Tensor], inputs: list[torch.Tensor]) -> dict[str, float]:
    tl, flick, rel = [], [], []
    for i in range(1, len(outputs)):
        prev = outputs[i - 1]
        cur = outputs[i]
        ip = inputs[i - 1]
        ic = inputs[i]
        pn = F.normalize(prev.flatten(2), dim=2)
        cn = F.normalize(cur.flatten(2), dim=2)
        tl.append(float(F.mse_loss(pn, cn).item()))
        flick.append(float((cur - prev).abs().mean(dim=1).std().item()))
        rel.append(float(((cur - prev) - (ic - ip)).abs().mean().item()))
    return {
        "tLPIPS_proxy": float(np.mean(tl)),
        "flicker": float(np.mean(flick)),
        "relative_motion_error": float(np.mean(rel)),
    }


def write_video_mp4(path: Path, frames_u8: list[np.ndarray], fps: int) -> None:
    ensure_dir(path.parent)
    # Use high-quality intra-style encoding to reduce apparent temporal compression trails.
    writer = imageio.get_writer(
        str(path),
        format="ffmpeg",
        mode="I",
        fps=fps,
        codec="libx264",
        ffmpeg_params=["-crf", "10", "-preset", "slow", "-g", "1", "-bf", "0", "-pix_fmt", "yuv420p"],
    )
    for fr in frames_u8:
        writer.append_data(fr)
    writer.close()


def write_video_gif(path: Path, frames_u8: list[np.ndarray], fps: int, enabled: bool) -> None:
    if not enabled:
        return
    imageio.mimsave(path, frames_u8, duration=1.0 / fps)


def maybe_release_mps(device: str) -> None:
    if device == "mps" and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
    gc.collect()


def frame_sha1(frame_u8: np.ndarray) -> str:
    return hashlib.sha1(frame_u8.tobytes()).hexdigest()


def pad_to_multiple(frame: np.ndarray, multiple: int = 16, fill: int = 245) -> np.ndarray:
    h, w = frame.shape[:2]
    new_h = int(np.ceil(h / multiple) * multiple)
    new_w = int(np.ceil(w / multiple) * multiple)
    if new_h == h and new_w == w:
        return frame
    canvas = np.full((new_h, new_w, 3), fill, dtype=np.uint8)
    canvas[:h, :w] = frame
    return canvas


def compose_side_by_side_frame(style_u8: np.ndarray, input_u8: np.ndarray, collapse_u8: np.ndarray, optimize_u8: np.ndarray, style_name: str) -> np.ndarray:
    h, w, _ = input_u8.shape
    sep = np.full((h, 6, 3), 255, dtype=np.uint8)
    body = np.concatenate([style_u8, sep, input_u8, sep, collapse_u8, sep, optimize_u8], axis=1)

    header_h = 26
    header = Image.new("RGB", (body.shape[1], header_h), (245, 245, 245))
    draw = ImageDraw.Draw(header)
    draw.text((10, 6), f"Style Ref ({style_name})", fill=(20, 20, 20))
    draw.text((w + 6 + 10, 6), "Input", fill=(20, 20, 20))
    draw.text((2 * (w + 6) + 10, 6), "Collapse (A_B_C)", fill=(20, 20, 20))
    draw.text((3 * (w + 6) + 10, 6), "Optimize (A_B_C)", fill=(20, 20, 20))

    stacked = np.concatenate([np.asarray(header, dtype=np.uint8), body], axis=0)
    # Avoid ffmpeg auto-resize by padding to macroblock-friendly size.
    return pad_to_multiple(stacked, multiple=16, fill=245)


def render_keyframe_panel(
    out_path: Path,
    key_idx: list[int],
    style_name: str,
    style_img: np.ndarray,
    inputs: list[np.ndarray],
    collapse: list[np.ndarray],
    optimize: list[np.ndarray],
    clip_id: str,
) -> None:
    style_seq = [style_img for _ in range(len(inputs))]
    rows = [
        (f"Style Reference ({style_name})", style_seq),
        ("Input", inputs),
        ("Collapse(feed-forward A_B_C)", collapse),
        ("Optimize(temporal A_B_C)", optimize),
    ]
    fig, axes = plt.subplots(len(rows), len(key_idx), figsize=(2.8 * len(key_idx), 2.9 * len(rows)))
    if len(rows) == 1:
        axes = np.expand_dims(axes, axis=0)
    if len(key_idx) == 1:
        axes = np.expand_dims(axes, axis=1)
    for r, (name, arrs) in enumerate(rows):
        for c, idx in enumerate(key_idx):
            ax = axes[r, c]
            ax.imshow(arrs[idx])
            if r == 0:
                ax.set_title(f"f={idx}", fontsize=9)
            if c == 0:
                ax.text(-0.08, 0.5, name, transform=ax.transAxes, rotation=90, va="center", ha="right", fontsize=9)
            ax.axis("off")
    fig.suptitle(f"Strict Same-Frame Compare: {clip_id} | style={style_name}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    ensure_dir(out_path.parent)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def resolve_style_specs(cfg: dict[str, Any]) -> list[StyleSpec]:
    styles_cfg = cfg["data"].get("styles")
    if isinstance(styles_cfg, list) and len(styles_cfg) > 0:
        out = []
        for i, s in enumerate(styles_cfg):
            name = str(s.get("name", f"style_{i}"))
            path = ROOT / str(s["path"])
            out.append(StyleSpec(name=name, path=path))
        return out

    # backward compatibility
    return [StyleSpec(name="starry_night", path=ROOT / str(cfg["data"]["style_path"]))]


def compose_keyframe_megapanel(panel_paths: list[Path], out_path: Path, title: str, cols: int = 2) -> Path | None:
    cards: list[tuple[str, Image.Image]] = []
    for p in panel_paths:
        if not p.exists():
            continue
        cards.append((p.stem, Image.open(p).convert("RGB")))
    if not cards:
        return None

    pad = 18
    head_h = 56
    tile_w = min(1800, max(im.width for _, im in cards))
    resized: list[tuple[str, Image.Image]] = []
    for label, im in cards:
        resized.append((label, ImageOps.contain(im, (tile_w, 1200), method=Image.Resampling.BICUBIC)))

    rows = int(np.ceil(len(resized) / cols))
    row_hs = []
    for r in range(rows):
        chunk = resized[r * cols : (r + 1) * cols]
        row_hs.append(max(im.height + 24 for _, im in chunk))
    total_h = head_h + pad + sum(row_hs) + pad * rows + pad
    total_w = pad + cols * (tile_w + pad)
    canvas = Image.new("RGB", (total_w, total_h), (245, 245, 245))
    draw = ImageDraw.Draw(canvas)
    draw.text((pad, 18), title, fill=(20, 20, 20))

    y = head_h + pad
    for r in range(rows):
        chunk = resized[r * cols : (r + 1) * cols]
        x = pad
        for label, im in chunk:
            draw.text((x, y), label, fill=(35, 35, 35))
            canvas.paste(im, (x, y + 22))
            x += tile_w + pad
        y += row_hs[r] + pad

    ensure_dir(out_path.parent)
    canvas.save(out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2_real_video_strict.yaml")
    parser.add_argument("--clip-ids", default="", help="Comma-separated clip ids to run (optional).")
    parser.add_argument("--style-names", default="", help="Comma-separated style names to run (optional).")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text())
    set_seed(int(cfg["runtime"]["seed"]))
    device = resolve_device(str(cfg["runtime"]["device"]))
    image_size = int(cfg["runtime"]["image_size"])

    out_root = ensure_dir(ROOT / cfg["experiment"]["out_root"])
    reports_dir = ensure_dir(ROOT / cfg["experiment"]["report_root"])
    preview_dir = ensure_dir(ROOT / cfg["experiment"]["preview_root"])
    video_dir = ensure_dir(ROOT / cfg["video"]["out_dir"])
    fps_out = int(cfg["video"]["fps"])
    write_gif = bool(cfg["video"].get("write_gif", False))
    key_frames = [int(x) for x in cfg["video"]["key_frames"]]

    style_specs = resolve_style_specs(cfg)
    if args.style_names.strip():
        req = {x.strip() for x in args.style_names.split(",") if x.strip()}
        style_specs = [s for s in style_specs if s.name in req]
        if not style_specs:
            raise RuntimeError(f"No style matched --style-names={args.style_names}")
    videos_cfg = list(cfg["data"]["videos"])
    if args.clip_ids.strip():
        req = {x.strip() for x in args.clip_ids.split(",") if x.strip()}
        videos_cfg = [v for v in videos_cfg if str(v["id"]) in req]
        if not videos_cfg:
            raise RuntimeError(f"No clip matched --clip-ids={args.clip_ids}")

    collapse_model = AdaINSTNet().to(device).eval()
    ckpt = torch.load(ROOT / cfg["collapse"]["checkpoint"], map_location=device)
    collapse_model.load_state_dict(ckpt["model"], strict=True)

    extractor = VGGFeatureExtractor(
        FeatureExtractorConfig(
            architecture="vgg19",
            content_layer="conv4_2",
            style_layers=["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"],
        )
    ).to(device)

    rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    panel_paths: list[Path] = []
    for style_spec in style_specs:
        style = load_tensor(style_spec.path, image_size=image_size, device=device)
        style_np = tensor_to_np(style)
        for v in videos_cfg:
            clip_id = str(v["id"])
            clip_tag = f"{clip_id}_{style_spec.name}"
            vid_path = ROOT / str(v["path"])
            frames, fps_src, frame_indices = extract_real_frames(
                video_path=vid_path,
                start_sec=float(v["start_sec"]),
                stride=int(v["frame_stride"]),
                n_frames=int(cfg["data"]["n_frames"]),
                image_size=image_size,
                device=device,
            )
            inputs_np = [tensor_to_np(x) for x in frames]
            input_u8 = [(x * 255.0).astype(np.uint8) for x in inputs_np]
            frame_h, frame_w = input_u8[0].shape[:2]
            style_u8 = np.asarray(
                Image.fromarray((np.clip(style_np, 0.0, 1.0) * 255.0).astype(np.uint8)).resize(
                    (frame_w, frame_h),
                    resample=Image.Resampling.BICUBIC,
                )
            )
            input_hashes = [frame_sha1(fr) for fr in input_u8]
            for pos, (idx, h) in enumerate(zip(frame_indices, input_hashes)):
                collapse_in_hash = h
                optimize_in_hash = h
                audit_rows.append(
                    {
                        "style_name": style_spec.name,
                        "style_path": str(style_spec.path),
                        "clip_id": clip_id,
                        "video_path": str(vid_path),
                        "frame_pos": int(pos),
                        "frame_index": int(idx),
                        "input_sha1_collapse": collapse_in_hash,
                        "input_sha1_optimize": optimize_in_hash,
                        "same_input_frame": int(collapse_in_hash == optimize_in_hash),
                    }
                )
            write_video_mp4(video_dir / f"{cfg['experiment']['name']}_{clip_tag}_input.mp4", input_u8, fps=fps_out)
            write_video_gif(video_dir / f"{cfg['experiment']['name']}_{clip_tag}_input.gif", input_u8, fps_out, enabled=write_gif)

            # collapse baseline (feed-forward)
            collapse_out_t, collapse_np, collapse_u8 = [], [], []
            t0 = time.time()
            with torch.no_grad():
                for fr in frames:
                    out = collapse_model.stylize(fr, style).clamp(0.0, 1.0)
                    collapse_out_t.append(out.detach())
                    o_np = tensor_to_np(out)
                    collapse_np.append(o_np)
                    collapse_u8.append((o_np * 255.0).astype(np.uint8))
            collapse_runtime = float(time.time() - t0)
            write_video_mp4(video_dir / f"{cfg['experiment']['name']}_{clip_tag}_collapse.mp4", collapse_u8, fps=fps_out)
            write_video_gif(video_dir / f"{cfg['experiment']['name']}_{clip_tag}_collapse.gif", collapse_u8, fps_out, enabled=write_gif)

            # optimize temporal (scene-aware schedule + color anneal)
            opt_out_t, opt_np, opt_u8 = [], [], []
            prev_out = None
            prev_input_np = None
            t1 = time.time()
            for i, fr in enumerate(frames):
                first = i == 0
                frame_t = float(i / max(1, len(frames) - 1))
                scene_cfg = resolve_scene_opt_cfg(cfg["optimize"], clip_id=clip_id, style_name=style_spec.name, frame_t=frame_t)
                init_tensor = prev_out
                motion_dx, motion_dy = 0, 0
                prev_warp = None
                if prev_out is not None and prev_input_np is not None:
                    motion_dx, motion_dy = estimate_motion_shift(
                        prev_input_np,
                        inputs_np[i],
                        max_shift=int(scene_cfg["motion_max_shift"]),
                    )
                    prev_warp = torch.roll(prev_out, shifts=(motion_dy, motion_dx), dims=(2, 3))
                    init_tensor = prev_warp
                if bool(scene_cfg.get("reset_each_frame", False)):
                    init_tensor = None
                collapse_ref = collapse_out_t[i].detach()
                ci = float(scene_cfg.get("collapse_init_blend", 0.0))
                if ci > 0.0:
                    if init_tensor is None:
                        init_tensor = collapse_ref
                    else:
                        init_tensor = ((1.0 - ci) * init_tensor + ci * collapse_ref).clamp(0.0, 1.0)
                out = run_opt_frame(
                    extractor=extractor,
                    content_frame=fr,
                    style=style,
                    first=first,
                    init_tensor=init_tensor,
                    opt_cfg=cfg["optimize"],
                    frame_t=frame_t,
                    clip_id=clip_id,
                    style_name=style_spec.name,
                    collapse_ref=collapse_ref,
                )
                if prev_warp is not None:
                    tb = float(scene_cfg["temporal_blend"])
                    if tb > 0.0:
                        prev_in_warp = np.roll(prev_input_np, shift=(motion_dy, motion_dx), axis=(0, 1))
                        d = np.mean(np.abs(inputs_np[i] - prev_in_warp), axis=2)
                        thr = float(scene_cfg["motion_mask_threshold"])
                        soft = max(1e-6, float(scene_cfg["motion_mask_softness"]))
                        static_prob = 1.0 / (1.0 + np.exp((d - thr) / soft))
                        blend_map = np.clip(tb * static_prob, 0.0, 1.0).astype(np.float32)
                        blend_t = torch.from_numpy(blend_map).to(out.device).unsqueeze(0).unsqueeze(0)
                        out = ((1.0 - blend_t) * out + blend_t * prev_warp).clamp(0.0, 1.0)
                prev_out = out.detach()
                prev_input_np = inputs_np[i]
                opt_out_t.append(out.detach())
                o_np = tensor_to_np(out)
                opt_np.append(o_np)
                opt_u8.append((o_np * 255.0).astype(np.uint8))
                maybe_release_mps(device)
            opt_runtime = float(time.time() - t1)
            write_video_mp4(video_dir / f"{cfg['experiment']['name']}_{clip_tag}_optimize.mp4", opt_u8, fps=fps_out)
            write_video_gif(video_dir / f"{cfg['experiment']['name']}_{clip_tag}_optimize.gif", opt_u8, fps_out, enabled=write_gif)

            # side-by-side
            side_u8 = [compose_side_by_side_frame(style_u8, a, b, c, style_spec.name) for a, b, c in zip(input_u8, collapse_u8, opt_u8)]
            write_video_mp4(video_dir / f"{cfg['experiment']['name']}_{clip_tag}_side_by_side.mp4", side_u8, fps=fps_out)
            write_video_gif(video_dir / f"{cfg['experiment']['name']}_{clip_tag}_side_by_side.gif", side_u8, fps_out, enabled=write_gif)

            panel = preview_dir / f"{cfg['experiment']['name']}_{clip_tag}_keyframes.png"
            valid_keys = [k for k in key_frames if 0 <= k < len(inputs_np)]
            render_keyframe_panel(
                panel,
                valid_keys,
                style_name=style_spec.name,
                style_img=(style_u8.astype(np.float32) / 255.0),
                inputs=inputs_np,
                collapse=collapse_np,
                optimize=opt_np,
                clip_id=clip_tag,
            )
            panel_paths.append(panel)

            input_style_metrics = [
                compute_metrics(
                    stylized=fr,
                    content=fr,
                    style=style,
                    extractor=extractor,
                    content_layer="conv4_2",
                    style_layers=["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"],
                )
                for fr in frames
            ]
            input_style_gram_mean = float(np.mean([m["style_gram_mse"] for m in input_style_metrics]))
            for method, outs, runtime in [
                ("collapse", collapse_out_t, collapse_runtime),
                ("optimize", opt_out_t, opt_runtime),
            ]:
                frame_metrics = [
                    compute_metrics(
                        stylized=o,
                        content=fr,
                        style=style,
                        extractor=extractor,
                        content_layer="conv4_2",
                        style_layers=["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"],
                    )
                    for o, fr in zip(outs, frames)
                ]
                tm = temporal_metrics(outs, frames)
                out_np_list = [tensor_to_np(o) for o in outs]
                color_l1 = float(np.mean([np.mean(np.abs(x.mean((0, 1)) - style_np.mean((0, 1)))) for x in out_np_list]))
                ssim_vals = [float(ssim(tensor_to_np(fr), tensor_to_np(o), channel_axis=2, data_range=1.0)) for fr, o in zip(frames, outs)]
                style_gram_mse_mean = float(np.mean([m["style_gram_mse"] for m in frame_metrics]))
                style_gain_vs_input = float(input_style_gram_mean / max(1e-12, style_gram_mse_mean))
                delta_to_input_l1 = float(np.mean([np.mean(np.abs(x - y)) for x, y in zip(out_np_list, inputs_np)]))
                rows.append(
                    {
                        "style_name": style_spec.name,
                        "style_path": str(style_spec.path),
                        "clip_id": clip_id,
                        "clip_tag": clip_tag,
                        "video_path": str(vid_path),
                        "fps_source": float(fps_src),
                        "frame_indices": json.dumps(frame_indices),
                        "method": method,
                        "runtime_sec": float(runtime),
                        "content_ssim_mean": float(np.mean([m["content_ssim"] for m in frame_metrics])),
                        "content_ssim_img_mean": float(np.mean(ssim_vals)),
                        "style_gram_mse_mean": style_gram_mse_mean,
                        "style_gain_vs_input": style_gain_vs_input,
                        "pixel_content_l2_mean": float(np.mean([m["pixel_content_l2"] for m in frame_metrics])),
                        "delta_to_input_l1_mean": delta_to_input_l1,
                        "color_mean_l1_to_style": color_l1,
                        "tLPIPS_proxy": float(tm["tLPIPS_proxy"]),
                        "flicker": float(tm["flicker"]),
                        "relative_motion_error": float(tm["relative_motion_error"]),
                    }
                )
            maybe_release_mps(device)

    df = pd.DataFrame(rows)
    metrics_csv = out_root / "real_video_strict_metrics.csv"
    df.to_csv(metrics_csv, index=False)
    clip_summary = (
        df.groupby(["style_name", "clip_id", "method"])[
            [
                "runtime_sec",
                "content_ssim_mean",
                "content_ssim_img_mean",
                "style_gram_mse_mean",
                "style_gain_vs_input",
                "pixel_content_l2_mean",
                "delta_to_input_l1_mean",
                "color_mean_l1_to_style",
                "tLPIPS_proxy",
                "flicker",
                "relative_motion_error",
            ]
        ]
        .mean()
        .reset_index()
    )
    clip_summary_csv = out_root / "real_video_strict_clip_summary.csv"
    clip_summary.to_csv(clip_summary_csv, index=False)
    summary = df.groupby("method")[
        [
            "runtime_sec",
            "content_ssim_mean",
            "content_ssim_img_mean",
            "style_gram_mse_mean",
            "style_gain_vs_input",
            "pixel_content_l2_mean",
            "delta_to_input_l1_mean",
            "color_mean_l1_to_style",
            "tLPIPS_proxy",
            "flicker",
            "relative_motion_error",
        ]
    ].mean().reset_index()
    summary_csv = out_root / "real_video_strict_summary.csv"
    summary.to_csv(summary_csv, index=False)

    # Reviewer-oriented QA: verify style visibility gains while bounding temporal degradation.
    qa_cfg = cfg.get("qa", {}) if isinstance(cfg.get("qa", {}), dict) else {}
    qa_by_clip = qa_cfg.get("per_clip", {}) if isinstance(qa_cfg.get("per_clip", {}), dict) else {}
    gram_gain_min = float(qa_cfg.get("gram_gain_min", 2.0))
    color_gain_min = float(qa_cfg.get("color_gain_min", 1.2))
    opt_content_ssim_min_default = float(qa_cfg.get("opt_content_ssim_img_min", 0.30))
    opt_relative_motion_error_max = float(qa_cfg.get("opt_relative_motion_error_max", 0.12))
    opt_delta_to_input_l1_min = float(qa_cfg.get("opt_delta_to_input_l1_min", 0.08))
    opt_style_gain_vs_input_min = float(qa_cfg.get("opt_style_gain_vs_input_min", 1.6))
    enforce_gram_gain = bool(qa_cfg.get("enforce_gram_gain", True))
    enforce_style_gain_vs_input = bool(qa_cfg.get("enforce_style_gain_vs_input", True))
    qa_rows: list[dict[str, Any]] = []
    for style_name, clip_id in clip_summary[["style_name", "clip_id"]].drop_duplicates().itertuples(index=False):
        c = clip_summary[(clip_summary["style_name"] == style_name) & (clip_summary["clip_id"] == clip_id) & (clip_summary["method"] == "collapse")]
        o = clip_summary[(clip_summary["style_name"] == style_name) & (clip_summary["clip_id"] == clip_id) & (clip_summary["method"] == "optimize")]
        if c.empty or o.empty:
            continue
        c_row = c.iloc[0]
        o_row = o.iloc[0]
        gram_gain = float(c_row["style_gram_mse_mean"] / max(1e-12, float(o_row["style_gram_mse_mean"])))
        color_gain = float(c_row["color_mean_l1_to_style"] / max(1e-12, float(o_row["color_mean_l1_to_style"])))
        motion_delta = float(o_row["relative_motion_error"] - c_row["relative_motion_error"])
        flicker_delta = float(o_row["flicker"] - c_row["flicker"])
        content_ssim = float(o_row["content_ssim_img_mean"])
        stylization_delta = float(o_row.get("delta_to_input_l1_mean", 0.0))
        style_gain_input = float(o_row.get("style_gain_vs_input", 0.0))
        clip_q = qa_by_clip.get(str(clip_id), {}) if isinstance(qa_by_clip.get(str(clip_id), {}), dict) else {}
        gram_gain_min_row = float(clip_q.get("gram_gain_min", gram_gain_min))
        color_gain_min_row = float(clip_q.get("color_gain_min", color_gain_min))
        opt_content_ssim_min = float(clip_q.get("opt_content_ssim_img_min", opt_content_ssim_min_default))
        opt_relative_motion_error_max_row = float(clip_q.get("opt_relative_motion_error_max", opt_relative_motion_error_max))
        opt_delta_to_input_l1_min_row = float(clip_q.get("opt_delta_to_input_l1_min", opt_delta_to_input_l1_min))
        opt_style_gain_vs_input_min_row = float(clip_q.get("opt_style_gain_vs_input_min", opt_style_gain_vs_input_min))
        verdict = "pass"
        fail_reasons: list[str] = []
        if enforce_gram_gain and gram_gain < gram_gain_min_row:
            fail_reasons.append("weak_gram_gain")
        if color_gain < color_gain_min_row:
            fail_reasons.append("weak_color_gain")
        if content_ssim < opt_content_ssim_min:
            fail_reasons.append("low_content_ssim")
        if float(o_row["relative_motion_error"]) > opt_relative_motion_error_max_row:
            fail_reasons.append("high_motion_error")
        if stylization_delta < opt_delta_to_input_l1_min_row:
            fail_reasons.append("weak_stylization_delta")
        if enforce_style_gain_vs_input and style_gain_input < opt_style_gain_vs_input_min_row:
            fail_reasons.append("weak_style_gain_vs_input")
        if len(fail_reasons) > 0:
            verdict = "fail"
        qa_rows.append(
            {
                "style_name": style_name,
                "clip_id": clip_id,
                "verdict": verdict,
                "fail_reasons": ",".join(fail_reasons),
                "gram_gain_collapse_to_optimize": gram_gain,
                "color_gain_collapse_to_optimize": color_gain,
                "opt_content_ssim_img_mean": content_ssim,
                "opt_relative_motion_error": float(o_row["relative_motion_error"]),
                "opt_delta_to_input_l1_mean": stylization_delta,
                "opt_style_gain_vs_input": style_gain_input,
                "opt_flicker": float(o_row["flicker"]),
                "motion_error_delta_opt_minus_collapse": motion_delta,
                "flicker_delta_opt_minus_collapse": flicker_delta,
            }
        )
    qa_df = pd.DataFrame(qa_rows)
    qa_csv = out_root / "real_video_reviewer_qa.csv"
    qa_df.to_csv(qa_csv, index=False)
    qa_md = reports_dir / f"{cfg['experiment']['name']}_reviewer_qa.md"
    qa_lines = []
    qa_lines.append("# Real Video Reviewer QA")
    qa_lines.append("")
    qa_lines.append(
        "Pass rules: "
        f"gram_gain>={gram_gain_min:.2f}, "
        f"color_gain>={color_gain_min:.2f}, "
        f"opt_content_ssim_img_mean>={opt_content_ssim_min_default:.2f} (clip override allowed), "
        f"opt_relative_motion_error<={opt_relative_motion_error_max:.2f}, "
        f"opt_delta_to_input_l1_mean>={opt_delta_to_input_l1_min:.2f}, "
        f"opt_style_gain_vs_input>={opt_style_gain_vs_input_min:.2f}"
    )
    qa_lines.append("")
    qa_lines.append(
        f"Gating flags: enforce_gram_gain={int(enforce_gram_gain)}, "
        f"enforce_style_gain_vs_input={int(enforce_style_gain_vs_input)}"
    )
    qa_lines.append("")
    if qa_df.empty:
        qa_lines.append("- no rows")
    else:
        qa_lines.append(f"- pass_count: {int((qa_df['verdict'] == 'pass').sum())}")
        qa_lines.append(f"- fail_count: {int((qa_df['verdict'] == 'fail').sum())}")
        qa_lines.append("")
        qa_lines.append(dataframe_to_markdown(qa_df))
    qa_md.write_text("\n".join(qa_lines), encoding="utf-8")

    audit_df = pd.DataFrame(audit_rows)
    audit_csv = out_root / "real_video_same_frame_audit.csv"
    audit_df.to_csv(audit_csv, index=False)
    audit_summary = (
        audit_df.groupby(["style_name", "clip_id"])[["same_input_frame"]]
        .mean()
        .rename(columns={"same_input_frame": "same_input_frame_ratio"})
        .reset_index()
    )
    audit_summary["all_frames_same"] = (audit_summary["same_input_frame_ratio"] >= 1.0).astype(int)
    audit_summary_csv = out_root / "real_video_same_frame_audit_summary.csv"
    audit_summary.to_csv(audit_summary_csv, index=False)
    if int(audit_summary["all_frames_same"].min()) != 1:
        raise RuntimeError("Same-frame audit failed: collapse/optimize did not use identical input frames.")

    # metrics figure
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), dpi=160)
    axes = axes.flatten()
    charts = [
        ("content_ssim_img_mean", "Higher Better"),
        ("style_gram_mse_mean", "Lower Better"),
        ("flicker", "Lower Better"),
        ("relative_motion_error", "Lower Better"),
    ]
    for ax, (m, hint) in zip(axes, charts):
        ax.bar(summary["method"], summary[m], color=["#b55", "#4a8"])
        ax.set_title(f"{m} ({hint})")
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Real Video Strict Compare: Same Frames, Two Methods")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    metrics_png = preview_dir / f"{cfg['experiment']['name']}_metrics.png"
    fig.savefig(metrics_png)
    plt.close(fig)

    # per-clip breakdown figure
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=160)
    axes = axes.flatten()
    clip_labels = [f"{r.style_name}:{r.clip_id}" for r in clip_summary[["style_name", "clip_id"]].drop_duplicates().itertuples(index=False)]
    metrics = [
        ("content_ssim_img_mean", "Higher Better"),
        ("style_gram_mse_mean", "Lower Better"),
        ("color_mean_l1_to_style", "Lower Better"),
        ("flicker", "Lower Better"),
    ]
    methods = [m for m in ["collapse", "optimize"] if m in set(clip_summary["method"])]
    x = np.arange(len(clip_labels))
    width = 0.34
    offsets = np.linspace(-width / 2.0, width / 2.0, num=max(1, len(methods)))
    for ax, (metric, hint) in zip(axes, metrics):
        for j, method in enumerate(methods):
            ys = []
            for style_clip in clip_labels:
                sty, clip = style_clip.split(":", 1)
                rec = clip_summary[(clip_summary["style_name"] == sty) & (clip_summary["clip_id"] == clip) & (clip_summary["method"] == method)]
                ys.append(float(rec[metric].iloc[0]) if not rec.empty else np.nan)
            ax.bar(x + offsets[j], ys, width=max(0.22, width / max(1, len(methods))), label=method)
        ax.set_xticks(x, clip_labels, rotation=30)
        ax.set_title(f"{metric} ({hint})")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="best", fontsize=8)
    fig.suptitle("Real Video Strict: Per-Clip Breakdown (Same Frame Indices)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    clip_breakdown_png = preview_dir / f"{cfg['experiment']['name']}_clip_breakdown.png"
    fig.savefig(clip_breakdown_png)
    plt.close(fig)

    panel_png = compose_keyframe_megapanel(
        panel_paths=panel_paths,
        out_path=preview_dir / f"{cfg['experiment']['name']}_all_keyframes.png",
        title="Real Video Strict: Input vs Collapse vs Optimize (same frame indices)",
        cols=2,
    )

    md = reports_dir / f"{cfg['experiment']['name']}_summary.md"
    lines = []
    lines.append(f"# {cfg['experiment']['name']} Summary")
    lines.append("")
    lines.append("- Protocol: real videos, same exact frame indices for both methods.")
    lines.append(f"- metrics_csv: `{metrics_csv}`")
    lines.append(f"- clip_summary_csv: `{clip_summary_csv}`")
    lines.append(f"- summary_csv: `{summary_csv}`")
    lines.append(f"- reviewer_qa_csv: `{qa_csv}`")
    lines.append(f"- reviewer_qa_md: `{qa_md}`")
    lines.append(f"- same_frame_audit_csv: `{audit_csv}`")
    lines.append(f"- same_frame_audit_summary_csv: `{audit_summary_csv}`")
    lines.append(f"- metrics_png: `{metrics_png}`")
    lines.append(f"- clip_breakdown_png: `{clip_breakdown_png}`")
    if panel_png is not None:
        lines.append(f"- keyframe_panel_png: `{panel_png}`")
    lines.append("")
    lines.append("## Same-Frame Audit")
    lines.append(dataframe_to_markdown(audit_summary))
    lines.append("")
    lines.append("## Method Means")
    headers = list(summary.columns)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, r in summary.iterrows():
        vals = []
        for h in headers:
            v = r[h]
            vals.append(f"{float(v):.6f}" if isinstance(v, float) else str(v))
        lines.append("| " + " | ".join(vals) + " |")
    md.write_text("\n".join(lines), encoding="utf-8")

    payload = {
        "status": "ok",
        "device": device,
        "metrics_csv": str(metrics_csv),
        "clip_summary_csv": str(clip_summary_csv),
        "summary_csv": str(summary_csv),
        "reviewer_qa_csv": str(qa_csv),
        "reviewer_qa_md": str(qa_md),
        "same_frame_audit_csv": str(audit_csv),
        "same_frame_audit_summary_csv": str(audit_summary_csv),
        "summary_md": str(md),
        "metrics_png": str(metrics_png),
        "clip_breakdown_png": str(clip_breakdown_png),
        "panel_png": str(panel_png) if panel_png is not None else "",
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
