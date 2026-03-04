#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from skimage import color
from skimage.metrics import structural_similarity as ssim
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

ROOT = Path(__file__).resolve().parents[1]

import sys

if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from nst.engine import TransferConfig, run_style_transfer
from nst.io import save_image
from nst.model import FeatureExtractorConfig, VGGFeatureExtractor


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


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


def load_manifest_rows(path: Path) -> list[dict[str, Any]]:
    return pd.read_csv(path).to_dict("records")


def load_tensor(path: Path, image_size: int, device: str) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    img = TF.resize(
        img,
        [image_size, image_size],
        interpolation=InterpolationMode.BICUBIC,
        antialias=True,
    )
    return TF.to_tensor(img).unsqueeze(0).to(device).clamp(0.0, 1.0)


def tensor_to_np(x: torch.Tensor) -> np.ndarray:
    return x.squeeze(0).detach().cpu().permute(1, 2, 0).numpy().clip(0.0, 1.0)


def np_color_hist_emd(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    vals = []
    for ch in range(3):
        ha, _ = np.histogram(np.clip(a[..., ch], 0, 1), bins=bins, range=(0.0, 1.0), density=True)
        hb, _ = np.histogram(np.clip(b[..., ch], 0, 1), bins=bins, range=(0.0, 1.0), density=True)
        ha = ha / max(1e-8, float(np.sum(ha)))
        hb = hb / max(1e-8, float(np.sum(hb)))
        ca = np.cumsum(ha)
        cb = np.cumsum(hb)
        vals.append(float(np.mean(np.abs(ca - cb))))
    return float(np.mean(vals))


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
    t_rgb = color.lab2rgb(t_lab)
    return np.clip(t_rgb, 0.0, 1.0)


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
    mixed = (1.0 - alpha) * o + alpha * c
    return np.clip(mixed, 0.0, 1.0)


def build_style_score(local_relpath: str) -> float:
    p = ROOT / "data" / "phase3_s" / str(local_relpath)
    try:
        img = Image.open(p).convert("RGB")
        img = TF.resize(
            img,
            [128, 128],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        arr = np.asarray(img, dtype=np.float32) / 255.0
        r = arr[..., 0]
        g = arr[..., 1]
        b = arr[..., 2]
        rg = r - g
        yb = 0.5 * (r + g) - b
        std_rg = float(np.std(rg))
        std_yb = float(np.std(yb))
        mean_rg = float(np.mean(np.abs(rg)))
        mean_yb = float(np.mean(np.abs(yb)))
        colorfulness = float(
            np.sqrt(std_rg * std_rg + std_yb * std_yb) + 0.3 * np.sqrt(mean_rg * mean_rg + mean_yb * mean_yb)
        )

        gray = 0.299 * r + 0.587 * g + 0.114 * b
        contrast = float(np.std(gray))
        gx = np.diff(gray, axis=1)[:-1, :]
        gy = np.diff(gray, axis=0)[:, :-1]
        edge = float(np.mean(np.sqrt(gx * gx + gy * gy)))
        return colorfulness + 0.75 * contrast + 2.2 * edge
    except Exception:
        return 0.0


@dataclass
class StyleRef:
    name: str
    style_id: int
    style_label: int
    source: str
    path: Path


def pick_top_styles(style_rows: list[dict[str, Any]], top_k: int, exclude_ids: set[int] | None = None) -> list[StyleRef]:
    exclude_ids = set() if exclude_ids is None else set(exclude_ids)

    best_by_label: dict[int, tuple[float, dict[str, Any]]] = {}
    for row in style_rows:
        sid = int(row["id"])
        if sid in exclude_ids:
            continue
        label = int(row["style"])
        score = build_style_score(str(row["local_relpath"]))
        cur = best_by_label.get(label)
        if cur is None or score > cur[0]:
            best_by_label[label] = (score, row)

    ranked = sorted(best_by_label.values(), key=lambda x: x[0], reverse=True)[:top_k]
    out: list[StyleRef] = []
    for score, row in ranked:
        sid = int(row["id"])
        out.append(
            StyleRef(
                name=f"sid_{sid}",
                style_id=sid,
                style_label=int(row["style"]),
                source="manifest",
                path=ROOT / "data" / "phase3_s" / str(row["local_relpath"]),
            )
        )
    return out


def run_one(
    extractor: VGGFeatureExtractor,
    content: torch.Tensor,
    style: torch.Tensor,
    method: str,
    cfg: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, Any]]:
    content_layer = "conv4_2"
    style_layers = ["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"]

    t0 = time.time()
    meta: dict[str, Any] = {}

    if method in {"BASE", "B_ONLY"}:
        tc = TransferConfig(
            steps=int(cfg["base"]["steps"]),
            optimizer="adam",
            lr=float(cfg["base"]["lr"]),
            content_weight=float(cfg["base"]["content_weight"]),
            style_weight=float(cfg["base"]["style_weight"]),
            tv_weight=float(cfg["base"]["tv_weight"]),
            log_interval=max(1, int(cfg["base"]["steps"]) // 4),
            init="content",
        )
        out, hist, _ = run_style_transfer(extractor, content, style, content_layer, style_layers, tc)
        meta["history"] = hist
        if method == "B_ONLY":
            out_np = tensor_to_np(out)
            c_np = tensor_to_np(content)
            s_np = tensor_to_np(style)
            out_np = reinhard_color_transfer(out_np, s_np)
            out_np = edge_preserve_blend(out_np, c_np, strength=float(cfg["b"]["edge_blend_strength"]))
            out = torch.from_numpy(out_np).permute(2, 0, 1).unsqueeze(0).to(content.device).float().clamp(0.0, 1.0)
    elif method in {"A_ONLY", "A_B"}:
        c1 = cfg["a_stage1"]
        tc1 = TransferConfig(
            steps=int(c1["steps"]),
            optimizer="lbfgs",
            lr=float(c1["lr"]),
            content_weight=float(c1["content_weight"]),
            style_weight=float(c1["style_weight"]),
            tv_weight=float(c1["tv_weight"]),
            log_interval=max(1, int(c1["steps"]) // 4),
            init="content",
        )
        mid, h1, _ = run_style_transfer(extractor, content, style, content_layer, style_layers, tc1)

        c2 = cfg["a_stage2"]
        tc2 = TransferConfig(
            steps=int(c2["steps"]),
            optimizer="adam",
            lr=float(c2["lr"]),
            content_weight=float(c2["content_weight"]),
            style_weight=float(c2["style_weight"]),
            tv_weight=float(c2["tv_weight"]),
            log_interval=max(1, int(c2["steps"]) // 4),
            init="tensor",
        )
        out, h2, _ = run_style_transfer(extractor, content, style, content_layer, style_layers, tc2, init_image=mid)
        meta["history_stage1"] = h1
        meta["history_stage2"] = h2

        if method == "A_B":
            out_np = tensor_to_np(out)
            c_np = tensor_to_np(content)
            s_np = tensor_to_np(style)
            out_np = reinhard_color_transfer(out_np, s_np)
            out_np = edge_preserve_blend(out_np, c_np, strength=float(cfg["b"]["edge_blend_strength"]))
            out = torch.from_numpy(out_np).permute(2, 0, 1).unsqueeze(0).to(content.device).float().clamp(0.0, 1.0)
    else:
        raise ValueError(f"Unknown method: {method}")

    meta["elapsed_sec"] = time.time() - t0
    return out, meta


def render_matrix(
    out_png: Path,
    title: str,
    rows: list[int],
    cols: list[StyleRef],
    image_map: dict[tuple[int, str], np.ndarray],
) -> None:
    n_r = len(rows)
    n_c = len(cols)
    fig, axes = plt.subplots(n_r, n_c, figsize=(2.9 * n_c, 2.9 * n_r))
    if n_r == 1 and n_c == 1:
        axes = np.array([[axes]])
    elif n_r == 1:
        axes = np.expand_dims(axes, axis=0)
    elif n_c == 1:
        axes = np.expand_dims(axes, axis=1)

    for r, cid in enumerate(rows):
        for c, spec in enumerate(cols):
            ax = axes[r, c]
            ax.imshow(image_map[(int(cid), spec.name)])
            ax.set_title(f"cid={cid} | {spec.name}", fontsize=8)
            ax.axis("off")

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    ensure_dir(out_png.parent)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    headers = [str(x) for x in df.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        vals: list[str] = []
        for h in headers:
            v = row[h]
            if isinstance(v, float):
                vals.append(f"{v:.6f}")
            else:
                vals.append(str(v))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/phase2_outstanding_recovery.yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load((ROOT / args.config).read_text())
    set_seed(int(cfg["runtime"]["seed"]))
    device = resolve_device(str(cfg["runtime"]["device"]))

    out_root = ensure_dir(ROOT / cfg["experiment"]["out_root"])
    preview_dir = ensure_dir(ROOT / "reports" / "preview")
    reports_dir = ensure_dir(ROOT / "reports")

    style_rows = load_manifest_rows(ROOT / cfg["data"]["style_manifest"])
    content_rows = {int(r["id"]): r for r in load_manifest_rows(ROOT / cfg["data"]["content_manifest"])}

    content_ids = [int(x) for x in cfg["data"]["content_ids"]]
    style_refs = [StyleRef(name="starry_night", style_id=-1, style_label=-1, source="external", path=ROOT / cfg["data"]["starry_style_path"])]
    style_refs.extend(pick_top_styles(style_rows, top_k=int(cfg["data"]["extra_style_count"])))

    extractor = VGGFeatureExtractor(
        FeatureExtractorConfig(
            architecture="vgg19",
            content_layer="conv4_2",
            style_layers=["conv1_1", "conv2_1", "conv3_1", "conv4_1", "conv5_1"],
        )
    ).to(device)

    methods = ["BASE", "A_ONLY", "B_ONLY", "A_B"]
    records: list[dict[str, Any]] = []
    stylized_map: dict[str, dict[tuple[int, str], np.ndarray]] = {m: {} for m in methods}

    for cid in content_ids:
        c_row = content_rows[int(cid)]
        c_path = ROOT / "data" / "phase3_s" / str(c_row["local_relpath"])
        c_tensor = load_tensor(c_path, image_size=int(cfg["runtime"]["image_size"]), device=device)
        c_np = tensor_to_np(c_tensor)

        for sref in style_refs:
            s_tensor = load_tensor(sref.path, image_size=int(cfg["runtime"]["image_size"]), device=device)
            s_np = tensor_to_np(s_tensor)
            ssim_cs = float(ssim(c_np, s_np, channel_axis=2, data_range=1.0))

            for method in methods:
                out, meta = run_one(extractor, c_tensor, s_tensor, method, cfg["train"])
                o_np = tensor_to_np(out)
                stylized_map[method][(int(cid), sref.name)] = o_np

                out_img = out_root / method.lower() / f"cid{cid}_{sref.name}.png"
                save_image(out, str(out_img))

                records.append(
                    {
                        "method": method,
                        "content_id": int(cid),
                        "style_name": sref.name,
                        "style_id": int(sref.style_id),
                        "style_label": int(sref.style_label),
                        "elapsed_sec": float(meta["elapsed_sec"]),
                        "ssim_content_style_input": ssim_cs,
                        "ssim_content_output": float(ssim(c_np, o_np, channel_axis=2, data_range=1.0)),
                        "color_mean_l1": float(np.mean(np.abs(o_np.mean((0, 1)) - s_np.mean((0, 1))))),
                        "color_std_l1": float(np.mean(np.abs(o_np.std((0, 1)) - s_np.std((0, 1))))),
                        "color_hist_emd": float(np_color_hist_emd(o_np, s_np, bins=32)),
                        "output_path": str(out_img),
                    }
                )

    df = pd.DataFrame(records)
    csv_path = out_root / "metrics_phase2_outstanding_recovery.csv"
    ensure_dir(csv_path.parent)
    df.to_csv(csv_path, index=False)

    summary_rows = []
    for method in methods:
        part = df[df["method"] == method]
        summary_rows.append(
            {
                "method": method,
                "pairs": int(len(part)),
                "elapsed_sec_mean": float(part["elapsed_sec"].mean()),
                "ssim_content_output_mean": float(part["ssim_content_output"].mean()),
                "color_mean_l1_mean": float(part["color_mean_l1"].mean()),
                "color_std_l1_mean": float(part["color_std_l1"].mean()),
                "color_hist_emd_mean": float(part["color_hist_emd"].mean()),
            }
        )
    summary_df = pd.DataFrame(summary_rows)
    summary_csv = out_root / "metrics_phase2_outstanding_recovery_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    for method in methods:
        render_matrix(
            out_png=preview_dir / f"phase2_outstanding_{method.lower()}_matrix.png",
            title=f"Phase2 Outstanding Recovery - {method}",
            rows=content_ids,
            cols=style_refs,
            image_map=stylized_map[method],
        )

    # Starry-only method comparison panel (rows=content, cols=method)
    starry = StyleRef(name="starry_night", style_id=-1, style_label=-1, source="external", path=ROOT / cfg["data"]["starry_style_path"])
    fig, axes = plt.subplots(len(content_ids), len(methods), figsize=(3.2 * len(methods), 3.0 * len(content_ids)))
    if len(content_ids) == 1:
        axes = np.expand_dims(axes, axis=0)
    for r, cid in enumerate(content_ids):
        for c, method in enumerate(methods):
            ax = axes[r, c]
            ax.imshow(stylized_map[method][(int(cid), starry.name)])
            ax.set_title(f"cid={cid} | {method}", fontsize=8)
            ax.axis("off")
    fig.suptitle("Starry Method Compare (rows=content, cols=method)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    starry_compare_png = preview_dir / "phase2_outstanding_starry_method_compare.png"
    fig.savefig(starry_compare_png, dpi=160)
    plt.close(fig)

    md_lines = []
    md_lines.append("# Phase2 Outstanding Recovery Summary")
    md_lines.append("")
    md_lines.append(f"- device: {device}")
    md_lines.append(f"- content_ids: {content_ids}")
    md_lines.append("- styles: " + ", ".join([f"{s.name}(id={s.style_id},label={s.style_label})" for s in style_refs]))
    md_lines.append("")
    md_lines.append("## Aggregate Metrics")
    md_lines.append(dataframe_to_markdown(summary_df))
    md_lines.append("")
    md_lines.append("## Visual Artifacts")
    md_lines.append(f"- BASE matrix: `{preview_dir / 'phase2_outstanding_base_matrix.png'}`")
    md_lines.append(f"- A_ONLY matrix: `{preview_dir / 'phase2_outstanding_a_only_matrix.png'}`")
    md_lines.append(f"- B_ONLY matrix: `{preview_dir / 'phase2_outstanding_b_only_matrix.png'}`")
    md_lines.append(f"- A_B matrix: `{preview_dir / 'phase2_outstanding_a_b_matrix.png'}`")
    md_lines.append(f"- Starry method compare: `{starry_compare_png}`")
    md_lines.append("")
    md_lines.append("## Decision Rule (visual-first)")
    md_lines.append("- Primary criterion: same-content cross-style outputs must show clear palette/texture differences by eye.")
    md_lines.append("- Secondary criterion: color_hist_emd/color_mean_l1 should not systematically regress against BASE.")

    summary_md = reports_dir / "phase2_outstanding_recovery_summary.md"
    summary_md.write_text("\n".join(md_lines), encoding="utf-8")

    payload = {
        "status": "ok",
        "device": device,
        "metrics_csv": str(csv_path),
        "summary_csv": str(summary_csv),
        "summary_md": str(summary_md),
        "preview": {
            "base_matrix": str(preview_dir / "phase2_outstanding_base_matrix.png"),
            "a_only_matrix": str(preview_dir / "phase2_outstanding_a_only_matrix.png"),
            "b_only_matrix": str(preview_dir / "phase2_outstanding_b_only_matrix.png"),
            "a_b_matrix": str(preview_dir / "phase2_outstanding_a_b_matrix.png"),
            "starry_method_compare": str(starry_compare_png),
        },
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
