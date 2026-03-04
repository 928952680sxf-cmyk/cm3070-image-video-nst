from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass
class ExperimentConfig:
    name: str
    run_dir: str
    content_path: str
    style_path: str
    device: str
    seed: int
    image_size: int
    architecture: str
    content_layer: str
    style_layers: list[str]
    steps: int
    optimizer: str
    lr: float
    content_weight: float
    style_weight: float
    tv_weight: float
    log_interval: int
    init: str


def load_config(path: str) -> ExperimentConfig:
    cfg = yaml.safe_load(Path(path).read_text())
    return ExperimentConfig(
        name=cfg["experiment"]["name"],
        run_dir=cfg["experiment"]["run_dir"],
        content_path=cfg["inputs"]["content_path"],
        style_path=cfg["inputs"]["style_path"],
        device=cfg["runtime"]["device"],
        seed=int(cfg["runtime"]["seed"]),
        image_size=int(cfg["runtime"]["image_size"]),
        architecture=cfg["model"]["architecture"],
        content_layer=cfg["model"]["content_layer"],
        style_layers=list(cfg["model"]["style_layers"]),
        steps=int(cfg["train"]["steps"]),
        optimizer=cfg["train"]["optimizer"],
        lr=float(cfg["train"]["lr"]),
        content_weight=float(cfg["train"]["content_weight"]),
        style_weight=float(cfg["train"]["style_weight"]),
        tv_weight=float(cfg["train"]["tv_weight"]),
        log_interval=int(cfg["train"]["log_interval"]),
        init=cfg["train"].get("init", "content"),
    )
