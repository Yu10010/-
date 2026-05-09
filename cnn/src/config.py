from __future__ import annotations

import copy
import json
from pathlib import Path


DEFAULT_CONFIG = {
    "dataset": {
        "data_dir": "data",
        "train_dir": "train",
        "val_dir": "valid",
        "test_dir": "test",
        "image_size": 224,
        "num_workers": 4,
        "train_multiplier": 3.0,
        "balance_train_classes": True,
    },
    "preprocess": {
        "normalization": "imagenet",
        "use_augmentation": True,
        "horizontal_flip": 0.2,
        "rotation": 3,
        "color_jitter": 0.0,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
    "model": {
        "name": "resnet18",
        "dropout": 0.3,
        "pretrained": True,
        "freeze_backbone": True,
    },
    "train": {
        "seed": 42,
        "batch_size": 64,
        "epochs": 80,
        "learning_rate": 1e-4,
        "weight_decay": 1e-5,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "early_stopping_patience": 5,
        "device": "auto",
        "output_dir": "outputs",
        "experiment_name": "baseline",
        "checkpoint_name": "best_model.pt",
        "latest_checkpoint_name": "latest_checkpoint.pt",
    },
}


def merge_dict(base_dict: dict, new_dict: dict) -> dict:
    merged = copy.deepcopy(base_dict)
    for key, value in new_dict.items():
        if isinstance(value, dict) and key in merged and isinstance(merged[key], dict):
            merged[key] = merge_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path | None = None) -> dict:
    config = copy.deepcopy(DEFAULT_CONFIG)
    if config_path is None:
        return config

    user_config = json.loads(Path(config_path).read_text())
    return merge_dict(config, user_config)


def save_config(config: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(config, indent=2))
