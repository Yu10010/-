from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


BASE_DIR = Path(__file__).resolve().parents[1]

CNN_SRC_DIR = BASE_DIR / "cnn" / "src"
CHECKPOINT_DIR = BASE_DIR / "cnn" / "checkpoints"

MODEL_SPECS = [
    {
        "name": "resnet18_i",
        "path": CHECKPOINT_DIR / "best_model_resnet18_i.pt",
        "weight": 0.7,
        "input_type": "normal_image",
    },
    {
        "name": "resnet18",
        "path": CHECKPOINT_DIR / "best_model_resnet18.pt",
        "weight": 0.3,
        "input_type": "heatmap",
    },
]

if str(CNN_SRC_DIR) not in sys.path:
    sys.path.append(str(CNN_SRC_DIR))

from model import create_model  # noqa: E402


_models: list[dict[str, Any]] | None = None
_device: torch.device | None = None
_class_names: list[str] | None = None


def _safe_torch_load(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _build_eval_transform(config: dict[str, Any]):
    image_size = config["dataset"].get("image_size", 224)
    mean = config["preprocess"].get("mean", [0.485, 0.456, 0.406])
    std = config["preprocess"].get("std", [0.229, 0.224, 0.225])

    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def _load_single_model(spec: dict[str, Any], device: torch.device) -> dict[str, Any]:
    checkpoint_path = spec["path"]

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = _safe_torch_load(checkpoint_path, device)

    config = checkpoint.get("config")
    if config is None:
        raise ValueError(f"{checkpoint_path} does not contain config.")

    class_names = checkpoint.get("class_names")
    if not class_names:
        class_names = ["fire", "nofire"]

    # Do not download ImageNet weights during API startup.
    config["model"]["pretrained"] = False

    model = create_model(config, num_classes=len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    transform = _build_eval_transform(config)

    return {
        "name": spec["name"],
        "weight": float(spec["weight"]),
        "input_type": spec["input_type"],
        "model": model,
        "class_names": class_names,
        "transform": transform,
        "config": config,
    }


def _load_ensemble():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loaded_models = [
        _load_single_model(spec, device)
        for spec in MODEL_SPECS
    ]

    reference_classes = loaded_models[0]["class_names"]

    for item in loaded_models:
        if item["class_names"] != reference_classes:
            raise ValueError(
                f"Class mismatch: {item['name']} has {item['class_names']}, "
                f"but expected {reference_classes}."
            )

    total_weight = sum(item["weight"] for item in loaded_models)
    if total_weight <= 0:
        raise ValueError("Model ensemble weights must sum to a positive number.")

    for item in loaded_models:
        item["weight"] = item["weight"] / total_weight

    return loaded_models, device, reference_classes


def _image_bytes_to_tensor(image_bytes: bytes, transform, device: torch.device):
    image = Image.open(BytesIO(image_bytes)).convert("RGB")
    return transform(image).unsqueeze(0).to(device)


def predict_wildfire(normal_image_bytes: bytes, heatmap_image_bytes: bytes) -> dict[str, Any]:
    global _models, _device, _class_names

    if _models is None:
        _models, _device, _class_names = _load_ensemble()

    combined_probabilities = torch.zeros(len(_class_names), device=_device)
    individual_results = []

    with torch.no_grad():
        for item in _models:
            if item["input_type"] == "normal_image":
                input_bytes = normal_image_bytes
            elif item["input_type"] == "heatmap":
                input_bytes = heatmap_image_bytes
            else:
                raise ValueError(f"Unknown input_type: {item['input_type']}")

            x = _image_bytes_to_tensor(input_bytes, item["transform"], _device)

            logits = item["model"](x)
            probabilities = F.softmax(logits, dim=1)[0]

            combined_probabilities += item["weight"] * probabilities

            pred_idx = int(torch.argmax(probabilities).item())

            individual_results.append(
                {
                    "model_name": item["name"],
                    "input_type": item["input_type"],
                    "weight": item["weight"],
                    "prediction": _class_names[pred_idx],
                    "confidence": float(probabilities[pred_idx].item()),
                    "class_probabilities": {
                        _class_names[i]: float(probabilities[i].item())
                        for i in range(len(_class_names))
                    },
                }
            )

    final_idx = int(torch.argmax(combined_probabilities).item())
    final_prediction = _class_names[final_idx]
    final_confidence = float(combined_probabilities[final_idx].item())

    return {
        "prediction": final_prediction,
        "confidence": final_confidence,
        "class_probabilities": {
            _class_names[i]: float(combined_probabilities[i].item())
            for i in range(len(_class_names))
        },
        "ensemble": {
            "method": "weighted_probability_average",
            "weights": {
                "resnet18_i": 0.7,
                "resnet18": 0.3,
            },
            "input_mapping": {
                "resnet18_i": "normal_image",
                "resnet18": "heatmap",
            },
            "individual_results": individual_results,
        },
        "source": "local_cnn_ensemble",
    }