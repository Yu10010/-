from __future__ import annotations

import torch.nn as nn
from torchvision import models

def create_resnet18(num_classes: int, dropout: float) -> nn.Module:
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def create_resnet34(num_classes: int, dropout: float) -> nn.Module:
    model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def freeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if not name.startswith("fc"):
            param.requires_grad = False


def create_model(config: dict, num_classes: int) -> nn.Module:
    model_name = config["model"].get("name", "resnet18").lower()
    dropout = config["model"]["dropout"]
    pretrained = config["model"].get("pretrained", True)
    freeze_backbone_flag = config["model"].get("freeze_backbone", True)

    if model_name == "resnet18":
        if pretrained:
            model = create_resnet18(num_classes=num_classes, dropout=dropout)
        else:
            model = models.resnet18(weights=None)
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )
        if freeze_backbone_flag:
            freeze_backbone(model)
        return model
    if model_name == "resnet34":
        if pretrained:
            model = create_resnet34(num_classes=num_classes, dropout=dropout)
        else:
            model = models.resnet34(weights=None)
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )
        if freeze_backbone_flag:
            freeze_backbone(model)
        return model

    raise ValueError("model.name must be one of: resnet18, resnet34")
