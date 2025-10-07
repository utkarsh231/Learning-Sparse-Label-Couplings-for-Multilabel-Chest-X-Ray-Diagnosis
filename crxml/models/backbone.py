import timm
import torch.nn as nn

def create_backbone(model_name: str, num_classes: int, drop_rate: float = 0.4, pretrained: bool = True) -> nn.Module:
    """Create a timm backbone with desired classifier head.
    """
    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes, drop_rate=drop_rate)
    return model