"""CRNN: CNN backbone -> (bi)RNN -> per-timestep class log-probabilities, trained with CTC."""

from __future__ import annotations

import timm
import torch
import torch.nn as nn


def _relax_vertical_stride(backbone: nn.Module) -> None:
    """Turn the stride-2 convs of ResNet stages 3 and 4 into (2, 1).

    Height is still downsampled (it is later averaged away) but width keeps 1/8 of the
    input resolution, giving ``W/8`` time steps instead of ``W/32`` -- enough room for
    CTC to emit repeated letters and long words.
    """
    for stage in ("layer3", "layer4"):
        layer = getattr(backbone, stage, None)
        if layer is None:
            continue
        for m in layer[0].modules():
            if isinstance(m, nn.Conv2d) and m.stride == (2, 2):
                m.stride = (2, 1)


class CRNN(nn.Module):
    def __init__(
        self,
        num_classes: int,
        backbone: str = "resnet34",
        hidden_size: int = 256,
        n_layers: int = 3,
        dropout: float = 0.2,
        rnn: str = "lstm",
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=pretrained, in_chans=1, num_classes=0, global_pool=""
        )
        _relax_vertical_stride(self.backbone)
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        feat_dim = self.backbone.num_features
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, 512), nn.ReLU(inplace=True), nn.Dropout(dropout)
        )
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn.lower()]
        self.rnn = rnn_cls(
            512,
            hidden_size,
            n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_size * 2)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, H, W) -> log-probs (T, B, num_classes), as expected by ``nn.CTCLoss``."""
        feats = self.backbone(x)  # (B, C, H', W')
        feats = feats.mean(dim=2)  # collapse height -> (B, C, W')
        feats = feats.permute(0, 2, 1)  # (B, T, C)
        feats = self.proj(feats)
        out, _ = self.rnn(feats)
        logits = self.classifier(self.norm(out))
        return logits.log_softmax(dim=2).permute(1, 0, 2)

    def parameter_groups(self, lr: float, backbone_lr_mult: float = 0.1) -> list[dict]:
        backbone = [p for p in self.backbone.parameters() if p.requires_grad]
        head = [
            p
            for n, p in self.named_parameters()
            if not n.startswith("backbone.") and p.requires_grad
        ]
        groups = [{"params": head, "lr": lr}]
        if backbone:
            groups.append({"params": backbone, "lr": lr * backbone_lr_mult})
        return groups
