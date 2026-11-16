"""B1/B2 pooled multimodal CP2 candidates, following CV01 design (CV review required).

Both backbones are fine-tuned from pinned pretrained weights, not from a B0
fine-tuned checkpoint. This is deliberate for fair comparison.
"""
from __future__ import annotations

import torch
from torch import nn

from .nlp_data import TEXT_MODEL, TEXT_REVISION, VIT_MODEL, VIT_REVISION


class PooledMultimodal(nn.Module):
    """B1: concat [t;v]. B2: t + sigmoid(g([t;u]))*u, u=Linear(v)."""

    VARIANTS = ("B1", "B2")

    def __init__(self, variant: str, dropout: float = 0.3,
                 text_encoder: nn.Module | None = None,
                 vision_encoder: nn.Module | None = None):
        super().__init__()
        if variant not in self.VARIANTS:
            raise ValueError(f"Only B1/B2 supported; got {variant!r}")
        if not (0 <= dropout < 1):
            raise ValueError("dropout must be in [0,1)")
        if text_encoder is None or vision_encoder is None:
            from transformers import AutoModel
            if text_encoder is None:
                text_encoder = AutoModel.from_pretrained(TEXT_MODEL, revision=TEXT_REVISION)
            if vision_encoder is None:
                vision_encoder = AutoModel.from_pretrained(VIT_MODEL, revision=VIT_REVISION)
        self.variant = variant
        self.text_encoder = text_encoder
        self.vision_encoder = vision_encoder
        for name, encoder in (("text", self.text_encoder), ("vision", self.vision_encoder)):
            hidden = int(getattr(encoder.config, "hidden_size", -1))
            if hidden != 768:
                raise ValueError(f"{name} encoder hidden_size={hidden}; expected 768")
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({
            "hate": nn.Linear(768, 2),
            "appropriate": nn.Linear(768, 2),
        })
        if variant == "B1":
            self.projection = nn.Linear(1536, 768)
        else:
            self.visual_projection = nn.Linear(768, 768)
            self.gate = nn.Linear(1536, 1)

    def fused_features(self, text_pooled: torch.Tensor, image_pooled: torch.Tensor,
                       visual_off: bool = False) -> torch.Tensor:
        if (text_pooled.ndim != 2 or text_pooled.shape[-1] != 768 or
                image_pooled.shape != text_pooled.shape):
            raise ValueError("Expected text and image poolers with shape [B,768]")
        if self.variant == "B1":
            # CV01 B1: zero the image feature BEFORE concat; projection bias stays.
            v = torch.zeros_like(image_pooled) if visual_off else image_pooled
            return self.projection(torch.cat([text_pooled, v], dim=-1))
        u = self.visual_projection(image_pooled)
        gate = torch.sigmoid(self.gate(torch.cat([text_pooled, u], dim=-1)))
        # CV01 B2: zero AFTER projection and bias. This is not train-time dropout.
        return text_pooled + (torch.zeros_like(u) if visual_off else gate * u)

    def forward(self, batch: dict, visual_off: bool = False) -> dict[str, torch.Tensor]:
        required = ("input_ids", "attention_mask", "pixel_values")
        if any(name not in batch for name in required):
            raise ValueError(f"B1/B2 batch must include {required}")
        pixels = batch["pixel_values"]
        if pixels.ndim != 4 or pixels.shape[1:] != (3, 224, 224):
            raise ValueError("Expected pixel_values of shape [B,3,224,224]")
        if pixels.shape[0] != batch["input_ids"].shape[0]:
            raise ValueError("Text/image batch size mismatch")
        if not torch.isfinite(pixels).all():
            raise ValueError("Nonfinite image pixels")
        t_out = self.text_encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        v_out = self.vision_encoder(pixel_values=pixels)
        t, v = t_out.pooler_output, v_out.pooler_output
        if t is None or v is None:
            raise RuntimeError("B1/B2 require pretrained text and ViT pooler_output")
        representation = self.dropout(self.fused_features(t, v, visual_off=visual_off))
        return {name: head(representation) for name, head in self.heads.items()}

    def parameter_summary(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        backbone = sum(p.numel() for p in self.text_encoder.parameters()) + sum(
            p.numel() for p in self.vision_encoder.parameters()
        )
        return {"variant": self.variant, "total": total,
                "trainable": sum(p.numel() for p in self.parameters() if p.requires_grad),
                "non_encoder": total - backbone}
