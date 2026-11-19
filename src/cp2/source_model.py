"""A shared architecture for clean, augmentation, and quality-supervised source gates."""

import torch
from torch import nn

from .mm_data import MemeMultimodalDataset
from .nlp_data import MAX_LENGTH, TEXT_MODEL, TEXT_REVISION, VIT_MODEL, VIT_REVISION


SOURCE_ARMS = ("S0", "S1", "S2")
QUALITY_WEIGHT = 0.1


class SourceDataset(MemeMultimodalDataset):
    def __init__(self, frame, images, tokenizer, processor, quality):
        super().__init__(frame, images, tokenizer, processor)
        texts = frame.loc[:, ["caption", "ocr"]].to_numpy().reshape(-1).tolist()
        tokens = tokenizer(
            texts, padding="max_length", truncation=True, max_length=MAX_LENGTH // 2,
            return_tensors="pt", return_token_type_ids=False,
        )
        self.source_ids = tokens["input_ids"].reshape(len(frame), 2, -1)
        self.source_mask = tokens["attention_mask"].reshape(len(frame), 2, -1)
        self.source_quality = torch.as_tensor(quality)
        self.source_present = self.source_quality >= 0

    def __getitem__(self, idx):
        item = super().__getitem__(idx)
        item.update(source_ids=self.source_ids[idx], source_mask=self.source_mask[idx],
                    source_present=self.source_present[idx], source_quality=self.source_quality[idx])
        return item


class SourceGate(nn.Module):
    def __init__(self, text_encoder=None, vision_encoder=None):
        super().__init__()
        if text_encoder is None or vision_encoder is None:
            from transformers import AutoModel
            if text_encoder is None:
                text_encoder = AutoModel.from_pretrained(TEXT_MODEL, revision=TEXT_REVISION)
            if vision_encoder is None:
                vision_encoder = AutoModel.from_pretrained(VIT_MODEL, revision=VIT_REVISION)
        self.text_encoder, self.vision_encoder = text_encoder, vision_encoder
        self.visual_projection = nn.Linear(768, 768)
        self.gate = nn.Sequential(nn.Linear(768 * 2, 128), nn.GELU(), nn.Linear(128, 1))
        self.dropout = nn.Dropout(0.3)
        self.heads = nn.ModuleDict({task: nn.Linear(768, 2) for task in ("hate", "appropriate")})

    def forward(self, batch, *, return_quality=False):
        ids, mask = batch["source_ids"], batch["source_mask"]
        text = self.text_encoder(input_ids=ids.flatten(0, 1), attention_mask=mask.flatten(0, 1))
        texts = text.pooler_output.reshape(ids.shape[0], 2, 768)
        image = self.visual_projection(self.vision_encoder(pixel_values=batch["pixel_values"]).pooler_output)
        available = torch.cat((batch["source_present"], torch.ones_like(batch["source_present"][:, :1])), dim=1)
        features = torch.cat((texts, image[:, None, :]), dim=1)
        features = features.masked_fill(~available.unsqueeze(-1), 0)
        # Score source agreement against the other available signals.
        others = (features.sum(dim=1, keepdim=True) - features) / (available.sum(dim=1) - 1).clamp_min(1)[:, None, None]
        gate_logits = self.gate(torch.cat((features, others), dim=-1)).squeeze(-1)
        weights = gate_logits.masked_fill(~available, -torch.inf).softmax(dim=1)
        fused = self.dropout((features * weights.unsqueeze(-1)).sum(dim=1))
        logits = {task: head(fused) for task, head in self.heads.items()}
        return (logits, gate_logits[:, :2]) if return_quality else logits

    def parameter_summary(self):
        total = sum(p.numel() for p in self.parameters())
        backbone = sum(p.numel() for encoder in (self.text_encoder, self.vision_encoder) for p in encoder.parameters())
        return {"total": total, "trainable": total, "non_encoder": total - backbone}


def quality_loss(logits, targets):
    observed = targets >= 0
    loss = nn.functional.binary_cross_entropy_with_logits(logits, targets.clamp_min(0).to(logits.dtype), reduction="none")
    return (loss * observed).sum() / observed.sum().clamp_min(1)
