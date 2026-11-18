"""NLP-owned B0 CP2 baseline; CV-owned B1/B2/M1/M2/M3 not silently substituted."""
import torch.nn as nn
from .nlp_data import TEXT_MODEL, TEXT_REVISION


class TextOnlyB0(nn.Module):
    def __init__(self, dropout=0.3):
        super().__init__()
        from transformers import AutoModel
        self.encoder = AutoModel.from_pretrained(TEXT_MODEL, revision=TEXT_REVISION)
        hidden = int(self.encoder.config.hidden_size)
        if hidden != 768:
            raise ValueError(f"Expected IndoBERTweet hidden size=768; got {hidden}")
        self.projection = nn.Sequential(nn.Dropout(dropout), nn.Linear(768, 384), nn.Dropout(dropout))
        self.heads = nn.ModuleDict({"hate": nn.Linear(384, 2), "appropriate": nn.Linear(384, 2)})

    def forward(self, batch):
        text = self.encoder(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
        if text.pooler_output is None:
            raise RuntimeError("IndoBERTweet pooler_output unavailable; do not silently switch pooling")
        h = self.projection(text.pooler_output)
        return {name: layer(h) for name, layer in self.heads.items()}
