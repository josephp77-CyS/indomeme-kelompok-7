from types import SimpleNamespace

import pandas as pd
import torch
from torch import nn

from cp2.nlp_cv_bridge import model_inputs
from cp2.queue_eval import simulate_review
from cp2.source_model import SourceGate, quality_loss


class TextEncoder(nn.Module):
    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(pooler_output=input_ids.float().mean(dim=1, keepdim=True).expand(-1, 768))


class ImageEncoder(nn.Module):
    def forward(self, pixel_values):
        return SimpleNamespace(pooler_output=pixel_values.mean(dim=(1, 2, 3))[:, None].expand(-1, 768))


def test_missing_source_does_not_influence_logits_and_targets_do_not_enter_inputs():
    model = SourceGate(TextEncoder(), ImageEncoder()).eval()
    source_ids = torch.ones(1, 2, 128, dtype=torch.long)
    inputs = {
        "input_ids": torch.ones(1, 256, dtype=torch.long), "attention_mask": torch.ones(1, 256, dtype=torch.long),
        "content_mask": torch.ones(1, 256, dtype=torch.bool), "pixel_values": torch.ones(1, 3, 2, 2),
        "source_ids": source_ids, "source_mask": torch.ones_like(source_ids),
        "source_present": torch.tensor([[False, True]]), "source_quality": torch.tensor([[-1, 1]]),
    }
    features = model_inputs(inputs, "S2")
    assert "source_quality" not in features
    before, scores = model(features, return_quality=True)
    features["source_ids"][:, 0].fill_(30)
    after = model(features)
    assert scores.shape == (1, 2)
    assert all(torch.equal(before[task], after[task]) for task in before)


def test_quality_loss_excludes_missing_source():
    logits = torch.zeros((1, 2), requires_grad=True)
    quality_loss(logits, torch.tensor([[-1, 1]])).backward()
    assert logits.grad.tolist() == [[0.0, -0.5]]


def test_review_is_paired_by_meme_and_uses_same_budget():
    rows = [
        {"image_path": image, "task": task, "truth": truth, "prediction_raw": truth, "probability_positive": confidence}
        for image, truth, confidence in (("a", 0, .1), ("b", 1, .8))
        for task in ("hate", "appropriate")
    ]
    result = simulate_review(pd.DataFrame(rows[::-1]), ["a", "b"], [0.2, 0.1])
    assert {budgets["0.1"]["reviewed"] for budgets in result.values()} == {1}
    assert result["positive_score"]["0.1"]["gold_positive_recall"]["hate"] == 1.0
