"""Offline CV01 B1/B2 regression tests. No HF downloads or GPU required."""
from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn

from cp2.mm_data import MemeMultimodalDataset
from cp2.mm_models import PooledMultimodal
from cp2.mm_runner import checked_args, train_epoch_mm
from cp2.nlp_evaluation import evaluate_records
from cp2.nlp_runner import model_loss


class TinyEncoder(nn.Module):
    def __init__(self, multimodal: bool):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=768)
        self.multimodal = multimodal
        self.weight = nn.Parameter(torch.ones(768) * 0.001)

    def forward(self, input_ids=None, attention_mask=None, pixel_values=None):
        if self.multimodal:
            assert pixel_values.shape[1:] == (3, 224, 224)
            scalar = pixel_values.mean(dim=(1, 2, 3)).unsqueeze(-1)
        else:
            assert input_ids.shape == attention_mask.shape
            scalar = input_ids.float().mean(dim=1, keepdim=True) / 100
        return SimpleNamespace(pooler_output=scalar * self.weight[None, :])


class TinyTokenizer:
    sep_token = "[SEP]"
    all_special_ids = [0, 101, 102]

    def __call__(self, texts, **kwargs):
        size = kwargs.get("max_length", 256)
        ids = [[101, 7, 102] + [0] * (size-3) for _ in texts]
        x = torch.tensor(ids)
        return {"input_ids": x, "attention_mask": (x != 0).long()}


def make_model(variant):
    return PooledMultimodal(variant, text_encoder=TinyEncoder(False),
                            vision_encoder=TinyEncoder(True))


def batch(n=2):
    return {
        "input_ids": torch.tensor([[101, 7, 102]] * n),
        "attention_mask": torch.ones(n, 3, dtype=torch.long),
        "pixel_values": torch.ones(n, 3, 224, 224),
    }


@pytest.mark.parametrize("variant,params", [("B1", 1183492), ("B2", 595205)])
def test_shapes_params_and_backprop(variant, params):
    torch.manual_seed(42)
    model = make_model(variant)
    assert model.parameter_summary()["non_encoder"] == params
    assert model.parameter_summary()["trainable"] == model.parameter_summary()["total"]
    outputs = model(batch())
    assert set(outputs) == {"hate", "appropriate"}
    assert all(v.shape == (2, 2) for v in outputs.values())
    loss = model_loss(outputs, torch.tensor([[0, 1], [1, 1]]))
    assert torch.isfinite(loss)
    loss.backward()
    assert model.text_encoder.weight.grad is not None
    assert model.vision_encoder.weight.grad is not None
    assert torch.isfinite(model.text_encoder.weight.grad).all()
    assert torch.isfinite(model.vision_encoder.weight.grad).all()


@pytest.mark.parametrize("variant", ["B1", "B2"])
def test_eval_deterministic_and_visual_off(variant):
    model = make_model(variant).eval()
    x = batch()
    a = model(x)
    b = model(x)
    assert all(torch.allclose(a[k], b[k]) for k in a)
    c = model(x, visual_off=True)
    # Visual off must have same logits even when image contents change.
    x2 = dict(x)
    x2["pixel_values"] = torch.full_like(x["pixel_values"], 0.2)
    d = model(x2, visual_off=True)
    assert all(torch.allclose(c[k], d[k]) for k in c)


def test_b2_gate_and_zero_after_bias():
    model = make_model("B2").eval()
    with torch.no_grad():
        model.visual_projection.bias.fill_(3)
        model.gate.bias.fill_(10)
    t, v = torch.randn(2, 768), torch.randn(2, 768)
    assert torch.allclose(model.fused_features(t, v, visual_off=True), t)
    gate = torch.sigmoid(model.gate(torch.cat([t, model.visual_projection(v)], dim=-1)))
    assert gate.shape == (2, 1) and torch.all((gate >= 0) & (gate <= 1))


def test_input_rejection():
    model = make_model("B1")
    x = batch()
    x["pixel_values"] = torch.ones(2, 3, 128, 128)
    with pytest.raises(ValueError, match="pixel_values"):
        model(x)
    with pytest.raises(ValueError):
        PooledMultimodal("M1", text_encoder=TinyEncoder(False), vision_encoder=TinyEncoder(True))


def test_multimodal_dataset_id_alignment_and_decode():
    buf = io.BytesIO()
    Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8)).save(buf, "PNG")
    frame = pd.DataFrame([{
        "image_path": "img1", "caption": "Halo", "ocr": "meme",
        "y_hate": 0, "y_appropriate": 1,
    }])
    def processor(images, return_tensors):
        assert images.mode == "RGB"
        return {"pixel_values": torch.zeros(1, 3, 224, 224)}
    ds = MemeMultimodalDataset(frame, {"img1": buf.getvalue()}, TinyTokenizer(), processor)
    row = ds[0]
    assert row["image_path"] == "img1"
    assert row["pixel_values"].shape == (3, 224, 224)
    assert row["labels"].tolist() == [0, 1]
    with pytest.raises(ValueError, match="Missing"):
        MemeMultimodalDataset(frame, {}, TinyTokenizer(), processor)
    bad = MemeMultimodalDataset(frame, {"img1": b"not-an-image"}, TinyTokenizer(), processor)
    with pytest.raises(ValueError, match="Cannot decode"):
        bad[0]


def args(**kw):
    data = dict(variant="B1", epochs=1, full=False, pilot_full_data=False,
                max_train=32, max_validation=16, resume=False)
    data.update(kw)
    return SimpleNamespace(**data)


def test_guard_full_and_pilot():
    checked_args(args())
    checked_args(args(full=True, epochs=5, max_train=None, max_validation=None))
    with pytest.raises(ValueError, match="Full run"):
        checked_args(args(full=True, epochs=5))
    with pytest.raises(ValueError, match="Pilot"):
        checked_args(args(pilot_full_data=True, max_train=None, max_validation=None, epochs=5))
    with pytest.raises(ValueError, match="Resume"):
        checked_args(args(resume=True))


def test_shared_evaluator_can_score_two_task_predictions():
    records = []
    for img, h, a in [("a", 0, 0), ("b", 1, 1)]:
        for name, y in (("hate", h), ("appropriate", a)):
            records.append({"image_path": img, "task": name, "truth": y,
                            "prediction_raw": y, "probability_positive": float(y)})
    scores, out = evaluate_records(pd.DataFrame(records))
    assert scores["selection_macro_f1_raw"] == 1
    assert len(out) == 4


def test_optimizer_step_mm_with_mock_encoders():
    model = make_model("B1")
    from torch.utils.data import DataLoader
    batches = []
    for _ in range(3):
        x = batch(1)
        x["labels"] = torch.tensor([[1, 1]])
        x["image_path"] = ["image-id"]
        batches.append(x)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lambda _: 1.)
    loss = train_epoch_mm(model, batches, optim, sched, torch.device("cpu"), accumulation=2)
    assert np.isfinite(loss)
