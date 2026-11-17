"""Offline CV01 B1–M3 regression tests. No HF downloads or GPU required."""
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
        pooled = scalar * self.weight[None, :]
        hidden = pooled[:, None, :].expand(-1, 50 if self.multimodal else input_ids.shape[1], -1)
        return SimpleNamespace(pooler_output=pooled, last_hidden_state=hidden)


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
        "content_mask": torch.tensor([[False, True, False]] * n),
        "pixel_values": torch.ones(n, 3, 224, 224),
    }


@pytest.mark.parametrize("variant,params", [
    ("B1", 1183492), ("B2", 595205), ("M1", 2365444), ("M2", 2366981), ("M3", 2366981),
])
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
    if variant.startswith("M"):
        assert model.attention.in_proj_weight.grad.abs().sum() > 0
    if variant in ("M2", "M3"):
        assert model.gate.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("variant", PooledMultimodal.VARIANTS)
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
        PooledMultimodal("B0", text_encoder=TinyEncoder(False), vision_encoder=TinyEncoder(True))


def test_cross_attention_excludes_cls_padding_and_empty_content():
    model = make_model("M1").eval()
    t, tokens, vision = torch.randn(2, 768), torch.randn(2, 3, 768), torch.randn(2, 50, 768)
    mask = torch.tensor([[True, False, False], [False, False, False]])
    output = model.attended_features(t, tokens, vision, mask)
    tokens[:, 1:] += 100
    vision[:, 0] += 100
    torch.testing.assert_close(model.attended_features(t, tokens, vision, mask), output)
    assert torch.equal(output[1], t[1])
    with pytest.raises(ValueError, match="49 patches"):
        model.attended_features(t, tokens, vision[:, :-1], mask)


def test_m3_dropout_per_meme_after_projection_and_checkpoint(tmp_path, monkeypatch):
    m2 = make_model("M2").eval()
    with torch.no_grad():
        m2.attention.out_proj.bias.fill_(1)
    checkpoint = tmp_path / "model.pt"
    torch.save(m2.state_dict(), checkpoint)
    m3 = make_model("M3").eval()
    m3.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True), strict=True)
    x = batch()
    for task in ("hate", "appropriate"):
        torch.testing.assert_close(m2(x)[task], m3(x)[task])

    t, tokens, vision = torch.randn(2, 768), torch.randn(2, 3, 768), torch.randn(2, 50, 768)
    expected = m2.attended_features(t, tokens, vision, x["content_mask"])
    monkeypatch.setattr(torch, "rand", lambda b, d, device: torch.tensor([[0.01], [0.99]], device=device))
    m3.train()
    fused = m3.attended_features(t, tokens, vision, x["content_mask"])
    assert torch.equal(fused[0], t[0])
    torch.testing.assert_close(fused[1], expected[1])
    m3.zero_grad()
    sum(v.sum() for v in m3(x, visual_off=True).values()).backward()
    assert m3.vision_encoder.weight.grad is None
    assert m3.attention.out_proj.bias.grad is None
    assert m3.text_encoder.weight.grad is not None


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


def test_m3_resume_matches_uninterrupted_epochs(tmp_path, monkeypatch):
    import json
    import sys
    from cp2 import mm_runner

    rows = pd.DataFrame([
        dict(image_path=str(i), caption="caption", ocr="OCR", y_hate=i % 2, y_appropriate=i % 2)
        for i in range(4)
    ])
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (64, 64, 64)).save(buf, "PNG")
    monkeypatch.setattr(mm_runner, "get_partitions", lambda _: ({"fit": rows[:2], "validation": rows[2:]}, "split"))
    monkeypatch.setattr(mm_runner, "load_train_images", lambda _: dict.fromkeys(rows.image_path, buf.getvalue()))
    monkeypatch.setattr(mm_runner, "image_processor", lambda: lambda **_: {"pixel_values": torch.ones(1, 3, 224, 224)})
    monkeypatch.setattr(mm_runner, "make_provenance", lambda digest: {"manifest_sha256": digest})
    monkeypatch.setattr(mm_runner, "tokenizer", TinyTokenizer)
    class MockMultimodal(PooledMultimodal):
        def __init__(self, variant):
            super().__init__(variant, text_encoder=TinyEncoder(False), vision_encoder=TinyEncoder(True))
    monkeypatch.setattr(mm_runner, "PooledMultimodal", MockMultimodal)
    monkeypatch.setattr(mm_runner, "check_checkpoint_disk", lambda *a: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        get_linear_schedule_with_warmup=lambda optimizer, num_warmup_steps, num_training_steps:
            torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 - step / num_training_steps)
    ))
    run = args(variant="M3", epochs=2, seed=42, require_gpu=False)
    # Use the inner runner to exercise real checkpoint restart in a short CPU fixture.
    mm_runner.train_run(run, tmp_path, tmp_path / "clean", None)
    original_train_epoch = mm_runner.train_epoch_mm
    calls = 0

    def interrupt_second_epoch(*a, **kw):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise InterruptedError("simulated interruption")
        return original_train_epoch(*a, **kw)

    monkeypatch.setattr(mm_runner, "train_epoch_mm", interrupt_second_epoch)
    with pytest.raises(InterruptedError):
        mm_runner.train_run(run, tmp_path, tmp_path / "restart", None)
    monkeypatch.setattr(mm_runner, "train_epoch_mm", original_train_epoch)
    run.resume = True
    mm_runner.train_run(run, tmp_path, tmp_path / "restart", None)
    first = tmp_path / "clean/M3_smoke_seed42"
    second = tmp_path / "restart/M3_smoke_seed42"
    states = [torch.load(d / "last.pt", map_location="cpu", weights_only=True) for d in (first, second)]
    a, b = states
    assert a["epoch_completed"] == b["epoch_completed"] == 2
    assert a["best"] == b["best"] and a["bad_epochs"] == b["bad_epochs"]
    assert a["scheduler"] == b["scheduler"]
    for name, value in a["model"].items():
        torch.testing.assert_close(value, b["model"][name], rtol=0, atol=0)
    assert json.loads((second / "result.json").read_text())["test_evaluated"] is False
