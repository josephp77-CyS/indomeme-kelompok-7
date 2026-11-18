"""CPU synthetic acceptance for NLP integrations B0–M3; no HF/GPU calls."""
from __future__ import annotations

import io
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

from cp2.mm_data import MemeMultimodalDataset
from cp2.nlp_cv_bridge import (
    NLPCheckedModel, VARIANTS, evaluate_cv_model, evaluate_robustness,
    model_inputs, validate_cp2_batch,
)
from cp2.nlp_data import MemeTextDataset


class TinyTokenizer:
    sep_token = "[SEP]"
    all_special_ids = [0, 101, 102]

    def __call__(self, texts, **kwargs):
        length = kwargs.get("max_length", 256)
        rows = []
        for s in texts:
            body = [102 if w == "[SEP]" else 10 + len(w) for w in s.split()]
            ids = [101] + body[:length-2] + [102]
            rows.append(ids + [0] * (length-len(ids)))
        x = torch.tensor(rows)
        return {"input_ids": x, "attention_mask": (x != 0).long()}


class TinyCV(nn.Module):
    def __init__(self, visual=True):
        super().__init__()
        self.visual = visual
        self.bias = nn.Parameter(torch.tensor([0.03]))

    def forward(self, features, visual_off=False):
        x = features["input_ids"][:, 1].float() * 0.01 + self.bias
        if self.visual and not visual_off:
            x = x + features["pixel_values"].mean(dim=(1, 2, 3)) * 0.05
        logits = torch.stack([-x, x], dim=-1)
        return {"hate": logits, "appropriate": logits.clone()}


def frame():
    return pd.DataFrame([
        {"image_path": "a", "caption": "dua kata", "ocr": "teks singkat", "y_hate": 0, "y_appropriate": 1},
        {"image_path": "b", "caption": "tiga kata", "ocr": "teks panjang", "y_hate": 1, "y_appropriate": 1},
    ])


def img():
    f = io.BytesIO()
    Image.fromarray(np.zeros((8, 12, 3), dtype=np.uint8)).save(f, "PNG")
    return f.getvalue()


def processor(images, return_tensors):
    assert images.mode == "RGB"
    return {"pixel_values": torch.ones(1, 3, 224, 224) * 0.1}


def get_batch(variant):
    tok = TinyTokenizer()
    if variant == "B0":
        ds = MemeTextDataset(frame(), tok)
    else:
        ds = MemeMultimodalDataset(frame(), {"a": img(), "b": img()}, tok, processor)
    return next(iter(DataLoader(ds, batch_size=2)))


@pytest.mark.parametrize("variant", VARIANTS)
def test_nlp_contract_all_variants(variant):
    b = get_batch(variant)
    assert validate_cp2_batch(b, variant, TinyTokenizer.all_special_ids) == 2
    features = model_inputs(b, variant)
    assert set(features) == (
        {"input_ids", "attention_mask", "content_mask"} if variant == "B0"
        else {"input_ids", "attention_mask", "content_mask", "pixel_values"}
    )
    assert "labels" not in features and "image_path" not in features


@pytest.mark.parametrize("variant", VARIANTS)
def test_shared_nlp03_evaluator(variant):
    scores, records = evaluate_cv_model(
        TinyCV(visual=variant != "B0"), [get_batch(variant)],
        torch.device("cpu"), variant, "synthetic_run", 42,
        special_ids=TinyTokenizer.all_special_ids,
    )
    assert set(scores) >= {"hate", "appropriate", "appropriate_constrained", "selection_macro_f1_raw"}
    assert len(records) == 4 and set(records.image_path) == {"a", "b"}


def test_reject_topic_duplicates_special_token():
    b = get_batch("M1")
    b["topic"] = "leak"
    with pytest.raises(ValueError, match="leaked"):
        validate_cp2_batch(b, "M1")
    del b["topic"]
    b["image_path"] = ["same", "same"]
    with pytest.raises(ValueError, match="duplicate"):
        validate_cp2_batch(b, "M1")
    b["image_path"] = ["a", "b"]
    b["content_mask"][b["input_ids"] == 102] = True
    with pytest.raises(ValueError, match="special"):
        validate_cp2_batch(b, "M1", TinyTokenizer.all_special_ids)


def test_reject_label_relation_and_nonfinite_image():
    b = get_batch("B2")
    b["labels"][0] = torch.tensor([1, 0])
    with pytest.raises(ValueError, match="hate"):
        validate_cp2_batch(b, "B2")
    b["labels"][0] = torch.tensor([0, 1])
    b["pixel_values"][0, 0, 0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_cp2_batch(b, "B2")


def test_visual_off_requires_cv_support():
    class MissingVisualOff(nn.Module):
        def forward(self, features):
            z = torch.zeros(features["input_ids"].shape[0], 2)
            return {"hate": z, "appropriate": z}
    with pytest.raises(TypeError, match="visual_off"):
        NLPCheckedModel(MissingVisualOff(), "M3", visual_off=True)(
            model_inputs(get_batch("M3"), "M3")
        )


def test_reject_incorrect_head_shape():
    class WrongOutput(nn.Module):
        def forward(self, features):
            z = torch.zeros(features["input_ids"].shape[0], 3)
            return {"hate": z, "appropriate": z}
    with pytest.raises(ValueError, match="logits"):
        NLPCheckedModel(WrongOutput(), "B1")(model_inputs(get_batch("B1"), "B1"))


@pytest.mark.parametrize("variant", ("B0", "B1", "M1", "M2", "M3"))
def test_validation_robustness_and_private_predictions(tmp_path, variant):
    kw = {} if variant == "B0" else {"images": {"a": img(), "b": img()}, "processor": processor}
    out = tmp_path / variant
    reports, changes = evaluate_robustness(
        TinyCV(visual=variant != "B0"), variant, frame(), TinyTokenizer(),
        torch.device("cpu"), f"{variant}_full_seed42", 42, out, **kw
    )
    expected = {"normal", "no_caption", "no_ocr"} | (
        {"visual_off"} if variant != "B0" else set()
    )
    assert set(reports) == expected
    assert set(changes) == expected - {"normal"}
    assert (out / "robustness_metrics.json").exists()
    for condition in expected:
        assert (out / f"validation_{condition}_predictions_PRIVATE.csv").exists()
    with pytest.raises(FileExistsError):
        evaluate_robustness(
            TinyCV(visual=variant != "B0"), variant, frame(), TinyTokenizer(),
            torch.device("cpu"), f"{variant}_full_seed42", 42, out, **kw
        )
