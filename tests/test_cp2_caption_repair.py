import json

import numpy as np
import pandas as pd
import pytest

from cp2.caption_triage import View
from cp2.recaption import MAX_NEW_TOKENS, MAX_PIXELS, MIN_PIXELS, PROMPT, REVISION, restore_caption_rows
from cp2.repair_router import error_cost, fit_router, replay
from cp2.repair_specialist import compare_at_fixed_routing
from cp2.repair_three_actions import actions
from cp2.source_noise import mixed_condition, noisy_sources, repair_training_captions


def test_repair_training_preserves_nonclean_assignments_and_labels():
    ids = [f"{i:03}" for i in range(100)]
    frame = pd.DataFrame({"image_path": ids, "caption": ids, "ocr": "original", "y_hate": 1})
    prepared, quality = noisy_sources(frame, "mixed", 2026)
    before, targets = prepared.copy(), quality.copy()
    repair_training_captions(prepared, quality, dict.fromkeys(ids, "replacement"), 2026)
    changed = prepared.caption.ne(before.caption)
    assert 10 < changed.sum() < 50
    assert all(mixed_condition(2026, meme) == "normal" for meme in prepared.loc[changed, "image_path"])
    assert prepared.drop(columns="caption").equals(before.drop(columns="caption"))
    assert np.array_equal(quality[~changed], targets[~changed])


def test_caption_artifact_checks_split_and_preserves_blank_fallback(tmp_path):
    path = tmp_path / "generated.json"
    path.write_text(json.dumps({
        "split": "validation", "manifest_sha256": "hash", "captioner_revision": REVISION,
        "prompt": PROMPT, "max_new_tokens": MAX_NEW_TOKENS, "min_pixels": MIN_PIXELS, "max_pixels": MAX_PIXELS,
        "complete": True, "items": {
            "a": {"caption": "Valid", "truncated": False},
            "b": {"caption": "Incomplete", "truncated": True},
        },
    }))
    assert restore_caption_rows(path, ("a", "b"), "validation", "hash") == {"a": "Valid", "b": ""}
    with pytest.raises(ValueError, match="provenance"):
        restore_caption_rows(path, ("a", "b"), "fit", "hash")
    with pytest.raises(ValueError, match="exact split IDs"):
        restore_caption_rows(path, ("a",), "validation", "hash")


def test_router_targets_action_gain_and_keeps_unavailable_replacements():
    truth = np.tile([[False, False], [True, True]], (30, 1))
    original = View(np.where(truth, 0.9, 0.1), truth)
    swapped = View(np.where(~truth, 0.9, 0.1), ~truth)
    noisy_replacement = View(np.where(truth, 0.9, 0.1), truth)
    views = dict(normal=original, no_caption=original, caption_swap=swapped, recaption=noisy_replacement)
    available = np.arange(len(truth)) % 5 != 0
    assert error_cost(truth[:2], ~truth[:2]).tolist() == [2, 3]
    _, cross_fit_rank = fit_router(truth, available, views)
    assert cross_fit_rank[1].mean() > cross_fit_rank[0].mean()
    _, cross_fit_confirmation = fit_router(truth, available, views, confirm=True)
    assert cross_fit_confirmation[1].mean() > cross_fit_confirmation[0].mean()
    results, _ = replay(truth, available, views, cross_fit_rank, 0.2)
    assert results["learned"]["routed"] == results["uncertainty"]["routed"] == 12
    assert results["all"]["usable_action"] == available.sum()
    assert results["learned"]["error_cost_per_100"] < results["uncertainty"]["error_cost_per_100"]
    assert results["learned"]["error_cost_per_100"] >= results["all"]["error_cost_per_100"]
    gate = np.full((2, len(truth)), -1.0)
    checked, _ = replay(truth, available, views, cross_fit_rank, 0.2, confirmation=gate)
    assert checked["learned_confirm"]["routed"] == checked["learned"]["routed"]
    assert checked["learned_confirm"]["usable_action"] == 0
    assert checked["learned_confirm"]["error_cost_per_100"] == checked["keep"]["error_cost_per_100"]
    specialist, contrast = compare_at_fixed_routing(truth, available, views, noisy_replacement, cross_fit_rank, 0.2)
    assert not contrast.any()
    assert specialist["S1_after_recaption"]["macro_f1"] == results["learned"]["macro_f1"]


def test_three_actions_rank_regeneration_against_the_cheaper_action():
    omit = np.array([2.0, -1.0, 1.0, 0.0])
    repair = np.array([2.4, 0.7, 0.1, 0.0])
    tie = np.arange(4)
    cheap, requested = actions(omit, repair, tie, 1)
    assert cheap.tolist() == [True, False, True, False]
    assert requested.tolist() == [1]
    assert actions(omit, repair, tie, 4)[1].tolist() == [1, 0]
