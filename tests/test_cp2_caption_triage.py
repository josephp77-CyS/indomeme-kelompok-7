import numpy as np
import pandas as pd
import pytest

from cp2.caption_triage import View, evaluate_replay, load_views


def test_disagreement_routes_swaps_when_confidence_is_unchanged():
    truth = np.array([[False, False], [True, True]] * 5)
    clean = View(np.where(truth, 0.875, 0.125), truth)
    swapped = View(1 - clean.probabilities, ~truth)
    wins, summary = evaluate_replay(truth, {"normal": clean, "no_caption": clean, "caption_swap": swapped})
    assert np.all(wins["disagreement"] == 1)
    assert np.all(wins["uncertainty"] == 0.5)
    assert summary["budget"] == 2
    assert summary["policies"]["disagreement"]["swap_recall"] == 1.0
    assert summary["policies"]["disagreement"]["fallback_macro_f1_delta_pp"] > 0


def test_caption_views_reject_misaligned_gold(tmp_path):
    for mode in ("normal", "no_caption", "caption_swap"):
        pred = pd.DataFrame([
            {"image_path": "meme", "task": task, "run_id": "B2_full_seed42", "split": "test", "seed": 42,
             "truth": int(mode == "caption_swap"), "prediction_raw": 0, "probability_positive": 0.1}
            for task in ("hate", "appropriate")
        ])
        pred.to_csv(tmp_path / f"test_{mode}_predictions_PRIVATE.csv", index=False)
    with pytest.raises(ValueError, match="labels differ"):
        load_views(tmp_path, "test", 1, 42, "B2_full_seed42")
