import io
import json

import numpy as np
import pandas as pd
from PIL import Image
import torch

from cp2 import noise_eval
from cp2.nlp_data import sha256_file
from cp2.official_source_eval import exclude_duplicate_images


def encoded(color, compression):
    buffer = io.BytesIO()
    Image.new("RGB", (2, 3), color).save(buffer, format="PNG", compress_level=compression)
    return buffer.getvalue()


def test_dedup_uses_rgb_pixels_instead_of_compressed_bytes():
    first = encoded((12, 24, 60), 0)
    same = encoded((12, 24, 60), 9)
    different = encoded((12, 24, 61), 9)
    assert first != same
    images = {"fit": first, "match": same, "heldout": different}
    assert exclude_duplicate_images(images, {"fit"}, {"match", "heldout"}) == {"match"}


def test_official_report_keeps_lock_and_excludes_paired_duplicate_rows(monkeypatch, tmp_path):
    run = tmp_path / "B0_full_seed42"
    run.mkdir()
    config = {"model": "B0", "scope": "full", "seed": 42, "run_id": run.name,
              "provenance": {"manifest_sha256": "fixture"}}
    config_file = run / "config.json"
    config_file.write_text(json.dumps(config))
    (run / "result.json").write_text(json.dumps({"test_evaluated": False, "epochs_completed": 3,
                                                 "best_raw_validation_macro_f1": 0.1}))
    checkpoint = run / "best.pt"
    torch.save({"config_hash": sha256_file(config_file), "model": {}}, checkpoint)
    monkeypatch.setattr(noise_eval, "load_reviewed_model", lambda _: torch.nn.Identity())
    original = pd.DataFrame({"image_path": ["a", "b"], "caption": ["a", "b"], "ocr": ["q", "r"]})
    monkeypatch.setattr(noise_eval, "MemeTextDataset", lambda *args: original)

    def fake_score(*args, split, **kwargs):
        assert split == "test"
        matches = pd.DataFrame([
            {"image_path": image, "task": task, "truth": label, "prediction_raw": label,
             "probability_positive": float(label), "split": split}
            for image, label in (("a", 0), ("b", 1)) for task in ("hate", "appropriate")
        ])
        return {"selection_macro_f1_raw": 1.0}, matches

    monkeypatch.setattr(noise_eval, "evaluate_cv_model", fake_score)
    out = tmp_path / "official"
    noise_eval.evaluate_checkpoint(checkpoint, config, out, {"normal": (original, np.ones((2, 2)))},
                                   type("Tokenizer", (), {"all_special_ids": ()})(), None, None,
                                   "fixture", torch.device("cpu"), 20261119,
                                   official_lock_sha="sealed", excluded_ids={"a"})
    saved = json.loads((out / run.name / "noise_metrics.json").read_text())
    assert saved["split"] == "test" and saved["official_lock_sha256"] == "sealed"
    assert saved["n_memes"] == 2 and saved["deduplicated_n_memes"] == 1
    assert (out / run.name / "test_normal_predictions_PRIVATE.csv").is_file()
