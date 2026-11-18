"""CPU-only synthetic tests for optional private CSV aggregate audit."""
import importlib.util
from pathlib import Path
import json

import pandas as pd
import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "cp2_nlp05_analyze.py"
spec = importlib.util.spec_from_file_location("cp2_nlp05_analyze", SCRIPT)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def sample_files(tmp_path):
    folder = tmp_path / "private"
    folder.mkdir()
    ids = ["a", "b"]
    truth = {"hate": [0, 1], "appropriate": [1, 1]}
    preds = {
        "normal": {"hate": [0, 0], "appropriate": [1, 0]},
        "no_caption": {"hate": [1, 1], "appropriate": [0, 1]},
        "no_ocr": {"hate": [0, 0], "appropriate": [1, 1]},
    }
    scores, frames = {}, {}
    for mode, per_task in preds.items():
        rows = []
        for task, choices in per_task.items():
            for idx, image_id in enumerate(ids):
                p = choices[idx]
                rows.append({"image_path": image_id, "task": task, "truth": truth[task][idx],
                             "prediction_raw": p, "probability_positive": 0.9 if p else 0.1,
                             "run_id": "B0_full_seed42", "seed": 42, "split": "validation"})
        frame = pd.DataFrame(rows)
        frame.to_csv(folder / f"validation_{mode}_predictions_PRIVATE.csv", index=False)
        frames[mode] = frame
        scores[mode] = mod.evaluate_records(frame)[0]
    summary = {"variant": "B0", "run_id": "B0_full_seed42", "seed": 42,
               "split": "validation", "number_of_ids": 2, "test_evaluated": False,
               "scores": {k: {"selection_macro_f1_raw": v["selection_macro_f1_raw"]} for k, v in scores.items()},
               "rescue_hurt_vs_normal": {
                   k: mod.rescue_hurt(frames["normal"], frame)
                   for k, frame in frames.items() if k != "normal"
               }}
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary))
    return summary_path, folder


def test_summary_only_marks_missing_fields(tmp_path):
    source, folder = sample_files(tmp_path)
    data = mod.read_metrics(source)
    report = mod.build_markdown(data)
    assert "—" in report and "Rescue/hurt" in report
    assert "NLP04/NLP05" in report
    assert "| a |" not in report


def test_private_csvs_match_summary_and_aggregate_only(tmp_path):
    source, folder = sample_files(tmp_path)
    data = mod.read_metrics(source)
    detail = mod.read_private_predictions(folder, data)
    assert set(detail) == {"normal", "no_caption", "no_ocr"}
    report = mod.build_markdown(data, detail)
    assert "TN=" in report and "FP=" in report and "FN=" in report
    assert str(folder) not in report and "| a |" not in report


def test_reject_changed_labels_or_mismatched_predictions(tmp_path):
    source, folder = sample_files(tmp_path)
    path = folder / "validation_no_ocr_predictions_PRIVATE.csv"
    frame = pd.read_csv(path)
    frame.loc[0, "truth"] = 1
    frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        mod.read_private_predictions(folder, mod.read_metrics(source))


def test_reject_test_evaluation_flag(tmp_path):
    source, folder = sample_files(tmp_path)
    content = json.loads(source.read_text())
    content["test_evaluated"] = True
    source.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="validation"):
        mod.read_metrics(source)
