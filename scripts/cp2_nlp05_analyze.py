#!/usr/bin/env python3
"""CP2 NLP04/NLP05 aggregate-only audit. Does not train or access official test.

Runs on CPU from a repository checkout. Private per-ID predictions are read only;
only counts and scores may be printed or written in the resulting report.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from cp2.nlp_evaluation import TASKS, evaluate_records, rescue_hurt  # noqa: E402

MODES = ("normal", "no_caption", "no_ocr", "visual_off")


def read_metrics(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("split") != "validation" or data.get("test_evaluated") is not False:
        raise ValueError("Only validation metrics with test_evaluated=false are allowed")
    if data.get("variant") not in {"B0", "B1", "B2", "M1", "M2", "M3"}:
        raise ValueError("Unknown variant")
    if not isinstance(data.get("seed"), int) or data.get("seed") < 0:
        raise ValueError("Invalid seed")
    if not isinstance(data.get("number_of_ids"), int) or data["number_of_ids"] < 1:
        raise ValueError("Missing positive number_of_ids")
    if data.get("run_id") != f'{data["variant"]}_full_seed{data["seed"]}':
        raise ValueError("Only full training checkpoint results are accepted")
    scores = data.get("scores")
    if not isinstance(scores, dict) or "normal" not in scores:
        raise ValueError("Missing normal validation metrics")
    if set(scores) - set(MODES):
        raise ValueError("Unexpected robustness condition")
    if data["variant"] == "B0" and "visual_off" in scores:
        raise ValueError("B0 cannot have visual_off")
    changes = data.get("rescue_hurt_vs_normal", {})
    if set(changes) != set(scores) - {"normal"}:
        raise ValueError("Rescue/hurt conditions do not match evaluated modes")
    for mode, stats in scores.items():
        value = stats.get("selection_macro_f1_raw")
        if not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"Invalid Macro-F1 for {mode}")
    for mode, per_task in changes.items():
        if set(per_task) != set(TASKS):
            raise ValueError(f"Missing rescue/hurt task for {mode}")
        for task, counts in per_task.items():
            if set(counts) != {"rescue", "hurt"} or any(
                not isinstance(v, int) or isinstance(v, bool) or v < 0 or v > data["number_of_ids"]
                for v in counts.values()
            ):
                raise ValueError(f"Invalid rescue/hurt counts for {mode}/{task}")
    return data


def read_private_predictions(folder: Path, data: dict) -> dict:
    """Read exact paired validation rows; return aggregate-only statistics."""
    reports, predictions = {}, {}
    for mode in data["scores"]:
        path = folder / f"validation_{mode}_predictions_PRIVATE.csv"
        frame = pd.read_csv(path, dtype={"image_path": str, "task": str, "run_id": str, "split": str})
        expected = {"image_path", "task", "truth", "prediction_raw", "probability_positive", "run_id", "seed", "split"}
        if not expected.issubset(frame.columns):
            raise ValueError(f"Missing prediction columns for {mode}")
        if len(frame) != 2 * data["number_of_ids"]:
            raise ValueError(f"Unexpected row count for {mode}")
        if frame.image_path.nunique() != data["number_of_ids"]:
            raise ValueError(f"Unexpected image ID count for {mode}")
        if not frame["split"].eq("validation").all():
            raise ValueError(f"Non-validation rows in {mode}")
        if not frame["run_id"].eq(data["run_id"]).all() or not frame["seed"].eq(data["seed"]).all():
            raise ValueError(f"Run/seed mismatch in {mode}")
        scores, validated = evaluate_records(frame)
        published = data["scores"][mode]["selection_macro_f1_raw"]
        if not math.isclose(scores["selection_macro_f1_raw"], published, abs_tol=1e-10):
            raise ValueError(f"CSV metrics differ from published summary for {mode}")
        reports[mode] = scores
        predictions[mode] = validated
    for mode in data["scores"]:
        if mode == "normal":
            continue
        actual = rescue_hurt(predictions["normal"], predictions[mode])
        if actual != data["rescue_hurt_vs_normal"][mode]:
            raise ValueError(f"CSV rescue/hurt mismatch for {mode}")
    return reports


def build_markdown(data: dict, csv_reports: dict | None = None) -> str:
    lines = [
        f'# NLP04/NLP05 — {data["variant"]} seed {data["seed"]} (validation)',
        "",
        f'Run: `{data["run_id"]}` · {data["number_of_ids"]} ID validation · Official test: **tidak digunakan**',
        "",
        '| Kondisi | Mean raw Macro-F1 | Δ vs normal | Hate Macro-F1 | Inappropriate Macro-F1 |',
        '| --- | ---: | ---: | ---: | ---: |',
    ]
    normal = data["scores"]["normal"]["selection_macro_f1_raw"]
    for mode, value in data["scores"].items():
        mean = value["selection_macro_f1_raw"]
        src = (csv_reports or {}).get(mode, value)
        h = src.get("hate", {}).get("macro_f1")
        a = src.get("appropriate", {}).get("macro_f1")
        f = lambda x: f"{x:.6f}" if x is not None else "—"
        lines.append(f'| `{mode}` | {mean:.6f} | {mean-normal:+.6f} | {f(h)} | {f(a)} |')
    lines.extend(["", "## Rescue/hurt dibanding kondisi normal", "",
                  '| Kondisi | Task | Rescue | Hurt | Net rescue − hurt |',
                  '| --- | --- | ---: | ---: | ---: |'])
    for mode, tasks in data["rescue_hurt_vs_normal"].items():
        for task in TASKS:
            v = tasks[task]
            lines.append(f'| `{mode}` | `{task}` | {v["rescue"]} | {v["hurt"]} | {v["rescue"] - v["hurt"]:+d} |')
    if csv_reports:
        lines.extend(["", "## Confusion matrix agregat [TN, FP; FN, TP]", ""])
        for mode, scores in csv_reports.items():
            for task in TASKS:
                (tn, fp), (fn, tp) = scores[task]["confusion_matrix_0_1"]
                lines.append(f'- `{mode}` / `{task}`: TN={tn}, FP={fp}, FN={fn}, TP={tp}')
        lines.extend(["", "CSV privat dipasangkan menurut `(image_path, task)`; skor dan rescue/hurt cocok dengan ringkasan."])
    else:
        lines.extend(["", "Metrik per tugas dan confusion matrix yang tidak tersedia pada JSON ringkasan ditandai **—**.",
                      "Untuk memverifikasinya, gunakan `--predictions-dir` dengan CSV validation privat yang sesuai."])
    lines.extend(["", "**Batas interpretasi:** Ini evaluasi validation, bukan official test;"
                  " ablasi tidak membuktikan sebab-akibat atau signifikansi statistik.",
                  "Tidak ada prediksi per-ID atau path privat pada laporan ini.", ""])
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate CPU-only, aggregate NLP04/NLP05 analysis")
    parser.add_argument("--summary", type=Path, default=REPO / "docs/cp2/metrics/b0_seed42_robustness_summary.json")
    parser.add_argument("--predictions-dir", type=Path, help="Private folder containing validation_*_predictions_PRIVATE.csv")
    parser.add_argument("--output", type=Path, help="Optional Markdown report file (aggregate only)")
    args = parser.parse_args(argv)
    data = read_metrics(args.summary)
    detail = read_private_predictions(args.predictions_dir, data) if args.predictions_dir else None
    report = build_markdown(data, detail)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report, encoding="utf-8")
        print(f"Aggregate report saved: {args.output}")
    else:
        print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
