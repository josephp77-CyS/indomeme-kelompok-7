"""Summarize a source study and estimate paired validation uncertainty."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .nlp_evaluation import TASKS, write_json
from .operations import private_path


COMPARISONS = (
    ("S2", "S1", "caption_swap"), ("S1", "S0", "caption_swap"),
    ("S2", "B2", "caption_swap"), ("S2", "S1", "normal"), ("S2", "B2", "normal"),
)


def score(truth, prediction):
    correct_positive = (truth & prediction).sum(axis=-2)
    correct_negative = (~truth & ~prediction).sum(axis=-2)
    errors = (truth != prediction).sum(axis=-2)
    f1_positive = 2 * correct_positive / np.maximum(2 * correct_positive + errors, 1)
    f1_negative = 2 * correct_negative / np.maximum(2 * correct_negative + errors, 1)
    return ((f1_positive + f1_negative) / 2).mean(axis=-1)


def load_pairs(directory, condition, model, seed):
    frame = pd.read_csv(directory / f"validation_{condition}_predictions_PRIVATE.csv", dtype={"image_path": str})
    if set(frame["split"]) != {"validation"} or set(frame["seed"]) != {seed} or set(frame["run_id"]) != {f"{model}_full_seed{seed}"}:
        raise ValueError("Unexpected paired prediction provenance")
    if not frame["truth"].isin((0, 1)).all() or not frame["prediction_raw"].isin((0, 1)).all():
        raise ValueError("Non-binary truth or predictions")
    paired = frame.pivot(index="image_path", columns="task", values=["truth", "prediction_raw"]).sort_index()
    if paired.shape != (804, 4) or paired.isna().any().any():
        raise ValueError("Expected 804 aligned IDs with both tasks")
    return paired.index, paired["truth"][list(TASKS)].to_numpy(dtype=bool), paired["prediction_raw"][list(TASKS)].to_numpy(dtype=bool)


def summarize(benchmark, seeds):
    rng = np.random.default_rng(20261119)
    sampled = rng.integers(0, 804, size=(10000, 804), dtype=np.int16)
    reports = {}
    scores = {}
    for seed in seeds:
        scores[str(seed)] = {}
        for variant in ("B2", "M2", "S0", "S1", "S2"):
            directory = benchmark / f"{variant}_full_seed{seed}"
            report = json.loads((directory / "noise_metrics.json").read_text(encoding="utf-8"))
            if (report["split"], report["n_memes"], report["model_seed"], report["noise_seed"]) != ("validation", 804, seed, 20261119):
                raise ValueError("Unexpected robustness report provenance")
            reports[variant, seed] = report
            scores[str(seed)][variant] = {mode: data["selection_macro_f1_raw"] for mode, data in report["scores"].items()}
    contrasts = []
    for candidate, reference, condition in COMPARISONS:
        per_seed, distributions = {}, []
        reference_ids = reference_truth = None
        for seed in seeds:
            loaded = [load_pairs(benchmark / f"{model}_full_seed{seed}", condition, model, seed)
                      for model in (candidate, reference)]
            ids, truth, candidate_predictions = loaded[0]
            other_ids, other_truth, reference_predictions = loaded[1]
            if not ids.equals(other_ids) or not np.array_equal(truth, other_truth):
                raise ValueError("IDs or labels differ between compared models")
            if reference_ids is not None and (not ids.equals(reference_ids) or not np.array_equal(truth, reference_truth)):
                raise ValueError("IDs or labels differ between matched seeds")
            reference_ids, reference_truth = ids, truth
            delta = float(score(truth, candidate_predictions) - score(truth, reference_predictions))
            expected = scores[str(seed)][candidate][condition] - scores[str(seed)][reference][condition]
            if abs(delta - expected) > 1e-9:
                raise ValueError("Recomputed paired score disagrees with the evaluation")
            per_seed[str(seed)] = 100 * delta
            boot = []
            for block in np.array_split(sampled, 40):
                boot.append(100 * (score(truth[block], candidate_predictions[block])
                                   - score(truth[block], reference_predictions[block])))
            distributions.append(np.concatenate(boot))
        interval = np.quantile(np.mean(distributions, axis=0), [0.025, 0.975]).tolist()
        contrasts.append({"candidate": candidate, "reference": reference, "condition": condition,
                          "delta_pp_by_seed": per_seed, "mean_delta_pp": float(np.mean(list(per_seed.values()))),
                          "ci95_pp": interval})
    return {
        "split": "validation", "seeds": seeds, "n_memes": 804,
        "interval_method": "10,000 exploratory paired bootstrap draws over memes, conditional on the requested seeds",
        "mean_raw_macro_f1": scores, "contrasts": contrasts,
        "offline_review_queue": {
            str(seed): reports["S2", seed]["offline_review_queue"] for seed in seeds
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Model seeds must be unique")
    write_json(private_path(args.output), summarize(private_path(args.benchmark), args.seeds))


if __name__ == "__main__":
    main()
