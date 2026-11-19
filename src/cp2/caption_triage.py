"""Offline caption triage on saved paired INDOMEME predictions."""

import argparse
import json
import math
from typing import NamedTuple

import numpy as np
import pandas as pd

from .nlp_evaluation import TASKS, write_json
from .operations import private_path
from .reliability_summary import score


SEEDS = (42, 123, 2026)
MODELS = ("B2", "S1")
CONDITIONS = ("normal", "no_caption", "caption_swap")
REPLAY_SEED = 20261119


class View(NamedTuple):
    probabilities: np.ndarray
    predictions: np.ndarray


def load_views(directory, split, n_memes, seed, run_id, conditions=CONDITIONS):
    expected_ids = expected_labels = None
    views = {}
    for condition in conditions:
        frame = pd.read_csv(directory / f"{split}_{condition}_predictions_PRIVATE.csv", dtype={"image_path": str})
        if (set(frame["split"]), set(frame["seed"]), set(frame["run_id"])) != ({split}, {seed}, {run_id}):
            raise ValueError("Unexpected caption view provenance")
        if not frame["truth"].isin((0, 1)).all() or not frame["prediction_raw"].isin((0, 1)).all():
            raise ValueError("Expected binary labels and predictions")
        if not frame["probability_positive"].between(0, 1).all():
            raise ValueError("Expected finite probabilities between zero and one")
        columns = ["truth", "prediction_raw", "probability_positive"]
        table = frame.pivot(index="image_path", columns="task", values=columns).sort_index()
        if set(frame["task"]) != set(TASKS) or table.shape != (n_memes, 6) or table.isna().any().any():
            raise ValueError(f"Expected {n_memes} paired memes with both tasks")
        labs = table["truth"][list(TASKS)].to_numpy(dtype=bool)
        if expected_ids is None:
            expected_ids, expected_labels = table.index, labs
        elif not expected_ids.equals(table.index) or not np.array_equal(expected_labels, labs):
            raise ValueError("Meme IDs or labels differ between caption views")
        views[condition] = View(
            table["probability_positive"][list(TASKS)].to_numpy(),
            table["prediction_raw"][list(TASKS)].to_numpy(dtype=bool),
        )
    return expected_ids, expected_labels, views


def priorities(original, masked):
    return {
        "uncertainty": (1 - np.abs(2 * original - 1)).max(axis=1),
        "disagreement": np.abs(original - masked).max(axis=1),
    }


def evaluate_replay(truth, views, *, replays=20):
    """Mix clean and swapped views without consulting outcomes for case selection."""
    clean, masked, swapped = (views[name] for name in CONDITIONS)
    clean_priority = priorities(clean.probabilities, masked.probabilities)
    swap_priority = priorities(swapped.probabilities, masked.probabilities)
    pair_wins = {name: (1 + np.sign(swap_priority[name] - clean_priority[name])) / 2 for name in clean_priority}
    sample_count = len(truth)
    budget = math.ceil(0.2 * sample_count)
    rng = np.random.default_rng(REPLAY_SEED)
    metrics = {name: [] for name in ("random", "uncertainty", "disagreement")}
    mixed_f1 = []
    for _ in range(replays):
        corrupted = np.zeros(sample_count, dtype=bool)
        corrupted[rng.permutation(sample_count)[:budget]] = True
        tie_break = rng.permutation(sample_count)
        initial = np.where(corrupted[:, None], swapped.predictions, clean.predictions)
        baseline = score(truth, initial)
        mixed_f1.append(baseline)
        rankings = {"random": np.zeros(sample_count)}
        for name in clean_priority:
            rankings[name] = np.where(corrupted, swap_priority[name], clean_priority[name])
        for name, rank in rankings.items():
            routed = np.lexsort((tie_break, -rank))[:budget]
            fallback = initial.copy()
            fallback[routed] = masked.predictions[routed]
            metrics[name].append((
                corrupted[routed].mean(),
                100 * (score(truth, fallback) - baseline),
                int((truth[:, 0] & ~fallback[:, 0]).sum() - (truth[:, 0] & ~initial[:, 0]).sum()),
            ))

    return pair_wins, {
        "budget": budget, "mean_unrouted_macro_f1": float(np.mean(mixed_f1)),
        "policies": {
            name: dict(zip(("swap_recall", "fallback_macro_f1_delta_pp", "fallback_missed_hate_delta"),
                           np.mean(rows, axis=0).tolist()))
            for name, rows in metrics.items()
        },
    }


def evaluate_split(root, split, n_memes, lock_sha=None):
    result = {}
    dataset_ids = dataset_truth = None
    for model in MODELS:
        pair_wins, replays = [], []
        for seed in SEEDS:
            run_id = f"{model}_full_seed{seed}"
            directory = root / run_id
            report = json.loads((directory / "noise_metrics.json").read_text(encoding="utf-8"))
            if (report["split"], report["n_memes"], report["model_seed"], report.get("official_lock_sha256")) != (
                    split, n_memes, seed, lock_sha):
                raise ValueError("Benchmark report provenance mismatch")
            if report["scores"]["caption_swap"]["sources_changed"]["caption"] != {"missing": 0, "altered": n_memes}:
                raise ValueError("Not every caption was swapped")
            ids, truth, views = load_views(directory, split, n_memes, seed, run_id)
            if dataset_ids is None:
                dataset_ids, dataset_truth = ids, truth
            elif not dataset_ids.equals(ids) or not np.array_equal(dataset_truth, truth):
                raise ValueError("Meme IDs or labels differ across models or seeds")
            for condition, view in views.items():
                expected = report["scores"][condition]["selection_macro_f1_raw"]
                if not math.isclose(score(truth, view.predictions), expected, abs_tol=1e-9):
                    raise ValueError("Predictions do not reproduce the recorded Macro-F1")
            wins, replay = evaluate_replay(truth, views)
            pair_wins.append(wins)
            replays.append(replay)

        paired_delta = np.mean([w["disagreement"] - w["uncertainty"] for w in pair_wins], axis=0)
        rng = np.random.default_rng(REPLAY_SEED)
        intervals = np.concatenate([
            100 * paired_delta[rng.integers(0, n_memes, size=(250, n_memes))].mean(axis=1)
            for _ in range(40)
        ])
        policies = replays[0]["policies"]
        result[model] = {
            "paired_win_rates": {name: float(np.mean([w[name] for w in pair_wins])) for name in pair_wins[0]},
            "paired_disagreement_minus_uncertainty_pp": 100 * float(paired_delta.mean()),
            "paired_ci95_pp": np.quantile(intervals, [0.025, 0.975]).tolist(),
            "budget": replays[0]["budget"],
            "mean_unrouted_macro_f1": float(np.mean([r["mean_unrouted_macro_f1"] for r in replays])),
            "policies": {
                name: {metric: float(np.mean([r["policies"][name][metric] for r in replays]))
                       for metric in policies[name]}
                for name in policies
            },
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    study, output = private_path(args.study), private_path(args.output)
    official = json.loads((study / "official" / "official_completed.json").read_text(encoding="utf-8"))
    if official["n_test"] != 1005 or not official["lock_sha256"]:
        raise ValueError("Expected the completed official study")
    report = {
        "model_seeds": SEEDS, "replay_seed": REPLAY_SEED, "replays_per_model_seed": 20,
        "swap_prevalence_fraction": 0.2, "routing_budget_fraction": 0.2,
        "validation_meme_count": 804, "official_meme_count": 1005,
        "scope": "retrospective, synthetic captions, exploratory; replace selected predictions with no-caption predictions",
        "paired_interval": "10000 bootstrap draws over paired meme IDs with model seeds fixed",
        "validation": evaluate_split(study / "benchmark", "validation", 804),
        "official_retrospective": evaluate_split(study / "official", "test", 1005, official["lock_sha256"]),
    }
    write_json(output, report)


if __name__ == "__main__":
    main()
