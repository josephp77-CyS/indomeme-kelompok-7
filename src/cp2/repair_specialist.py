"""Test the S3 classifier as the recaption branch of the frozen S1 router."""

import argparse

import numpy as np

from .caption_triage import SEEDS, load_views
from .nlp_data import sha256_file
from .nlp_evaluation import write_json
from .operations import private_path
from .recaption import read, restore_caption_rows
from .reliability_summary import score
from .repair_router import SPLITS, average_replays, error_cost, features, fit_router, replay_assignments


def compare_at_fixed_routing(truth, available, views, specialist, learned, swap_fraction):
    policy_predictions = {"S1_after_recaption": views["recaption"].predictions, "S3_after_recaption": specialist.predictions}
    metrics = {policy: [] for policy in policy_predictions}
    contrast = np.zeros(len(truth))
    for swapped, tie, budget in replay_assignments(len(truth), swap_fraction):
        initial = np.where(swapped[:, None], views["caption_swap"].predictions, views["normal"].predictions)
        rank = np.where(swapped, learned[1], learned[0])
        selected = np.lexsort((tie, -rank))[:budget]
        usable = selected[available[selected]]
        costs = {}
        for policy, replacements in policy_predictions.items():
            final = initial.copy()
            final[usable] = replacements[usable]
            costs[policy] = error_cost(truth, final)
            metrics[policy].append([score(truth, final), 100 * costs[policy].mean(),
                                    int((truth[:, 0] & ~final[:, 0]).sum()), int((~truth[:, 0] & final[:, 0]).sum())])
        contrast += (costs["S3_after_recaption"] - costs["S1_after_recaption"]) / 20
    fields = ("macro_f1", "error_cost_per_100", "missed_hate", "false_hate")
    return {policy: dict(zip(fields, np.mean(rows, axis=0).tolist())) for policy, rows in metrics.items()}, contrast


def summarize(study, job):
    main = read(job / "router_results.json")
    classifier = read(job / "classifier_results.json")
    outcomes = {split: {"swap_20pct": [], "clean": []} for split in SPLITS}
    for seed in SEEDS:
        run_id, candidate = f"S1_full_seed{seed}", f"S3_full_seed{seed}"
        for split, n in SPLITS.items():
            base = study / ("benchmark" if split == "validation" else "official") / run_id
            root = job / "classifier_" / split
            artifact = job / f"{split}_captions_PRIVATE.json"
            provenance = read(root / "completed.json")
            if (provenance["checkpoint_lock_sha256"], provenance["caption_sha256"], provenance["protocol_sha256"]) != (
                    classifier["checkpoint_lock_sha256"], sha256_file(artifact), main["protocol_sha256"]):
                raise ValueError("S3 comparison requires compatible frozen prediction artifacts")
            ids, truth, views = load_views(base, split, n, seed, run_id)
            for variant_root, path_id in ((job / "router_" / split, run_id), (root, candidate)):
                pair_ids, pair_truth, alternative = load_views(variant_root / path_id, split, n, seed, path_id, ("recaption",))
                if not ids.equals(pair_ids) or not np.array_equal(truth, pair_truth):
                    raise ValueError("Specialist comparison requires identical paired labels")
                if path_id == run_id:
                    views.update(alternative)
                else:
                    specialist = alternative["recaption"]
            captions = restore_caption_rows(artifact, ids, split, read(base / "noise_metrics.json")["manifest_sha256"])
            available = np.array([bool(captions[image_id]) for image_id in ids])
            if split == "validation":
                model, learned = fit_router(truth, available, views)
            else:
                learned = np.stack([model.predict(features(views[mode], views["no_caption"]))
                                    for mode in ("normal", "caption_swap")])
            dataset = "validation_out_of_fold" if split == "validation" else "official_retrospective"
            for scenario, fraction in (("swap_20pct", 0.2), ("clean", 0)):
                result = compare_at_fixed_routing(truth, available, views, specialist, learned, fraction)
                expected = main[dataset]["S1"][scenario]["per_seed"][str(seed)]["learned"]
                if any(abs(value - expected[metric]) > 1e-12
                       for metric, value in result[0]["S1_after_recaption"].items()):
                    raise ValueError("Control differs from the frozen S1 router")
                outcomes[split][scenario].append(result)
    return {
        "scope": "Secondary comparison at 20% identical S1 routes; only the classifier after a usable recaption changes.",
        "seeds": SEEDS, "source_protocol_sha256": main["protocol_sha256"],
        "splits": {
            split: {scenario: average_replays(result, SPLITS[split], "S3_minus_S1_after_recaption")
                    for scenario, result in cases.items()}
            for split, cases in outcomes.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    study, job = map(private_path, (args.study, args.job))
    write_json(job / "specialist_results.json", summarize(study, job))


if __name__ == "__main__":
    main()
