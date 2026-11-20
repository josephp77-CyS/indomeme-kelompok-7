"""Compare whether to retain, omit, or regenerate a caption under a regeneration cap."""

import argparse
import math

import numpy as np

from .caption_triage import CONDITIONS, SEEDS, load_views, priorities
from .nlp_data import sha256_file
from .nlp_evaluation import write_json
from .operations import private_path
from .recaption import read, restore_caption_rows
from .reliability_summary import score
from .repair_router import SPLITS, average_replays, error_cost, features, fit_router, replay_assignments


CAPS = (0.05, 0.10, 0.20)
SCENARIOS = {"swap_20pct": 0.20, "clean": 0}
COMPARATORS = ("keep", "omit_only", "uncertainty", "original_router", "positive_repair")


def actions(omission, regeneration, tie, limit):
    cheap_omit = omission > 0
    gain = regeneration - np.maximum(omission, 0)
    order = np.lexsort((tie, -gain))
    return cheap_omit, order[gain[order] > 0][:limit]


def compare(truth, available, views, omit_gain, repair_gain, swap_fraction, cap):
    clean, masked, swapped, alternate = (views[name] for name in (*CONDITIONS, "recaption"))
    names = (*COMPARATORS, "three_actions")
    metrics = {name: [] for name in names}
    paired = {name: np.zeros(len(truth)) for name in COMPARATORS}
    limit = math.ceil(cap * len(truth))
    uncertainty = [priorities(view.probabilities, masked.probabilities)["uncertainty"] for view in (clean, swapped)]
    empty = np.empty(0, dtype=int)
    for contaminated, tie, _ in replay_assignments(len(truth), swap_fraction):
        initial = np.where(contaminated[:, None], swapped.predictions, clean.predictions)
        omit = np.where(contaminated, omit_gain[1], omit_gain[0])
        repair = np.where(contaminated, repair_gain[1], repair_gain[0])
        doubt = np.where(contaminated, uncertainty[1], uncertainty[0])
        omit_cheap, combined = actions(omit, repair, tie, limit)
        repair_order = np.lexsort((tie, -repair))[:limit]
        cheap = np.flatnonzero(omit_cheap)
        selections = (
            (empty, empty),
            (cheap, empty),
            (empty, np.lexsort((tie, -doubt))[:limit]),
            (empty, repair_order),
            (empty, repair_order[repair[repair_order] > 0]),
            (cheap, combined),
        )
        errors = {}
        for name, (omitted, requested) in zip(names, selections, strict=True):
            applied = requested[available[requested]]
            prediction = initial.copy()
            prediction[omitted] = masked.predictions[omitted]
            prediction[applied] = alternate.predictions[applied]
            errors[name] = error_cost(truth, prediction)
            metrics[name].append((float(score(truth, prediction)), 100 * errors[name].mean(),
                                  int((truth[:, 0] & ~prediction[:, 0]).sum()),
                                  int((~truth[:, 0] & prediction[:, 0]).sum()),
                                  len(requested), len(applied), len(omitted) - int(np.isin(applied, omitted).sum())))
        for name in COMPARATORS:
            paired[name] += (errors["three_actions"] - errors[name]) / 20
    fields = ("macro_f1", "error_cost_per_100", "missed_hate", "false_hate",
              "regeneration_requests", "regenerations_applied", "omissions_applied")
    return {name: dict(zip(fields, np.mean(rows, axis=0).tolist())) for name, rows in metrics.items()}, paired


def load_case(study, job, split, seed, n, original):
    run_id = f"S1_full_seed{seed}"
    base = study / ("benchmark" if split == "validation" else "official") / run_id
    predictions = job / "router_" / split
    artifact = job / f"{split}_captions_PRIVATE.json"
    receipt = read(predictions / "completed.json")
    lock = sha256_file(study / "official" / "checkpoints_locked_BEFORE_TEST_PRIVATE.json")
    if ((receipt["checkpoint_lock_sha256"], receipt["caption_sha256"], receipt["protocol_sha256"], receipt["split"]) !=
            (lock, sha256_file(artifact), original["protocol_sha256"], split) or run_id not in receipt["run_ids"]):
        raise ValueError("Caption predictions do not match the locked source study")
    ids, truth, views = load_views(base, split, n, seed, run_id)
    new_ids, new_truth, alternatives = load_views(predictions / run_id, split, n, seed, run_id, ("recaption",))
    if not ids.equals(new_ids) or not np.array_equal(truth, new_truth):
        raise ValueError("Case IDs or labels differ across interventions")
    metrics = read(base / "noise_metrics.json")
    if any(not math.isclose(score(truth, view.predictions), metrics["scores"][mode]["selection_macro_f1_raw"],
                            abs_tol=1e-9) for mode, view in views.items()):
        raise ValueError("Base predictions do not reproduce the locked scores")
    captions = restore_caption_rows(artifact, ids, split, metrics["manifest_sha256"])
    return truth, np.array([bool(captions[image_id]) for image_id in ids]), views | alternatives


def summarize(study, job):
    original = read(job / "router_results.json")
    cases = {split: {(name, cap): [] for name in SCENARIOS for cap in CAPS} for split in SPLITS}
    coefficients = {}
    for seed in SEEDS:
        for split, n in SPLITS.items():
            truth, available, views = load_case(study, job, split, seed, n, original)
            if split == "validation":
                omit_model, omit_gain = fit_router(truth, available, views, omit=True)
                repair_model, repair_gain = fit_router(truth, available, views)
                coefficients[str(seed)] = {}
                for name, model in (("omit", omit_model), ("recaption", repair_model)):
                    scaler, ridge = model.steps[0][1], model.steps[1][1]
                    coefficients[str(seed)][name] = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                                                     "coefficients": ridge.coef_.tolist(), "intercept": float(ridge.intercept_)}
            else:
                design = [features(views[mode], views["no_caption"]) for mode in ("normal", "caption_swap")]
                omit_gain = np.stack([omit_model.predict(row) for row in design])
                repair_gain = np.stack([repair_model.predict(row) for row in design])
            original_split = "validation_out_of_fold" if split == "validation" else "official_retrospective"
            for (name, cap), results in cases[split].items():
                result = compare(truth, available, views, omit_gain, repair_gain, SCENARIOS[name], cap)
                if cap == 0.20:
                    stored = original[original_split]["S1"][name]["per_seed"][str(seed)]
                    for reference, control in (("original_router", "learned"), ("keep", "keep"),
                                               ("uncertainty", "uncertainty")):
                        if any(abs(result[0][reference][metric] - stored[control][metric]) > 1e-12
                               for metric in ("macro_f1", "error_cost_per_100")):
                            raise ValueError("Control differs from the frozen 20% experiment")
                results.append(result)
    splits = {}
    budget_comparison = {}
    for split, n in SPLITS.items():
        splits[split] = {name: {} for name in SCENARIOS}
        budget_comparison[split] = {}
        for (name, cap), results in cases[split].items():
            policies = average_replays([(metrics, paired["keep"]) for metrics, paired in results], n)["policies"]
            contrasts = {}
            for reference in COMPARATORS:
                comparison = average_replays([(metrics, paired[reference]) for metrics, paired in results], n)
                contrasts[reference] = {
                    "delta_cost_per_100": comparison["learned_minus_uncertainty_cost_per_100"],
                    "ci95_cost_per_100": comparison["paired_ci95_cost_per_100"],
                }
            splits[split][name][str(round(100 * cap))] = {"policies": policies, "three_actions_minus": contrasts}
        for name in SCENARIOS:
            pairs = []
            for ten, twenty in zip(cases[split][name, 0.10], cases[split][name, 0.20], strict=True):
                # Both budget runs share memes, ranks, replay order, and the keep prediction.
                ten_minus_keep = ten[1]["keep"] - ten[1]["original_router"]
                twenty_minus_keep = twenty[1]["keep"] - twenty[1]["original_router"]
                pairs.append((ten[0], ten_minus_keep - twenty_minus_keep))
            outcome = average_replays(pairs, n, "ten_minus_twenty")
            budget_comparison[split][name] = {
                "delta_cost_per_100": outcome["ten_minus_twenty_cost_per_100"],
                "ci95_cost_per_100": outcome["paired_ci95_cost_per_100"],
            }
    return {
        "scope": "Secondary retrospective S1-only study, frozen validation fit, synthetic corruption",
        "source_protocol_sha256": original["protocol_sha256"],
        "seeds": SEEDS, "replays": 20, "split_sizes": SPLITS, "model_parameters": coefficients, "splits": splits,
        "original_router_budget_10_minus_20_posthoc": budget_comparison,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--study", required=True)
    p.add_argument("--job", required=True)
    p.add_argument("--output", help="New aggregate result path. Defaults to the study directory.")
    args = p.parse_args()
    study, job = map(private_path, (args.study, args.job))
    output = private_path(args.output) if args.output else job / "three_actions_results.json"
    if output.exists():
        raise FileExistsError(output)
    write_json(output, summarize(study, job))


if __name__ == "__main__":
    main()
