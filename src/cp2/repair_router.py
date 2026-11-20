"""Train and evaluate caption routing under a fixed regeneration budget."""

import argparse
import math

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .caption_triage import CONDITIONS, MODELS, REPLAY_SEED, SEEDS, load_views, priorities
from .nlp_data import sha256_file
from .nlp_evaluation import TASKS, write_json
from .operations import private_path, require_frozen_protocol
from .recaption import read, restore_caption_rows
from .reliability_summary import score


WEIGHTS = (0.8, 0.2)
SPLITS = {"validation": 804, "test": 1005}


def error_cost(truth, predicted):
    return ((truth != predicted).sum(axis=-1) + (truth[..., 0] & ~predicted[..., 0])).astype(float)


def features(displayed, masked):
    p, q = displayed.probabilities, masked.probabilities
    return np.concatenate((p, q, np.abs(p - q), np.abs(p - 0.5), np.abs(q - 0.5)), axis=1)


def confirmation_features(displayed, masked, alternate):
    base = features(displayed, masked)
    return np.concatenate((base, alternate.probabilities, np.abs(alternate.probabilities - 0.5),
                           alternate.probabilities - displayed.probabilities), axis=1)


def fit_router(truth, available, views, *, confirm=False, omit=False):
    if confirm and omit:
        raise ValueError("An omission router runs before generation")
    clean, masked, swapped, alternate = (views[name] for name in (*CONDITIONS, "recaption"))
    cases = (clean, swapped)
    design = np.concatenate([confirmation_features(view, masked, alternate) if confirm else features(view, masked)
                             for view in cases])
    action = masked if omit else alternate
    usable = np.ones(len(truth), dtype=bool) if omit else available
    targets = np.concatenate([
        np.where(usable, error_cost(truth, view.predictions) - error_cost(truth, action.predictions), 0)
        for view in cases
    ])
    count = len(truth)
    groups = np.tile(np.arange(count), 2)
    weights = np.repeat(WEIGHTS, count)
    oof = np.empty(len(targets))
    model = make_pipeline(StandardScaler(), Ridge(alpha=10))
    for train, valid in GroupKFold(5).split(design, targets, groups):
        model.fit(design[train], targets[train], ridge__sample_weight=weights[train])
        oof[valid] = model.predict(design[valid])
    model.fit(design, targets, ridge__sample_weight=weights)
    return model, oof.reshape(2, count)


def replay_assignments(n, swap_fraction):
    budget, n_swap = math.ceil(0.2 * n), math.ceil(swap_fraction * n)
    rng = np.random.default_rng(REPLAY_SEED)
    for _ in range(20):
        replaced = np.zeros(n, dtype=bool)
        replaced[rng.permutation(n)[:n_swap]] = True
        yield replaced, rng.permutation(n), budget


def replay(truth, available, views, learned, swap_fraction, confirmation=None):
    clean, masked, swapped, alternate = (views[name] for name in (*CONDITIONS, "recaption"))
    hints_clean, hints_swap = (priorities(view.probabilities, masked.probabilities) for view in (clean, swapped))
    n = len(truth)
    policies = ("keep", "random", "uncertainty", "disagreement", "learned", "learned_omit", "all")
    if confirmation is not None:
        policies += ("learned_confirm",)
    metrics = {policy: [] for policy in policies}
    paired = np.zeros(n)
    for replaced, tie, budget in replay_assignments(n, swap_fraction):
        initial = np.where(replaced[:, None], swapped.predictions, clean.predictions)
        rank = {"random": np.zeros(n), "learned": np.where(replaced, learned[1], learned[0])}
        rank.update({name: np.where(replaced, hints_swap[name], hints_clean[name]) for name in hints_clean})
        choices = {name: np.lexsort((tie, -priority))[:budget] for name, priority in rank.items()}
        choices.update(keep=np.empty(0, dtype=int), all=np.arange(n), learned_omit=choices["learned"])
        if confirmation is not None:
            choices["learned_confirm"] = choices["learned"]
            confirmed = np.where(replaced, confirmation[1], confirmation[0]) > 0
        costs = {}
        initial_cost = error_cost(truth, initial)
        for name in policies:
            selected = choices[name]
            changed = selected if name == "learned_omit" else selected[available[selected]]
            if name == "learned_confirm":
                changed = changed[confirmed[changed]]
            pred = initial.copy()
            pred[changed] = masked.predictions[changed] if name == "learned_omit" else alternate.predictions[changed]
            costs[name] = error_cost(truth, pred)
            change = costs[name] - initial_cost
            missed, false = (truth & ~pred).sum(axis=0), (~truth & pred).sum(axis=0)
            metrics[name].append([score(truth, pred), 100 * costs[name].mean(), *missed, *false,
                                  len(selected), len(changed), int(replaced[selected].sum()),
                                  int((change < 0).sum()), int((change > 0).sum()),
                                  float(change[replaced].sum()), float(change[~replaced].sum())])
        if confirmation is None:
            paired += (costs["learned"] - costs["uncertainty"]) / 20
        else:
            paired += (costs["learned_confirm"] - costs["learned"]) / 20
    fields = ("macro_f1", "error_cost_per_100", "missed_hate", "missed_inappropriate", "false_hate",
              "false_inappropriate", "routed", "usable_action", "synthetic_swaps_routed",
              "lower_cost_memes", "higher_cost_memes", "cost_delta_on_swaps", "cost_delta_on_originals")
    return {name: dict(zip(fields, np.mean(rows, axis=0).tolist())) for name, rows in metrics.items()}, paired


def average_replays(results, n, contrast="learned_minus_uncertainty"):
    rows, paired = zip(*results, strict=True)
    paired = 100 * np.mean(paired, axis=0)
    rng = np.random.default_rng(REPLAY_SEED)
    boot = np.concatenate([paired[rng.integers(0, n, size=(200, n))].mean(axis=1) for _ in range(25)])
    return {
        "policies": {name: {metric: float(np.mean([row[name][metric] for row in rows]))
                             for metric in rows[0][name]} for name in rows[0]},
        f"{contrast}_cost_per_100": float(paired.mean()),
        "paired_ci95_cost_per_100": np.quantile(boot, [0.025, 0.975]).tolist(),
        "per_seed": dict(zip(SEEDS, rows, strict=True)),
    }


def run(args):
    study, job = map(private_path, (args.study, args.job))
    if job.joinpath("router_results.json").exists():
        raise FileExistsError("Router results already completed")
    protocol = require_frozen_protocol("cp2_caption_repair.json")
    lock = sha256_file(study / "official" / "checkpoints_locked_BEFORE_TEST_PRIVATE.json")
    if lock != read(study / "official" / "official_completed.json")["lock_sha256"]:
        raise ValueError("Original ledger hash differs from the completed benchmark")
    summaries = {split: {} for split in SPLITS}
    coverages = {}
    parameter_rows = {}
    for variant in MODELS:
        evaluations = {split: {"swap_20pct": [], "clean": []} for split in SPLITS}
        for seed in SEEDS:
            run_id = f"{variant}_full_seed{seed}"
            for split, n in SPLITS.items():
                stage = study / ("benchmark" if split == "validation" else "official") / run_id
                router_folder = job / "router_" / split
                generated = job / f"{split}_captions_PRIVATE.json"
                provenance = read(router_folder / "completed.json")
                anchor = read(stage / "noise_metrics.json")
                if ((provenance["checkpoint_lock_sha256"], provenance["caption_sha256"], provenance["protocol_sha256"],
                     provenance["split"]) != (lock, sha256_file(generated), protocol, split)
                        or run_id not in provenance["run_ids"]):
                    raise ValueError("Generated prediction provenance mismatch")
                if ((anchor["split"], anchor["noise_seed"], anchor.get("official_lock_sha256")) !=
                        (split, REPLAY_SEED, lock if split == "test" else None)
                        or anchor["scores"]["caption_swap"]["sources_changed"]["caption"] != {"missing": 0, "altered": n}):
                    raise ValueError("Unexpected saved caption-swap benchmark")
                ids, truth, views = load_views(stage, split, n, seed, run_id)
                alt_ids, alt_truth, alt_views = load_views(router_folder / run_id, split, n, seed, run_id, ("recaption",))
                if not ids.equals(alt_ids) or not np.array_equal(truth, alt_truth):
                    raise ValueError("Generated caption prediction IDs or labels differ")
                for name, view in views.items():
                    if not math.isclose(score(truth, view.predictions),
                                        anchor["scores"][name]["selection_macro_f1_raw"], abs_tol=1e-9):
                        raise ValueError("Saved predictions do not reproduce the benchmark score")
                views.update(alt_views)
                captions = restore_caption_rows(generated, ids, split, anchor["manifest_sha256"])
                available = np.array([bool(captions[image_id]) for image_id in ids])
                coverages[split] = int(available.sum())
                if split == "validation":
                    model, learned = fit_router(truth, available, views)
                    scale, ridge = model.steps[0][1], model.steps[1][1]
                    parameter_rows[run_id] = {"mean": scale.mean_.tolist(), "scale": scale.scale_.tolist(),
                                              "coefficients": ridge.coef_.tolist(), "intercept": float(ridge.intercept_)}
                else:
                    learned = np.stack([model.predict(features(views[mode], views["no_caption"]))
                                        for mode in ("normal", "caption_swap")])
                for condition, fraction in (("swap_20pct", 0.2), ("clean", 0)):
                    evaluations[split][condition].append(replay(truth, available, views, learned, fraction))
        for split, n in SPLITS.items():
            summaries[split][variant] = {condition: average_replays(results, n)
                                         for condition, results in evaluations[split].items()}
    write_json(job / "router_models.json", {"tasks": TASKS, "models": parameter_rows, "protocol_sha256": protocol})
    write_json(job / "router_results.json", {
        "protocol_sha256": protocol, "split_size": SPLITS, "generated_caption_coverage": coverages,
        "paired_bootstrap": "5,000 draws across memes, conditional on frozen seeds and replays",
        "validation_out_of_fold": summaries["validation"], "official_retrospective": summaries["test"],
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--job", required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
