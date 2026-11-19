"""Learn whether to keep the displayed caption after regenerating one at the same routing budget."""

import argparse

import numpy as np

from .caption_triage import SEEDS, load_views
from .nlp_evaluation import write_json
from .operations import private_path
from .recaption import read, restore_caption_rows
from .repair_router import SPLITS, average_replays, confirmation_features, features, fit_router, replay


def confirm_selected(study, job):
    main = read(job / "router_results.json")
    output = {split: {"swap_20pct": [], "clean": []} for split in SPLITS}
    for seed in SEEDS:
        run_id = f"S1_full_seed{seed}"
        for split, n in SPLITS.items():
            base = study / ("benchmark" if split == "validation" else "official") / run_id
            recaption = job / "router_" / split / run_id
            artifact = job / f"{split}_captions_PRIVATE.json"
            ids, truth, views = load_views(base, split, n, seed, run_id)
            new_ids, new_truth, new_views = load_views(recaption, split, n, seed, run_id, ("recaption",))
            if not ids.equals(new_ids) or not np.array_equal(truth, new_truth):
                raise ValueError("Confirmation requires paired predictions")
            views.update(new_views)
            captions = restore_caption_rows(artifact, ids, split, read(base / "noise_metrics.json")["manifest_sha256"])
            available = np.array([bool(captions[image_id]) for image_id in ids])
            if split == "validation":
                router, learned = fit_router(truth, available, views)
                acceptance, confirmed = fit_router(truth, available, views, confirm=True)
            else:
                clean, swapped, masked, new = (views[mode] for mode in ("normal", "caption_swap", "no_caption", "recaption"))
                learned = np.stack([router.predict(features(view, masked)) for view in (clean, swapped)])
                confirmed = np.stack([acceptance.predict(confirmation_features(view, masked, new))
                                      for view in (clean, swapped)])
            split_key = "validation_out_of_fold" if split == "validation" else "official_retrospective"
            for condition, fraction in (("swap_20pct", 0.2), ("clean", 0)):
                outcome = replay(truth, available, views, learned, fraction, confirmation=confirmed)
                control = main[split_key]["S1"][condition]["per_seed"][str(seed)]["learned"]
                if outcome[0]["learned"] != control:
                    raise ValueError("Confirmation replay did not reproduce frozen routing")
                output[split][condition].append(outcome)
    return {
        "scope": "Secondary S1-only comparison at the identical 20% routing budget; positive predicted utility accepts",
        "seeds": SEEDS, "source_protocol_sha256": main["protocol_sha256"],
        "splits": {
            split: {condition: average_replays(result, SPLITS[split], "confirm_minus_accept_all_routed")
                    for condition, result in rows.items()}
            for split, rows in output.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    study, job = map(private_path, (args.study, args.job))
    write_json(job / "confirmation_results.json", confirm_selected(study, job))


if __name__ == "__main__":
    main()
