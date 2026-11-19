"""Paired comparison of S3 against S1 on original, corrupted, and regenerated captions."""

import argparse

import numpy as np

from .caption_triage import CONDITIONS, REPLAY_SEED, SEEDS, load_views
from .nlp_data import sha256_file
from .nlp_evaluation import write_json
from .operations import private_path, require_frozen_protocol
from .recaption import read, restore_caption_rows
from .reliability_summary import score
from .repair_router import SPLITS


def summarize(study, job):
    protocol = require_frozen_protocol("cp2_caption_repair.json")
    lock = sha256_file(job / "S3_checkpoints_locked_BEFORE_REPLAY_PRIVATE.json")
    output = {"protocol_sha256": protocol, "checkpoint_lock_sha256": lock, "seeds": SEEDS}
    for split, n in SPLITS.items():
        base_root = study / ("benchmark" if split == "validation" else "official")
        model_root, reference_root = (job / folder / split for folder in ("classifier_", "router_"))
        artifact = job / f"{split}_captions_PRIVATE.json"
        artifact_sha = sha256_file(artifact)
        for root in (model_root, reference_root):
            report = read(root / "completed.json")
            if (report["protocol_sha256"], report["caption_sha256"], report["split"]) != (protocol, artifact_sha, split):
                raise ValueError("Expected matched, frozen caption inference artifacts")
        if read(model_root / "completed.json")["checkpoint_lock_sha256"] != lock:
            raise ValueError("Classifier checkpoint lock mismatch")
        rng = np.random.default_rng(REPLAY_SEED)
        drawn = rng.integers(0, n, size=(5000, n), dtype=np.int16)
        conditions = (*CONDITIONS, "recaption")
        summaries = {mode: [] for mode in conditions}
        intervals = {mode: [] for mode in conditions}
        for seed in SEEDS:
            reference, candidate = f"S1_full_seed{seed}", f"S3_full_seed{seed}"
            expected_ids, expected_truth, ref = load_views(base_root / reference, split, n, seed, reference)
            alt_ids, alt_truth, alternative = load_views(reference_root / reference, split, n, seed, reference, ("recaption",))
            ids, truth, new = load_views(model_root / candidate, split, n, seed, candidate, conditions)
            if (not ids.equals(expected_ids) or not ids.equals(alt_ids) or
                    not np.array_equal(truth, expected_truth) or not np.array_equal(truth, alt_truth)):
                raise ValueError("Expected identical paired meme IDs and truth")
            report = read(model_root / candidate / "noise_metrics.json")
            available = restore_caption_rows(artifact, ids, split, report["manifest_sha256"])
            ref.update(alternative)
            for mode in conditions:
                # Missing generated captions use the original for both models.
                baseline, current = ref[mode].predictions, new[mode].predictions
                baseline_f1, current_f1 = map(float, (score(truth, baseline), score(truth, current)))
                if abs(current_f1 - report["scores"][mode]["selection_macro_f1_raw"]) > 1e-9:
                    raise ValueError("Stored classifier scores do not match saved predictions")
                summaries[mode].append({"S1": baseline_f1, "S3": current_f1,
                                        "delta_pp": 100 * (current_f1 - baseline_f1)})
                intervals[mode].append(np.concatenate([
                    100 * (score(truth[batch], current[batch]) - score(truth[batch], baseline[batch]))
                    for batch in np.array_split(drawn, 25)
                ]))
        output[split] = {
            "n_memes": n, "usable_generated_captions": sum(map(bool, available.values())),
            "conditions": {
                mode: {
                    "per_seed": dict(zip(SEEDS, results, strict=True)),
                    "mean_delta_pp": float(np.mean([row["delta_pp"] for row in results])),
                    "paired_ci95_pp": np.quantile(np.mean(intervals[mode], axis=0), [0.025, 0.975]).tolist(),
                }
                for mode, results in summaries.items()
            },
        }
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    study, job = map(private_path, (args.study, args.job))
    write_json(job / "classifier_results.json", summarize(study, job))


if __name__ == "__main__":
    main()
