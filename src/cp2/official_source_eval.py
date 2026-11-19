"""Seal all planned checkpoints, then evaluate the untouched official INDOMEME test."""

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
from pathlib import Path

from PIL import Image
import torch

from .mm_data import image_processor
from .nlp_data import get_partitions, sha256_file
from .nlp_evaluation import write_json
from .nlp_runner import tokenizer
from .nlp_cv_eval import load_reviewed_model
from .noise_eval import evaluate_checkpoint
from .operations import GIB, REPO, check_disk, check_gpu, private_path, require_frozen_protocol, worker_lock
from .source_noise import CONDITIONS, noisy_sources


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def pixel_digest(raw):
    with Image.open(io.BytesIO(raw)) as source:
        rgb = source.convert("RGB")
        key = hashlib.sha256(f"{rgb.width}x{rgb.height}:".encode())
        key.update(rgb.tobytes())
        return key.digest()


def exclude_duplicate_images(images, training_ids, test_ids):
    known = {pixel_digest(images[image]) for image in training_ids}
    return {image for image in test_ids if pixel_digest(images[image]) in known}


def planned_checkpoints(study, b0, cp2, protocol):
    for seed in protocol["seeds"]:
        variants = protocol["seed42_variants"] if seed == 42 else protocol["repeated_variants"]
        for variant in variants:
            run_id = f"{variant}_full_seed{seed}"
            if variant == "B0" and seed == 42:
                checkpoint = b0
            elif variant in ("B1", "B2", "M1", "M2", "M3") and seed == 42:
                checkpoint = cp2 / run_id / "best.pt"
            else:
                checkpoint = study / "runs" / run_id / "best.pt"
            config = read(checkpoint.parent / "config.json")
            validation = read(study / "benchmark" / run_id / "noise_metrics.json")
            training = read(checkpoint.parent / "result.json")
            config_sha = sha256_file(checkpoint.parent / "config.json")
            validation_f1 = validation["scores"]["normal"]["selection_macro_f1_raw"]
            if (config["scope"], config["seed"], config["run_id"]) != ("full", seed, run_id):
                raise ValueError("Invalid planned checkpoint provenance")
            if (validation["config_sha256"] != config_sha or training["test_evaluated"] is not False
                    or abs(validation_f1 - training["best_raw_validation_macro_f1"]) > 1e-9):
                raise ValueError("Validation provenance does not match training")
            conditions = list(CONDITIONS)
            if seed == 42 and variant in ("B1", "B2", "M1", "M2", "M3"):
                conditions.append("visual_off")
            yield {
                "run_id": run_id, "variant": variant, "seed": seed, "conditions": conditions,
                "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
                "config_sha256": config_sha, "normal_validation_macro_f1": validation_f1,
            }


def official(args):
    study, workspace, b0, cp2, output = (private_path(path) for path in (
        args.study, args.workspace, args.b0_checkpoint, args.cp2_runs, args.output,
    ))
    protocol_name = "cp2_official_source_study.json"
    protocol_sha = require_frozen_protocol(protocol_name)
    protocol = read(REPO / "configs" / protocol_name)
    if not (study / "summary_all_seeds.json").is_file():
        raise ValueError("Complete the validation study before opening official test")
    lock_path = output / "checkpoints_locked_BEFORE_TEST_PRIVATE.json"
    with worker_lock(workspace):
        check_disk(output, 20 * GIB)
        entries = list(planned_checkpoints(study, b0, cp2, protocol))
        candidates = [entry for entry in entries if entry["seed"] == 42 and entry["variant"] in protocol["cp2_variants"]]
        best = max(row["normal_validation_macro_f1"] for row in candidates)
        tied = [row for row in candidates if row["normal_validation_macro_f1"] == best]
        selected = tied[0] if len(tied) == 1 else min(
            tied, key=lambda row: sum(p.numel() for p in load_reviewed_model(row["variant"]).parameters()),
        )
        expected = {"protocol_sha256": protocol_sha, "entries": entries, "cp2_selected_on_validation": selected["run_id"]}
        if lock_path.exists():
            previous = read(lock_path)
            if any(previous[key] != value for key, value in expected.items()):
                raise ValueError("Existing lock does not match the exact checkpoints and protocol")
        else:
            write_json(lock_path, {**expected, "locked_at_wib": datetime.now(timezone(timedelta(hours=7))).isoformat()})
        lock_sha = sha256_file(lock_path)
        print(f"Checkpoints locked before official test: {len(entries)}; SHA-256={lock_sha}", flush=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.require_gpu and device.type != "cuda":
            raise RuntimeError("GPU required")
        check_gpu(device)
        partitions, manifest = get_partitions(workspace)
        test = partitions["test"]
        if manifest != protocol["manifest_sha256"] or len(test) != protocol["expected_official_test"]:
            raise ValueError("Unexpected official split")
        from master_workflow import load_images
        images = load_images(workspace, splits=("train", "test"))
        training_ids = set(partitions["fit"].image_path) | set(partitions["validation"].image_path)
        test_ids = set(test.image_path)
        if training_ids & test_ids or set(images) != training_ids | test_ids:
            raise ValueError("Dataset image IDs do not match the pinned partitions")
        excluded = exclude_duplicate_images(images, training_ids, test_ids)
        if len(excluded) != protocol["expected_cross_split_test_duplicates"]:
            raise ValueError("Exact pixel duplicate count changed")
        test_images = {image: images[image] for image in test_ids}
        del images
        noise_seed = protocol["noise_seed"]
        conditions = {mode: noisy_sources(test, mode, noise_seed) for mode in CONDITIONS}
        conditions["visual_off"] = conditions["normal"]
        tok, processor = tokenizer(), image_processor()
        for entry in entries:
            checkpoint = Path(entry["checkpoint"])
            config = read(checkpoint.parent / "config.json")
            evaluate_checkpoint(checkpoint, config, output,
                                {mode: conditions[mode] for mode in entry["conditions"]},
                                tok, test_images, processor, manifest, device, noise_seed,
                                official_lock_sha=lock_sha, excluded_ids=excluded)
        write_json(output / "official_completed.json", {
            "lock_sha256": lock_sha, "models": len(entries), "n_test": len(test),
            "deduplicated_n_test": len(test) - len(excluded), "cp2_selected_on_validation": selected["run_id"],
        })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for option in ("study", "workspace", "b0-checkpoint", "cp2-runs", "output"):
        parser.add_argument(f"--{option}", required=True)
    parser.add_argument("--require-gpu", action="store_true")
    official(parser.parse_args())


if __name__ == "__main__":
    main()
