"""Generate paired predictions on the independent image-only captions."""

import argparse
from datetime import datetime, timezone

import torch
from torch.utils.data import DataLoader

from master_workflow import load_images

from .caption_triage import MODELS, SEEDS
from .mm_data import MemeMultimodalDataset, image_processor
from .nlp_cv_bridge import evaluate_cv_model
from .nlp_cv_eval import load_reviewed_model
from .nlp_data import get_partitions, sha256_file
from .nlp_evaluation import write_json
from .nlp_runner import tokenizer
from .noise_eval import evaluate_checkpoint
from .operations import GIB, check_disk, check_gpu, private_path, require_frozen_protocol, worker_lock
from .recaption import read, restore_caption_rows
from .source_model import SourceDataset, SourceGate
from .source_noise import noisy_sources


def lock_entry(path):
    config_path = path.parent / "config.json"
    result = read(path.parent / "result.json")
    if result["test_evaluated"] is not False or result["epochs_completed"] < 1:
        raise ValueError("Expected completed validation-only training")
    return {
        "run_id": path.parent.name, "checkpoint": str(path), "checkpoint_sha256": sha256_file(path),
        "config_sha256": sha256_file(config_path),
        "normal_validation_macro_f1": result["best_raw_validation_macro_f1"],
    }


def execute(args):
    workspace, study, job = map(private_path, (args.workspace, args.study, args.job))
    protocol = require_frozen_protocol("cp2_caption_repair.json")
    official = study / "official"
    original_lock = official / "checkpoints_locked_BEFORE_TEST_PRIVATE.json"
    original_sha = sha256_file(original_lock)
    if original_sha != read(official / "official_completed.json")["lock_sha256"]:
        raise ValueError("Original ledger hash differs from the completed benchmark")
    models = ("S3",) if args.new_model else MODELS
    runs = {f"{model}_full_seed{seed}" for model in models for seed in SEEDS}
    if args.new_model:
        lock = job / "S3_checkpoints_locked_BEFORE_REPLAY_PRIVATE.json"
        entries = [lock_entry(job / "runs" / run_id / "best.pt") for run_id in sorted(runs)]
        identity = {"protocol_sha256": protocol, "entries": entries}
        if lock.exists():
            if any(read(lock).get(key) != value for key, value in identity.items()):
                raise ValueError("S3 checkpoints changed after freeze")
        else:
            if args.split != "validation":
                raise ValueError("Freeze completed S3 checkpoints during validation first")
            write_json(lock, {**identity, "locked_at": datetime.now(timezone.utc).isoformat()})
    else:
        lock = original_lock
        entries = [entry for entry in read(lock)["entries"] if entry["run_id"] in runs]
    if len(entries) != len(runs) or {entry["run_id"] for entry in entries} != runs:
        raise ValueError("Unexpected model list in checkpoint ledger")
    lock_sha = sha256_file(lock)
    split = args.split
    output = job / ("classifier_" if args.new_model else "router_") / split
    report_path = output / "completed.json"
    artifact = job / f"{split}_captions_PRIVATE.json"
    provenance = {"checkpoint_lock_sha256": lock_sha, "caption_sha256": sha256_file(artifact),
                  "protocol_sha256": protocol, "split": split, "run_ids": sorted(runs)}
    if report_path.exists():
        if read(report_path) != provenance:
            raise ValueError("Completed caption inference provenance mismatch")
        return

    with worker_lock(workspace):
        device = torch.device("cuda")
        if not torch.cuda.is_available():
            raise RuntimeError("GPU required for caption inference")
        check_gpu(device)
        check_disk(output, 12 * GIB)
        partitions, manifest = get_partitions(workspace)
        original = partitions[split]
        captions = restore_caption_rows(artifact, original.image_path.astype(str), split, manifest)
        repaired, quality = noisy_sources(original, "normal", 20261119)
        replacement = repaired.image_path.astype(str).map(captions)
        replaced = replacement.ne("")
        repaired.loc[replaced, "caption"] = replacement[replaced]
        quality[replaced.to_numpy(), 0] = 0
        images = load_images(workspace, splits=(("test",) if split == "test" else ("train",)))
        tok, processor = tokenizer(), image_processor()
        for entry in entries:
            run_id = entry["run_id"]
            checkpoint = private_path(entry["checkpoint"])
            hashes = (sha256_file(checkpoint), sha256_file(checkpoint.parent / "config.json"))
            if hashes != (entry["checkpoint_sha256"], entry["config_sha256"]):
                raise ValueError(f"Checkpoint or config changed for {run_id}")
            config = read(checkpoint.parent / "config.json")
            variant, seed = config["variant"], config["seed"]
            if (config["run_id"] != run_id or config["scope"] != "full" or
                    config["provenance"]["manifest_sha256"] != manifest):
                raise ValueError("Checkpoint config provenance mismatch")
            if args.new_model:
                conditions = {mode: noisy_sources(original, mode, 20261119)
                              for mode in ("normal", "no_caption", "caption_swap")}
                conditions["recaption"] = repaired, quality
                evaluate_checkpoint(checkpoint, config, output, conditions, tok, images, processor, manifest,
                                    device, 20261119, official_lock_sha=lock_sha if split == "test" else None)
                continue
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if state["config_hash"] != hashes[1]:
                raise ValueError("Checkpoint has a different training config")
            model = SourceGate() if variant == "S1" else load_reviewed_model(variant)
            model.load_state_dict(state["model"], strict=True)
            del state
            model.to(device)
            dataset = (SourceDataset(repaired, images, tok, processor, quality) if variant == "S1"
                       else MemeMultimodalDataset(repaired, images, tok, processor))
            metrics, predictions = evaluate_cv_model(
                model, DataLoader(dataset, batch_size=1), device, variant, run_id, seed,
                special_ids=tok.all_special_ids, split=split,
            )
            folder = output / run_id
            folder.mkdir(mode=0o700, parents=True, exist_ok=True)
            predictions.to_csv(folder / f"{split}_recaption_predictions_PRIVATE.csv", index=False)
            print(f"{run_id} {split} recaption: {metrics['selection_macro_f1_raw']:.6f}", flush=True)
            del model, dataset
        write_json(report_path, provenance)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--workspace", required=True)
    p.add_argument("--study", required=True)
    p.add_argument("--job", required=True)
    p.add_argument("--split", choices=("validation", "test"), required=True)
    p.add_argument("--new-model", action="store_true")
    execute(p.parse_args())


if __name__ == "__main__":
    main()
