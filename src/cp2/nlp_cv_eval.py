"""Validation-only NLP04 robustness CLI for completed CP2 model checkpoints.

Uses the same B0–M3 model classes as training.
Never evaluates official test or retrains the checkpoint.
"""
from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path

import torch

from .nlp_cv_bridge import VARIANTS, evaluate_robustness
from .nlp_data import get_partitions, sha256_file
from .nlp_runner import tokenizer
from .operations import GIB, check_disk, check_gpu, private_path, worker_lock


def load_reviewed_model(variant, factory_spec=None):
    if variant == "B0":
        from .nlp_models import TextOnlyB0
        return TextOnlyB0()
    if factory_spec is None:
        from .mm_models import PooledMultimodal
        return PooledMultimodal(variant)
    if variant not in ("M1", "M2", "M3") or ":" not in factory_spec:
        raise ValueError("A custom CV factory must be module:factory and is only for M1–M3")
    mod, method = factory_spec.split(":", 1)
    model = getattr(importlib.import_module(mod), method)(variant=variant)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("CV factory must return a torch.nn.Module")
    return model


def run(args):
    if args.variant not in VARIANTS:
        raise ValueError("Unknown variant")
    workspace, output = private_path(args.workspace), private_path(args.output)
    with worker_lock(workspace):
        return run_validation(args, workspace, output)


def run_validation(args, workspace, output):
    if (output / "robustness_metrics.json").exists():
        raise FileExistsError("Robustness report already exists; choose new private output")
    check_disk(output, 12 * GIB)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_gpu and device.type != "cuda":
        raise RuntimeError("GPU required; CUDA not available")
    check_gpu(device)
    checkpoint = private_path(args.checkpoint)
    if checkpoint.name != "best.pt" or not checkpoint.is_file():
        raise ValueError("Expected existing private best.pt full checkpoint")
    config_file = checkpoint.parent / "config.json"
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if config.get("scope") != "full" or config.get("seed") != args.seed:
        raise ValueError("Checkpoint scope/seed mismatch")
    if config.get("variant", config.get("model")) != args.variant:
        raise ValueError("Checkpoint variant mismatch")
    run_id = f"{args.variant}_full_seed{args.seed}"
    if config.get("run_id") != run_id:
        raise ValueError("Unexpected full checkpoint run ID")
    partitions, manifest_sha = get_partitions(workspace)
    if config.get("provenance", {}).get("manifest_sha256") != manifest_sha:
        raise ValueError("Training/validation split hash mismatch")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state.get("config_hash") != sha256_file(config_file):
        raise ValueError("Checkpoint does not match config hash")
    if "model" not in state:
        raise ValueError("Missing model state")
    model = load_reviewed_model(args.variant, args.cv_factory)
    model.load_state_dict(state["model"], strict=True)
    del state
    model.to(device)
    images = processor = None
    if args.variant != "B0":
        from .mm_data import image_processor, load_train_images
        images, processor = load_train_images(workspace), image_processor()
    scores, changes = evaluate_robustness(
        model, args.variant, partitions["validation"], tokenizer(), device,
        run_id, args.seed, output, images=images, processor=processor,
    )
    for name, report in scores.items():
        print(f"{name}: mean raw validation Macro-F1={report['selection_macro_f1_raw']:.6f}")
    print("PRIVATE validation-only results:", output)
    return scores, changes


def main():
    parser = argparse.ArgumentParser(description="CP2 B0–M3 NLP validation-only ablation")
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    parser.add_argument("--checkpoint", required=True, help="Private full run best.pt")
    parser.add_argument("--workspace", required=True, help="Pinned CP2 data workspace")
    parser.add_argument("--output", required=True, help="New private output folder outside repo")
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--cv-factory", help="Reviewed M1–M3 constructor module:factory(variant=...)")
    parser.add_argument("--require-gpu", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
