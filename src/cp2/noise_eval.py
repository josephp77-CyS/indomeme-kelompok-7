"""Benchmark missing and synthetically altered text on completed CP2 checkpoints."""

import argparse
import json
from pathlib import Path
import time

import torch
from torch.utils.data import DataLoader

from .mm_data import MemeMultimodalDataset, image_processor, load_train_images
from .nlp_cv_bridge import evaluate_cv_model, VARIANTS
from .nlp_cv_eval import load_reviewed_model
from .nlp_data import MemeTextDataset, get_partitions, sha256_file
from .nlp_evaluation import evaluate_records, rescue_hurt, write_json
from .nlp_runner import tokenizer
from .operations import GIB, check_disk, check_gpu, private_path, worker_lock
from .queue_eval import simulate_review
from .source_model import SOURCE_ARMS, SourceDataset, SourceGate
from .source_noise import CONDITIONS, noisy_sources


class ReviewScores(torch.nn.Module):
    """Collect deployment-visible doubt without using corruption targets or gold labels."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.doubt = []

    def forward(self, inputs):
        outputs, quality = self.model(inputs, return_quality=True)
        reliability = quality.sigmoid().masked_fill(~inputs["source_present"], 0)
        self.doubt.extend((1 - reliability.min(dim=1).values).cpu().tolist())
        return outputs


def benchmark(args):
    workspace, output = private_path(args.workspace), private_path(args.output)
    checkpoints = [private_path(path) for path in args.checkpoints]
    if not all(path.name == "best.pt" and path.is_file() for path in checkpoints):
        raise ValueError("Provide completed run best.pt checkpoints")
    with worker_lock(workspace):
        check_disk(output, 12 * GIB)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.require_gpu and device.type != "cuda":
            raise RuntimeError("GPU required")
        check_gpu(device)
        partitions, manifest = get_partitions(workspace)
        conditions = {mode: noisy_sources(partitions["validation"], mode, args.noise_seed) for mode in CONDITIONS}
        tok = tokenizer()
        configs = [json.loads((path.parent / "config.json").read_text(encoding="utf-8")) for path in checkpoints]
        if any(c.get("variant", c.get("model")) != "B0" for c in configs):
            images, processor = load_train_images(workspace), image_processor()
        else:
            images = processor = None
        for path, config in zip(checkpoints, configs):
            evaluate_checkpoint(path, config, output, conditions, tok, images, processor, manifest, device, args.noise_seed)


def evaluate_checkpoint(checkpoint, config, output, conditions, tok, images, processor, manifest, device, noise_seed,
                        *, official_lock_sha=None, excluded_ids=()):
    variant, seed, run_id = config.get("variant", config.get("model")), config["seed"], config["run_id"]
    if variant not in (*VARIANTS, *SOURCE_ARMS) or config["scope"] != "full" or run_id != f"{variant}_full_seed{seed}":
        raise ValueError("Unexpected variant, scope, seed, or run ID")
    if config["provenance"]["manifest_sha256"] != manifest:
        raise ValueError("Checkpoint split manifest mismatch")
    training = json.loads((checkpoint.parent / "result.json").read_text(encoding="utf-8"))
    if training.get("test_evaluated") is not False or training["epochs_completed"] < 1:
        raise ValueError("Expected completed validation-only training provenance")
    folder = output / run_id
    report = folder / "noise_metrics.json"
    config_sha = sha256_file(checkpoint.parent / "config.json")
    split = "test" if official_lock_sha else "validation"
    if report.exists():
        saved = json.loads(report.read_text(encoding="utf-8"))
        if (saved["config_sha256"], saved["noise_seed"], saved["split"], saved.get("official_lock_sha256")) != (
                config_sha, noise_seed, split, official_lock_sha):
            raise ValueError("Existing benchmark has different provenance")
        print(f"Already evaluated: {run_id}", flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if state["config_hash"] != config_sha:
        raise ValueError("Checkpoint config hash does not match")
    model = SourceGate() if variant in SOURCE_ARMS else load_reviewed_model(variant)
    model.load_state_dict(state["model"], strict=True)
    del state
    check_gpu(device)
    model.to(device)
    reports, predictions, review = {}, {}, {}
    for mode, (frame, quality) in conditions.items():
        start = time.perf_counter()
        if variant == "B0":
            data = MemeTextDataset(frame, tok)
        elif variant in SOURCE_ARMS:
            data = SourceDataset(frame, images, tok, processor, quality)
        else:
            data = MemeMultimodalDataset(frame, images, tok, processor)
        loader = DataLoader(data, batch_size=1)
        evaluated_model = ReviewScores(model) if variant == "S2" else model
        score, prediction = evaluate_cv_model(
            evaluated_model, loader, device, variant, run_id, seed, special_ids=tok.all_special_ids,
            visual_off=mode == "visual_off", split=split,
        )
        if len(prediction) != 2 * len(frame):
            raise ValueError("Unexpected paired evaluation row count")
        score["seconds"] = time.perf_counter() - start
        score["sources_changed"] = {
            source: {"missing": int((quality[:, i] == -1).sum()), "altered": int((quality[:, i] == 0).sum())}
            for i, source in enumerate(("caption", "ocr"))
        }
        reports[mode], predictions[mode] = score, prediction
        if variant == "S2":
            review[mode] = simulate_review(prediction, frame.image_path.astype(str), evaluated_model.doubt)
        prediction.to_csv(folder / f"{split}_{mode}_predictions_PRIVATE.csv", index=False)
        print(f"{run_id} {mode}: {score['selection_macro_f1_raw']:.6f} ({score['seconds']:.1f}s)", flush=True)
    if split == "validation" and abs(reports["normal"]["selection_macro_f1_raw"]
                                     - training["best_raw_validation_macro_f1"]) > 1e-9:
        raise ValueError("Clean validation disagrees with the selected training checkpoint")
    changes = {mode: rescue_hurt(predictions["normal"], p) for mode, p in predictions.items() if mode != "normal"}
    result = {
        "run_id": run_id, "model_seed": seed, "noise_seed": noise_seed, "split": split,
        "n_memes": len(conditions["normal"][0]), "manifest_sha256": manifest,
        "config_sha256": config_sha, "scores": reports, "rescue_hurt": changes, "offline_review_queue": review,
    }
    if official_lock_sha:
        normal = predictions["normal"]
        retained = normal.loc[~normal.image_path.isin(excluded_ids)]
        if len(normal) - len(retained) != 2 * len(excluded_ids):
            raise ValueError("Duplicate exclusion did not match both task labels")
        subset, _ = evaluate_records(retained)
        result.update(official_lock_sha256=official_lock_sha, deduplicated_normal=subset,
                      deduplicated_n_memes=len(retained) // 2)
    write_json(report, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--noise-seed", type=int, default=20261119)
    parser.add_argument("checkpoints", nargs="+", type=Path)
    benchmark(parser.parse_args())


if __name__ == "__main__":
    main()
