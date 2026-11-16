"""B1/B2 pooled-vision CP2 training. Run from repo root with PYTHONPATH=src.

No official test inference, no automatic heavy training, and no GitHub data upload.
Separate run directories from B0 prevent accidental checkpoint overwrites.
CV01 design remains subject to CV/NLP joint review.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .mm_data import MemeMultimodalDataset, image_processor, load_train_images
from .mm_models import PooledMultimodal
from .nlp_data import MAX_LENGTH, get_partitions, sha256_file
from .nlp_evaluation import evaluate_model, write_json
from .nlp_runner import environment_snapshot, make_provenance, model_loss, seed_all, selection_value, tokenizer


def checked_args(args):
    if args.variant not in PooledMultimodal.VARIANTS:
        raise ValueError("Variant must be B1 or B2")
    if args.epochs <= 0 or args.max_train is not None and args.max_train <= 0 or args.max_validation is not None and args.max_validation <= 0:
        raise ValueError("Epochs and optional sample bounds must be positive")
    if args.pilot_full_data and args.full:
        raise ValueError("Pilot and full cannot be combined")
    if args.full and (args.epochs != 5 or args.max_train is not None or args.max_validation is not None):
        raise ValueError("Full run requires exactly --epochs 5 and all data")
    if args.pilot_full_data and (args.epochs != 1 or args.max_train is not None or args.max_validation is not None):
        raise ValueError("Pilot requires one epoch and all data")
    if args.resume and not args.full:
        raise ValueError("Resume only supported for full mode")


def train_epoch_mm(model, loader, optimizer, scheduler, device, accumulation=8):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for step, batch in enumerate(loader):
        inputs = {k: v.to(device) for k, v in batch.items() if k not in ("labels", "image_path")}
        loss = model_loss(model(inputs), batch["labels"].to(device))
        if not torch.isfinite(loss).item():
            raise RuntimeError("Non-finite loss: STOP training")
        group_size = min(accumulation, len(loader) - (step // accumulation) * accumulation)
        (loss / group_size).backward()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % accumulation == 0 or step + 1 == len(loader):
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    return float(np.mean(losses))


def run_training(args):
    checked_args(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_gpu and device.type != "cuda":
        raise RuntimeError("GPU required; no visible CUDA device")
    seed_all(args.seed)
    partitions, digest = get_partitions(args.workspace)
    fit, val = partitions["fit"], partitions["validation"]
    scope = "full" if args.full else ("pilot_full_data" if args.pilot_full_data else "smoke")
    if scope == "smoke":
        fit = fit.sample(n=min(args.max_train or 32, len(fit)), random_state=args.seed).reset_index(drop=True)
        val = val.sample(n=min(args.max_validation or 16, len(val)), random_state=args.seed).reset_index(drop=True)

    out = Path(args.output).expanduser().resolve() / f"{args.variant}_{scope}_seed{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    # Provenance includes runtime, source checksums and pinned revisions, never credentials.
    provenance = make_provenance(digest)
    provenance["multimodal_source_sha256"] = {
        fn: sha256_file(Path(__file__).resolve().parent / fn)
        for fn in ("mm_data.py", "mm_models.py", "mm_runner.py")
    }
    config = {
        "run_id": out.name, "scope": scope, "variant": args.variant, "seed": args.seed,
        "input": "Caption+OCR+Image", "forbidden_input": "topic",
        "max_length": MAX_LENGTH, "image_size": 224, "epochs": args.epochs,
        "batch_size": 1, "grad_accumulation": 8, "lr": 2e-5,
        "weight_decay": 0.01, "dropout": 0.3, "patience": 2,
        "precision": "float32", "loss_weights": {"hate": 0.5, "appropriate": 0.5},
        "checkpoint_selection": "mean raw validation macro-F1, first epoch wins ties",
        "fit_rows": len(fit), "validation_rows": len(val),
        "image_split_loaded": "train only",
        "provenance": provenance,
    }
    cf = out / "config.json"
    if cf.exists() and json.loads(cf.read_text()) != config:
        # An exact byte-for-byte provenance match is expected on resume.
        raise ValueError("Run config/provenance mismatch; do not overwrite an existing run")
    write_json(cf, config)
    if (out / "result.json").exists():
        raise FileExistsError(f"Completed run cannot be overwritten: {out}")

    # Loading image bytes may download a large pinned TRAIN parquet; no test parquet.
    images = load_train_images(args.workspace)
    needed_ids = set(fit.image_path) | set(val.image_path)
    if not needed_ids.issubset(images):
        raise ValueError("Image IDs missing from pinned TRAIN parquet")
    tok, processor = tokenizer(), image_processor()
    fit_ds = MemeMultimodalDataset(fit, images, tok, processor)
    val_ds = MemeMultimodalDataset(val, images, tok, processor)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    model = PooledMultimodal(args.variant).to(device)
    model_params = model.parameter_summary()
    expected_nonencoder = {"B1": 1183492, "B2": 595205}[args.variant]
    if model_params["non_encoder"] != expected_nonencoder:
        raise RuntimeError(f"CV01 non-encoder parameter mismatch: {model_params}")
    write_json(out / "parameter_summary.json", model_params)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    from transformers import get_linear_schedule_with_warmup
    steps_per_epoch = math.ceil(len(fit_ds) / 8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=steps_per_epoch * args.epochs
    )
    last = out / "last.pt"
    start_epoch, best, bad_epochs, history = 0, float("-inf"), 0, []
    if last.exists():
        if not args.resume:
            raise FileExistsError(f"Found incomplete checkpoint {last}; use --resume after review")
        state = torch.load(last, map_location=device, weights_only=True)
        if state["config_hash"] != sha256_file(cf):
            raise ValueError("Checkpoint config hash mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, best = int(state["epoch_completed"]), float(state["best"])
        bad_epochs, history = int(state["bad_epochs"]), state["history"]
    elif args.resume:
        raise FileNotFoundError(f"Cannot resume: no checkpoint at {last}")

    begin = time.perf_counter()
    for epoch in range(start_epoch, args.epochs):
        if bad_epochs >= 2:
            break
        seed_all(args.seed + epoch)
        gen = torch.Generator().manual_seed(args.seed + epoch)
        fit_loader = DataLoader(fit_ds, batch_size=1, shuffle=True, generator=gen, num_workers=0)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        start = time.perf_counter()
        loss = train_epoch_mm(model, fit_loader, optimizer, scheduler, device)
        scores, predictions = evaluate_model(model, val_loader, device, out.name, args.seed)
        if device.type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        score = selection_value(scores)
        if score > best:
            best, bad_epochs = score, 0
            best_state = {"model": model.state_dict(), "config_hash": sha256_file(cf),
                          "epoch_completed": epoch + 1}
            tmp = out / "best.pt.tmp"
            torch.save(best_state, tmp)
            tmp.replace(out / "best.pt")
            predictions.to_csv(out / "validation_predictions_PRIVATE.csv", index=False)
            write_json(out / "validation_metrics.json", scores)
        else:
            bad_epochs += 1
        history.append({
            "epoch": epoch + 1, "train_loss": loss,
            "raw_validation_selection_macro_f1": score,
            "hate_macro_f1": scores["hate"]["macro_f1"],
            "appropriate_macro_f1": scores["appropriate"]["macro_f1"],
            "appropriate_constrained_macro_f1": scores["appropriate_constrained"]["macro_f1"],
            "seconds": seconds,
            "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
            "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
        })
        write_json(out / "history.json", history)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "config_hash": sha256_file(cf),
                 "epoch_completed": epoch + 1, "best": best, "bad_epochs": bad_epochs,
                 "history": history}
        tmp = out / "last.pt.tmp"
        torch.save(state, tmp)
        tmp.replace(last)
        print(f"{args.variant} epoch {epoch+1}: loss={loss:.4f}, raw-val-macro-F1={score:.4f}, seconds={seconds:.1f}", flush=True)

    result = {"variant": args.variant, "scope": scope, "seed": args.seed,
              "best_raw_validation_macro_f1": best, "epochs_completed": len(history),
              "test_evaluated": False, "total_seconds": time.perf_counter() - begin,
              "parameter_summary": model_params}
    write_json(out / "result.json", result)
    print(f"{args.variant} {scope} finished; output PRIVAT: {out}", flush=True)


def main():
    p = argparse.ArgumentParser(description="Train CV01 B1/B2 CP2 multimodal pooled fusion")
    p.add_argument("--variant", choices=["B1", "B2"], required=True)
    p.add_argument("--workspace", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-train", type=int)
    p.add_argument("--max-validation", type=int)
    p.add_argument("--pilot-full-data", action="store_true")
    p.add_argument("--full", action="store_true")
    p.add_argument("--require-gpu", action="store_true")
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    run_training(args)


if __name__ == "__main__":
    main()
