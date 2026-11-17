"""Command-line CP2 NLP tools. Run from repo root: PYTHONPATH=src python -m cp2.nlp_runner ..."""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .nlp_data import (
    DATASET_REVISION, EXPECTED_MANIFEST_SHA256, MAX_LENGTH, TEXT_MODEL, TEXT_REVISION,
    VIT_MODEL, VIT_REVISION, MemeTextDataset, audit_tokens, get_partitions, sha256_file,
)
from .nlp_evaluation import TASKS, evaluate_model, write_json
from .nlp_models import TextOnlyB0
from .operations import GIB, check_checkpoint_disk, check_disk, check_gpu, private_path, require_frozen_protocol, worker_lock


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(TEXT_MODEL, revision=TEXT_REVISION)


def environment_snapshot():
    import importlib.metadata
    names = ["torch", "transformers", "pandas", "numpy", "scikit-learn", "huggingface-hub"]
    versions = {}
    for pkg in names:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    gpu = []
    try:
        nvidia_smi = subprocess.check_output([
            "nvidia-smi", "--query-gpu=uuid,name,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits"
        ], stderr=subprocess.DEVNULL, text=True, timeout=8).strip().splitlines()
    except Exception:
        nvidia_smi = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            prop = torch.cuda.get_device_properties(i)
            gpu.append({"name": prop.name, "total_vram_bytes": prop.total_memory})
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        commit = "unknown"
    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "versions": versions, "cuda_runtime": torch.version.cuda,
        "gpu": gpu, "nvidia_smi": nvidia_smi, "git_commit": commit,
    }


def make_provenance(manifest_sha):
    directory = Path(__file__).resolve().parent
    return {
        "dataset_revision": DATASET_REVISION,
        "text_model": TEXT_MODEL, "text_revision": TEXT_REVISION,
        "vit_model": VIT_MODEL, "vit_revision": VIT_REVISION,
        "manifest_sha256": manifest_sha,
        "source_sha256": {p.name: sha256_file(p) for p in directory.glob("nlp_*.py")},
        "environment": {k: v for k, v in environment_snapshot().items() if k != "nvidia_smi"},
    }


def cmd_init_data(args):
    # The existing CP1 functions reproduce and verify the official pinned CSV and historic split.
    from master_workflow import load_metadata, make_split
    workspace = private_path(args.workspace)
    with worker_lock(workspace):
        check_disk(workspace, 12 * GIB)
        metadata = load_metadata(workspace)
        make_split(metadata, workspace)
        partitions, split_hash = get_partitions(workspace)
    print(f"VALID SPLIT {split_hash}: " + ", ".join(f"{k}={len(v)}" for k, v in partitions.items()))


def cmd_audit(args):
    output = private_path(args.output)
    partitions, digest = get_partitions(private_path(args.workspace))
    tok = tokenizer()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    token_report = audit_tokens(partitions, tok, output)
    counts = {}
    for name, frame in partitions.items():
        counts[name] = {
            "rows": len(frame),
            "joint_labels": frame.groupby(["y_hate", "y_appropriate"]).size().rename_axis(["hate", "inappropriate"]).to_dict(),
        }
        counts[name]["joint_labels"] = {f"{a}|{b}": int(v) for (a, b), v in counts[name]["joint_labels"].items()}
    result = {"status": "audited", "counts": counts, "tokens": token_report,
              "provenance": make_provenance(digest)}
    write_json(output / "nlp_input_audit.json", result)
    print(f"AUDIT OK. Output privat: {output / 'nlp_input_audit.json'}")


def model_loss(logits, labels):
    return 0.5 * nn.functional.cross_entropy(logits["hate"], labels[:, 0]) + \
           0.5 * nn.functional.cross_entropy(logits["appropriate"], labels[:, 1])


def selection_value(scores):
    # Always raw; constrained must NEVER be used in CP2 checkpoint selection.
    return float(scores["selection_macro_f1_raw"])


def train_epoch(model, loader, optimizer, scheduler, device, accumulation: int):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    losses = []
    for step, batch in enumerate(loader):
        inputs = {k: v.to(device) for k, v in batch.items() if k not in ("labels", "image_path")}
        loss = model_loss(model(inputs), batch["labels"].to(device))
        if not torch.isfinite(loss).item():
            raise RuntimeError("NaN/Inf loss: STOP training")
        group_size = min(accumulation, len(loader) - (step // accumulation) * accumulation)
        (loss / group_size).backward()
        losses.append(float(loss.detach().cpu()))
        if (step + 1) % accumulation == 0 or step + 1 == len(loader):
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0, error_if_nonfinite=True)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
    return float(np.mean(losses))


def cmd_train_b0(args):
    if args.full and (args.max_train is not None or args.max_validation is not None):
        raise ValueError("Full run tidak boleh mengurangi ukuran fit/validation")
    if args.full and args.epochs != 5:
        raise ValueError("Full run hanya menggunakan 5 epoch maksimum sesuai protokol")
    if args.pilot_full_data and args.full:
        raise ValueError("Pilot dan full harus terpisah")
    if args.pilot_full_data and args.epochs != 1:
        raise ValueError("Pilot seluruh data harus tepat 1 epoch")
    if args.resume and not args.full:
        raise ValueError("Resume only supported for full mode")
    workspace, output = private_path(args.workspace), private_path(args.output)
    protocol_sha = require_frozen_protocol() if args.full else None
    with worker_lock(workspace):
        return train_b0(args, workspace, output, protocol_sha)


def train_b0(args, workspace, output, protocol_sha):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if (args.require_gpu or args.full) and device.type != "cuda":
        raise RuntimeError("GPU required but no CUDA device visible")
    check_gpu(device)
    seed_all(args.seed)
    frames, split_sha = get_partitions(workspace)
    fit, validation = frames["fit"], frames["validation"]
    if not args.full and not args.pilot_full_data:
        # Smoke defaults are applied only in smoke mode. Full and pilot use all rows.
        max_train = args.max_train if args.max_train is not None else 32
        max_validation = args.max_validation if args.max_validation is not None else 16
        fit = fit.sample(n=min(max_train, len(fit)), random_state=args.seed).reset_index(drop=True)
        validation = validation.sample(n=min(max_validation, len(validation)), random_state=args.seed).reset_index(drop=True)
    scope = "full" if args.full else ("pilot_full_data" if args.pilot_full_data else "smoke")
    epochs = args.epochs
    run_id = f"B0_{scope}_seed{args.seed}"
    out = private_path(output / run_id)
    if (out / "result.json").exists():
        raise FileExistsError(f"Run already completed: {out}. Do not overwrite; create separate output")
    check_checkpoint_disk(out, 111_000_000)
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    config = {
        "run_id": run_id, "scope": scope, "seed": args.seed, "model": "B0",
        "input": "Caption+OCR", "forbidden_input": "topic", "max_length": MAX_LENGTH,
        "epochs": epochs, "batch_size": 1, "grad_accumulation": 8,
        "lr": 2e-5, "weight_decay": 0.01, "dropout": 0.3, "patience": 2,
        "precision": "float32", "loss_weights": {"hate": 0.5, "appropriate": 0.5},
        "checkpoint_selection": "mean raw validation macro-F1, first epoch wins ties",
        "fit_rows": len(fit), "validation_rows": len(validation),
        "frozen_protocol_sha256": protocol_sha,
        "provenance": make_provenance(split_sha),
    }
    config_file = out / "config.json"
    if config_file.exists() and json.loads(config_file.read_text()) != config:
        raise ValueError(f"Config changed for existing run {out}; use a fresh private output folder")
    write_json(config_file, config)
    from transformers import get_linear_schedule_with_warmup
    tok = tokenizer()
    fit_ds = MemeTextDataset(fit, tok)
    val_ds = MemeTextDataset(validation, tok)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)
    check_gpu(device)
    model = TextOnlyB0().to(device)
    parameters = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    steps_per_epoch = math.ceil(math.ceil(len(fit_ds) / 1) / 8)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=steps_per_epoch * epochs)
    start_epoch = 0
    best = float("-inf")
    bad_epochs = 0
    history = []
    last_checkpoint = out / "last.pt"
    if last_checkpoint.exists():
        if not args.resume:
            raise FileExistsError(f"Existing checkpoint {last_checkpoint}; add --resume or new output")
        state = torch.load(last_checkpoint, map_location="cpu", weights_only=True)
        if state["config_hash"] != sha256_file(config_file):
            raise ValueError("Resume config hash mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = int(state["epoch_completed"])
        best = float(state["best"])
        bad_epochs = int(state["bad_epochs"])
        history = state["history"]
        del state
    elif args.resume:
        raise FileNotFoundError(f"Cannot resume: no checkpoint at {last_checkpoint}")
    begin = time.perf_counter()
    for epoch in range(start_epoch, epochs):
        if bad_epochs >= 2:
            break
        seed_all(args.seed + epoch)
        generator = torch.Generator().manual_seed(args.seed + epoch)
        fit_loader = DataLoader(fit_ds, batch_size=1, shuffle=True, generator=generator, num_workers=0)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        train_loss = train_epoch(model, fit_loader, optimizer, scheduler, device, 8)
        scores, predictions = evaluate_model(model, val_loader, device, run_id, args.seed)
        score = selection_value(scores)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - start
        check_checkpoint_disk(out, parameters)
        improved = score > best  # earlier epoch wins exact ties
        if improved:
            best = score
            bad_epochs = 0
            state = {"model": model.state_dict(), "config_hash": sha256_file(config_file),
                     "epoch_completed": epoch + 1}
            temp = out / "best.pt.tmp"
            torch.save(state, temp)
            temp.replace(out / "best.pt")
            predictions.to_csv(out / "validation_predictions_PRIVATE.csv", index=False)
            write_json(out / "validation_metrics.json", scores)
        else:
            bad_epochs += 1
        history.append({"epoch": epoch + 1, "train_loss": train_loss,
                        "validation_loss": scores["loss"],
                        "raw_validation_selection_macro_f1": score,
                        "hate_macro_f1": scores["hate"]["macro_f1"],
                        "appropriate_macro_f1": scores["appropriate"]["macro_f1"],
                        "appropriate_constrained_macro_f1": scores["appropriate_constrained"]["macro_f1"],
                        "seconds": elapsed,
                        "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
                        "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None})
        write_json(out / "history.json", history)
        state = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                 "scheduler": scheduler.state_dict(), "config_hash": sha256_file(config_file),
                 "epoch_completed": epoch + 1, "best": best,
                 "bad_epochs": bad_epochs, "history": history}
        temp = out / "last.pt.tmp"
        torch.save(state, temp)
        temp.replace(last_checkpoint)
        print(f"Epoch {epoch+1} loss={train_loss:.4f}, raw-val-macro-F1={score:.4f}, seconds={elapsed:.1f}", flush=True)
    result = {"scope": scope, "run_id": run_id, "best_raw_validation_macro_f1": best,
              "epochs_completed": len(history), "test_evaluated": False,
              "total_seconds": time.perf_counter() - begin}
    write_json(out / "result.json", result)
    print(f"B0 {scope} finished; output PRIVAT: {out}")


def cmd_calibrate(args):
    output = private_path(args.output)
    with worker_lock(args.workspace):
        return calibrate(args, output)


def calibrate(args, output):
    if args.batch <= 0:
        raise ValueError("Batch must be positive")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.require_gpu and device.type != "cuda":
        raise RuntimeError("GPU required but CUDA unavailable")
    check_gpu(device)
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    seed_all(42)
    if args.component == "attention":
        model = nn.MultiheadAttention(768, 8, batch_first=True).to(device)
        # 49 ViT patches, 256 input tokens; these are synthetic tensors, not the full model.
        q = torch.randn(args.batch, 256, 768, device=device, requires_grad=True)
        kv = torch.randn(args.batch, 49, 768, device=device, requires_grad=True)
        def run():
            attended, _ = model(q, kv, kv, need_weights=False)
            mask = torch.ones(args.batch, 256, 1, device=device)
            pooled = (attended * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1)
            return pooled.square().mean()
    elif args.component == "b0":
        model = TextOnlyB0().to(device)
        tok = tokenizer()
        batch = tok(["uji kalibrasi caption [SEP] ocr"] * args.batch,
                    padding="max_length", truncation=True, max_length=256,
                    return_tensors="pt")
        input_batch = {k: v.to(device) for k, v in batch.items() if k in ("input_ids", "attention_mask")}
        def run():
            logits = model(input_batch)
            label = torch.zeros(args.batch, 2, dtype=torch.long, device=device)
            return model_loss(logits, label)
    else:  # ViT only, not multimodal training
        from transformers import ViTModel
        model = ViTModel.from_pretrained(VIT_MODEL, revision=VIT_REVISION).to(device)
        pixels = torch.randn(args.batch, 3, 224, 224, device=device)
        def run():
            result = model(pixel_values=pixels)
            return result.pooler_output.square().mean()
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        free_vram, total_vram = torch.cuda.mem_get_info()
    else:
        free_vram = total_vram = None
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    loss = run()
    if not torch.isfinite(loss).item():
        raise RuntimeError("Non-finite calibration loss")
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - start
    total_params = sum(p.numel() for p in model.parameters())
    report = {
        "component": args.component, "interpretation": "component-only, NOT complete B1/M3 calibration",
        "batch": args.batch, "device": str(device), "loss": float(loss.detach().cpu()),
        "elapsed_seconds_first_optimizer_step_including_initialization_of_optimizer_state": seconds,
        "total_parameters": total_params, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "estimated_model_weights_bytes_fp32": total_params * 4,
        "vram_free_before_step": free_vram, "vram_total": total_vram,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated() if device.type == "cuda" else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved() if device.type == "cuda" else None,
        "disk_free_bytes": shutil.disk_usage(output).free,
        "environment": environment_snapshot(),
    }
    write_json(output / f"calibration_{args.component}_batch{args.batch}.json", report)
    print(json.dumps(report, indent=2))


def make_parser():
    p = argparse.ArgumentParser(description="CP2 NLP implementation: pinned input, evaluation, B0 and component calibration")
    sub = p.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init-data", help="Download pinned CP1 CSV supplement and reconstruct historic manifest")
    init.add_argument("--workspace", required=True)
    init.set_defaults(func=cmd_init_data)
    audit = sub.add_parser("audit", help="Check historic partitions, token lengths and input integrity")
    audit.add_argument("--workspace", required=True)
    audit.add_argument("--output", required=True)
    audit.set_defaults(func=cmd_audit)
    tr = sub.add_parser("train-b0", help="Train text-only two-head CP2 baseline; never evaluates official test")
    tr.add_argument("--workspace", required=True)
    tr.add_argument("--output", required=True)
    tr.add_argument("--seed", type=int, default=42)
    tr.add_argument("--epochs", type=int, default=1)
    tr.add_argument("--max-train", type=int, default=None)
    tr.add_argument("--max-validation", type=int, default=None)
    tr.add_argument("--pilot-full-data", action="store_true", help="One-epoch full fit/validation pilot; NOT final")
    tr.add_argument("--full", action="store_true", help="3214/804 and 5 epochs. Use --epochs 5")
    tr.add_argument("--require-gpu", action="store_true")
    tr.add_argument("--resume", action="store_true")
    tr.set_defaults(func=cmd_train_b0)
    ca = sub.add_parser("calibrate", help="One optimizer step on B0, ViT, or synthetic MHA 49 patches")
    ca.add_argument("--component", choices=["b0", "vision", "attention"], required=True)
    ca.add_argument("--batch", type=int, default=1)
    ca.add_argument("--output", required=True)
    ca.add_argument("--workspace", default=os.environ.get("CP2_WORKSPACE", str(Path.home() / "kelompok-7-private/cp2_data")))
    ca.add_argument("--require-gpu", action="store_true")
    ca.set_defaults(func=cmd_calibrate)
    return p


def main():
    args = make_parser().parse_args()
    if args.command == "train-b0":
        if args.epochs <= 0 or (args.max_train is not None and args.max_train <= 0) or \
                (args.max_validation is not None and args.max_validation <= 0):
            raise ValueError("Positive epochs/max-train/max-validation required")
    args.func(args)


if __name__ == "__main__":
    main()
