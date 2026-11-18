"""Shared CLI boundaries for private CP2 data and compute."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import subprocess

import torch

from .nlp_data import sha256_file

GIB = 1024**3
REPO = Path(__file__).resolve().parents[2]


def private_path(path):
    path = Path(path).expanduser().resolve()
    if path.is_relative_to(REPO) or any((p / ".git").exists() for p in (path, *path.parents)):
        raise ValueError("Store CP2 data and predictions outside Git checkouts")
    return path


@contextmanager
def worker_lock(workspace):
    workspace = private_path(workspace)
    workspace.mkdir(parents=True, exist_ok=True, mode=0o700)
    # shortcut: one worker per shared data workspace; use a scheduler for cross-account GPU reservation.
    with (workspace / ".cp2-worker.lock").open("a+b") as lock:
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another CP2 worker holds the data workspace lock") from exc
        yield


def check_disk(path, minimum_bytes):
    parent = next(p for p in (path, *path.parents) if p.exists())
    free = shutil.disk_usage(parent).free
    if free < minimum_bytes:
        raise RuntimeError(f"Disk needs {minimum_bytes/GIB:.1f} GiB free at {parent}; available {free/GIB:.1f} GiB")


def check_checkpoint_disk(out, parameters):
    check_disk(out, 13 * GIB + parameters * 32)


def check_gpu(device):
    if device.type != "cuda":
        return
    uuid = str(torch.cuda.get_device_properties(device).uuid)
    if not uuid.startswith(("GPU-", "MIG-")):
        uuid = "GPU-" + uuid
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", "-i", uuid, "--query-gpu=utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=8, stderr=subprocess.PIPE,
        )
        utilization = int(raw.strip())
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise RuntimeError("Cannot verify utilization of the selected CUDA device") from exc
    free, _ = torch.cuda.mem_get_info(device)
    if not 0 <= utilization <= 10 or free < 10 * GIB:
        raise RuntimeError(f"GPU needs ≥10 GiB free and ≤10% utilization; got {free/GIB:.2f} GiB / {utilization}%")
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        for line in meminfo.read_text().splitlines():
            if line.startswith("MemAvailable:") and int(line.split()[1]) * 1024 < 8 * GIB:
                raise RuntimeError("GPU worker needs at least 8 GiB available host RAM")


def require_frozen_protocol():
    path = REPO / "configs/cp2_nlp_protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    receipt = protocol.get("freeze_receipt")
    if protocol.get("status") != "FROZEN" or not isinstance(receipt, str) or not receipt.strip():
        raise RuntimeError("Full training requires a frozen protocol and a recorded authorization")
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--", "src", "configs", "scripts"], cwd=REPO, text=True,
    )
    if dirty.strip():
        raise RuntimeError("Commit the reviewed CP2 source and protocol before full training")
    return sha256_file(path)
