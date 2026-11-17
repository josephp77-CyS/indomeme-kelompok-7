from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cp2 import operations


def test_private_path_rejects_repo_root_descendant_and_other_checkout(tmp_path):
    for path in (operations.REPO, operations.REPO / "cp2_runs"):
        with pytest.raises(ValueError, match="outside Git"):
            operations.private_path(path)
    repo = tmp_path / "other"
    repo.mkdir()
    (repo / ".git").write_text("gitdir: elsewhere", encoding="utf-8")
    with pytest.raises(ValueError, match="outside Git"):
        operations.private_path(repo / "runs")
    assert operations.private_path(tmp_path / "private" / "runs") == tmp_path / "private" / "runs"


def test_worker_refuses_concurrent_access_and_releases_after_error(tmp_path):
    with pytest.raises(ValueError):
        with operations.worker_lock(tmp_path):
            with pytest.raises(RuntimeError, match="Another CP2 worker"):
                with operations.worker_lock(tmp_path):
                    pytest.fail("Second worker acquired the lock")
            raise ValueError("run failed")
    with operations.worker_lock(tmp_path):
        pass


def test_checkpoint_space_uses_nearest_existing_directory(tmp_path, monkeypatch):
    def disk_usage(parent):
        assert parent == tmp_path
        return SimpleNamespace(free=12 * operations.GIB)
    monkeypatch.setattr(operations.shutil, "disk_usage", disk_usage)
    with pytest.raises(RuntimeError, match="Disk needs"):
        operations.check_checkpoint_disk(tmp_path / "new" / "run", 200_000_000)


@pytest.mark.parametrize("memory,util,allowed", [(10, 10, True), (9, 0, False), (80, 100, False)])
def test_selected_gpu_needs_free_memory_and_low_utilization(monkeypatch, memory, util, allowed):
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda d: SimpleNamespace(uuid="abc"))
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d: (memory * operations.GIB, 80 * operations.GIB))
    monkeypatch.setattr(Path, "is_file", lambda p: False)
    def measure(command, **kwargs):
        assert command[1:3] == ["-i", "GPU-abc"]
        return str(util)
    monkeypatch.setattr(operations.subprocess, "check_output", measure)
    if allowed:
        operations.check_gpu(torch.device("cuda"))
    else:
        with pytest.raises(RuntimeError, match="GPU needs"):
            operations.check_gpu(torch.device("cuda"))


def test_draft_protocol_refuses_full_run():
    with pytest.raises(RuntimeError, match="joint CV/NLP review"):
        operations.require_frozen_protocol()
