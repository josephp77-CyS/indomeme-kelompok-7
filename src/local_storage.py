"""Disk budgets for local baseline workers; no model/data/protocol changes."""
import os
import shutil
import time
from pathlib import Path

GIB = 2**30
PROJECT = Path(__file__).resolve().parents[1]
CACHE = Path(os.environ.get('HF_HUB_CACHE', Path.home() / '.cache/huggingface/hub'))
RESERVE_CACHE = 10 * GIB
RESERVE_PROJECT = 12 * GIB
# Conservative encoder upper bounds, including room for heads and serialization.
PARAMETERS = {'SwinV2': 32_000_000, 'ViT': 92_000_000, 'CLIP': 92_000_000,
              'IndoBERT': 130_000_000, 'IndoBERTweet': 130_000_000, 'XLM-R': 285_000_000}


def check(condition=None):
    parameters = sum(PARAMETERS[condition[key]] for key in ['image', 'text'] if condition.get(key)) if condition else 0
    # Full-size old/new optimizer checkpoints coexist during atomic replacement.
    checkpoint_peak = parameters * 32 + GIB if parameters else 0
    download_peak = parameters * 8 + GIB if parameters else 0
    cache_free = shutil.disk_usage(CACHE).free
    project_free = shutil.disk_usage(PROJECT).free
    return {'allowed': cache_free >= RESERVE_CACHE + download_peak and project_free >= RESERVE_PROJECT + checkpoint_peak,
            'cache_free': cache_free, 'project_free': project_free,
            'cache_reserve': RESERVE_CACHE, 'project_reserve': RESERVE_PROJECT,
            'estimated_download_peak': download_peak, 'estimated_checkpoint_peak': checkpoint_peak}


def monitor(process):
    while process.poll() is None:
        if not check()['allowed']:
            process.terminate()
            process.wait()
            return False
        time.sleep(1)
    return True
