import numpy as np
import pandas as pd
import pytest

from cp2.source_noise import noisy_sources


@pytest.fixture
def examples():
    return pd.DataFrame({
        "image_path": ["a", "b", "c"], "caption": ["Deskripsi A", "Deskripsi B", "Deskripsi C"],
        "ocr": ["sosial", "berita", "slogan"], "y_hate": [0, 1, 0], "y_appropriate": [1, 1, 0],
    })


@pytest.mark.parametrize("condition,index,target", [
    ("no_caption", 0, -1), ("caption_swap", 0, 0), ("no_ocr", 1, -1), ("ocr_noise", 1, 0),
])
def test_paired_noise_changes_only_intended_source(examples, condition, index, target):
    altered, labels = noisy_sources(examples, condition, 31)
    preserved = ("caption", "ocr")[1-index]
    assert altered.drop(columns=["caption", "ocr"]).equals(examples.drop(columns=["caption", "ocr"]))
    assert altered[preserved].equals(examples[preserved])
    assert labels[:, index].tolist() == [target] * len(examples)


def test_noise_is_deterministic_per_id_and_label_blind(examples):
    swapped, _ = noisy_sources(examples, "caption_swap", 31)
    reordered = examples.iloc[::-1].assign(y_hate=0)
    again, _ = noisy_sources(reordered, "caption_swap", 31)
    assert swapped.set_index("image_path").caption.equals(again.set_index("image_path").sort_index().caption)
    normal, quality = noisy_sources(examples, "normal", 31)
    assert normal.equals(examples) and np.all(quality == 1)
