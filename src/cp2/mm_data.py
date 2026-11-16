"""Secure CP2 image batch adapter: only the pinned official TRAIN Parquet.

The fit/validation assignment comes exclusively from the existing NLP manifest.
No image or labels from official test are loaded during training/selection.
"""
from __future__ import annotations

import io
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from torch.utils.data import Dataset

from .nlp_data import MemeTextDataset, VIT_MODEL, VIT_REVISION, sha256_file


def image_processor():
    from transformers import AutoImageProcessor
    return AutoImageProcessor.from_pretrained(VIT_MODEL, revision=VIT_REVISION)


def load_train_images(workspace: str | Path) -> dict[str, bytes]:
    """Use CP1 pinned SHA-checked Parquet; reject missing/duplicate image IDs."""
    from master_workflow import load_images
    # load_images downloads only train parquet when splits=('train',).
    images = load_images(Path(workspace), splits=("train",))
    if not images or not all(isinstance(v, bytes) and v for v in images.values()):
        raise ValueError("Train image mapping empty or contains invalid byte payloads")
    return images


class MemeMultimodalDataset(Dataset):
    """B0 text representation + RGB image. Hard fail on missing/corrupt images."""

    def __init__(self, frame, images: dict[str, bytes], tokenizer, processor, max_length=256):
        self.text = MemeTextDataset(frame, tokenizer, max_length=max_length)
        if len(self.text) != len(set(self.text.ids)):
            raise ValueError("Duplicated image IDs")
        self.images = images
        self.processor = processor
        missing = set(self.text.ids) - set(images)
        if missing:
            raise ValueError(f"Missing {len(missing)} image IDs in pinned TRAIN parquet")
        if not callable(processor):
            raise ValueError("Image processor is required")

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        item = self.text[idx]
        image_id = item["image_path"]
        try:
            with Image.open(io.BytesIO(self.images[image_id])) as image:
                image.load()
                rgb = image.convert("RGB")
                pixels = self.processor(images=rgb, return_tensors="pt")["pixel_values"][0]
        except (OSError, UnidentifiedImageError, ValueError, KeyError) as e:
            raise ValueError(f"Cannot decode/process image ID {image_id!r}") from e
        if pixels.shape != (3, 224, 224) or not pixels.isfinite().all():
            raise ValueError(f"Invalid ViT processed tensor for ID {image_id!r}")
        item["pixel_values"] = pixels.float()
        return item
