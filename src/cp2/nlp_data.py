"""CP2 input contract, dataset validation, tokenization and audit.

Designed to be copied into src/cp2/ in indomeme-kelompok-7.
No dataset or checkpoint is included in this repository package.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset

DATASET_REVISION = "3e0cb8cae9cd726582f69afdb122163d0eab108b"
TEXT_MODEL = "indolem/indobertweet-base-uncased"
TEXT_REVISION = "32e28c05b47e33b6675d2670a1745c50a65e987a"
VIT_MODEL = "google/vit-base-patch32-224-in21k"
VIT_REVISION = "98cb6f7099d372694c8547bf7002289b36857dc7"
EXPECTED_MANIFEST_SHA256 = "4368e5767d73dfe988f9734dde59a146e125a0cda82c2783ee8eff46acab1a39"
EXPECTED_COUNTS = {"fit": 3214, "validation": 804, "test": 1005}
MAX_LENGTH = 256


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def clean_field(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "null"} else text


def compose_text(row: dict, sep_token: str, mode: str = "normal") -> str:
    if not sep_token:
        raise ValueError("Tokenizer harus menyediakan sep_token")
    if mode not in {"normal", "no_caption", "no_ocr"}:
        raise ValueError(f"Mode teks tidak dikenal: {mode}")
    caption = clean_field(row.get("caption")) if mode != "no_caption" else ""
    ocr = clean_field(row.get("ocr")) if mode != "no_ocr" else ""
    if not (caption or ocr):
        raise ValueError(f"Caption dan OCR keduanya kosong pada ID {row.get('image_path', '<unknown>')}")
    return f"{caption} {sep_token} {ocr}" if caption and ocr else caption or ocr


def make_content_mask(input_ids: torch.Tensor, attention_mask: torch.Tensor, special_ids: list[int]) -> torch.Tensor:
    """Exclude PAD, CLS, SEP and any other tokenizer special token, including inner SEP."""
    if input_ids.shape != attention_mask.shape:
        raise ValueError("input_ids dan attention_mask harus mempunyai shape sama")
    special = torch.tensor(sorted(set(int(x) for x in special_ids)), device=input_ids.device, dtype=input_ids.dtype)
    excluded = torch.isin(input_ids, special) if special.numel() else torch.zeros_like(input_ids, dtype=torch.bool)
    return attention_mask.bool() & ~excluded


def get_partitions(workspace: str | Path, enforce_expected: bool = True):
    """Read historic CP1 manifest; reject re-splits and ID/label inconsistencies.

    The dataset's official test rows are returned only for input audit, NOT model selection.
    """
    root = Path(workspace)
    meta_path = root / "data/processed/metadata.csv"
    manifest_path = root / "splits/train_validation.csv"
    if not meta_path.exists() or not manifest_path.exists():
        raise FileNotFoundError(
            "metadata.csv/manifest belum tersedia. Jalankan `python -m cp2.nlp_runner init-data --workspace ...`"
        )
    digest = sha256_file(manifest_path)
    if enforce_expected and digest != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"SHA split tidak cocok: {digest}; expected {EXPECTED_MANIFEST_SHA256}. Jangan split ulang")
    meta = pd.read_csv(meta_path, keep_default_na=False)
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    for c in ["image_path", "caption", "ocr", "y_hate", "y_appropriate", "split"]:
        if c not in meta.columns:
            raise ValueError(f"Metadata tidak mempunyai kolom {c}")
    for c in ["image_path", "partition"]:
        if c not in manifest.columns:
            raise ValueError(f"Manifest tidak mempunyai kolom {c}")
    if not meta.image_path.is_unique or not manifest.image_path.is_unique:
        raise ValueError("image_path duplikat pada metadata/manifest")
    for c in ["y_hate", "y_appropriate"]:
        if not set(meta[c].unique()).issubset({0, 1}):
            raise ValueError(f"Label {c} bukan 0/1")
    if ((meta.y_hate == 1) & (meta.y_appropriate == 0)).any():
        raise ValueError("Label melanggar relasi hate => inappropriate")
    if not set(manifest.partition.unique()) <= {"train", "validation"}:
        raise ValueError("Kolom partition tidak dikenal; harap gunakan manifest historis")
    if {"hate_final", "appropriate_final", "joint_label"}.issubset(meta.columns):
        joined_label = meta["hate_final"].astype(str) + " | " + meta["appropriate_final"].astype(str)
        if not joined_label.equals(meta["joint_label"].astype(str)):
            raise ValueError("joint_label pada metadata berbeda dari pasangan gold labels")
    for _, row in meta.iterrows():
        compose_text(row, "[SEP]")
    if "joint_label" in manifest.columns:
        train_joint = meta.loc[meta.split == "train", ["image_path", "joint_label"]]
        check = manifest.merge(train_joint, on="image_path", validate="one_to_one", suffixes=("_manifest", "_metadata"))
        if len(check) != len(manifest) or not check["joint_label_manifest"].equals(check["joint_label_metadata"]):
            raise ValueError("joint_label manifest tidak sesuai metadata")
    official_train = meta.loc[meta.split == "train"].copy()
    official_test = meta.loc[meta.split == "test"].copy().reset_index(drop=True)
    if set(manifest.image_path) != set(official_train.image_path):
        raise ValueError("ID dalam manifest tidak sama dengan official train")
    joined = official_train.merge(manifest[["image_path", "partition"]], on="image_path", validate="one_to_one")
    frames = {
        "fit": joined.loc[joined.partition == "train"].reset_index(drop=True),
        "validation": joined.loc[joined.partition == "validation"].reset_index(drop=True),
        "test": official_test,
    }
    if enforce_expected:
        for part, count in EXPECTED_COUNTS.items():
            if len(frames[part]) != count:
                raise ValueError(f"Count {part}={len(frames[part])} tidak sama dengan {count}")
    sets = {k: set(f.image_path) for k, f in frames.items()}
    if any(sets[a] & sets[b] for a, b in [("fit", "validation"), ("fit", "test"), ("validation", "test")]):
        raise ValueError("Ada image_path overlap antarsplit")
    return frames, digest


class MemeTextDataset(Dataset):
    """B0 dataset. CV can reuse token fields and append pixel_values externally."""

    def __init__(self, frame: pd.DataFrame, tokenizer, mode: str = "normal", max_length: int = MAX_LENGTH):
        self.frame = frame.reset_index(drop=True)
        self.ids = self.frame.image_path.astype(str).tolist()
        if len(self.ids) != len(set(self.ids)):
            raise ValueError("Duplicate image_path in dataset")
        texts = [compose_text(r, tokenizer.sep_token, mode) for r in self.frame.to_dict("records")]
        self.tokens = tokenizer(
            texts, padding="max_length", truncation=True, max_length=max_length,
            return_tensors="pt", return_token_type_ids=False,
        )
        self.content_mask = make_content_mask(
            self.tokens["input_ids"], self.tokens["attention_mask"], tokenizer.all_special_ids,
        )

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        return {
            "image_path": self.ids[idx],
            "input_ids": self.tokens["input_ids"][idx].long(),
            "attention_mask": self.tokens["attention_mask"][idx].long(),
            "content_mask": self.content_mask[idx],
            "labels": torch.tensor([int(row.y_hate), int(row.y_appropriate)], dtype=torch.long),
        }


def audit_tokens(partitions: dict, tokenizer, output_dir: str | Path, max_length: int = MAX_LENGTH):
    """Audit without training; test text is examined only for input integrity, not labels/scores.

    On fast tokenizers, OCR clipping uses character offsets on the *combined* string;
    if offset mapping is unavailable, OCR clipping is explicitly null.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for part, frame in partitions.items():
        lengths = []
        missing_caption = missing_ocr = 0
        ocr_trimmed_total = 0
        ocr_clipping_available = bool(getattr(tokenizer, "is_fast", False))
        for r in frame.to_dict("records"):
            caption, ocr = clean_field(r.get("caption")), clean_field(r.get("ocr"))
            missing_caption += not bool(caption)
            missing_ocr += not bool(ocr)
            text = compose_text(r, tokenizer.sep_token)
            enc = tokenizer(text, truncation=False, padding=False,
                            return_offsets_mapping=ocr_clipping_available, add_special_tokens=True)
            ids = enc["input_ids"]
            lengths.append(len(ids))
            if ocr_clipping_available and ocr:
                offset = len(caption) + len(f" {tokenizer.sep_token} ") if caption else 0
                ocr_positions = [i for i, (start, end) in enumerate(enc["offset_mapping"])
                                 if end > start and start >= offset]
                ocr_trimmed_total += sum(i >= max_length for i in ocr_positions)
        rows.append({
            "split": part, "rows": len(frame), "missing_caption": int(missing_caption),
            "missing_ocr": int(missing_ocr), "tokens_max": max(lengths, default=0),
            "tokens_above_max": sum(x > max_length for x in lengths),
            "fraction_above_max": sum(x > max_length for x in lengths) / max(1, len(lengths)),
            "ocr_tokens_trimmed_offset_based": ocr_trimmed_total if ocr_clipping_available else None,
            "max_length": max_length,
        })
    out = output_dir / "token_audit.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return rows
