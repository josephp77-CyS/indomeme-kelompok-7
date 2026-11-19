"""NLP-owned shared contract + evaluator for CP2 B0/B1/B2/M1/M2/M3.

CV owns the model/fusion implementation. No official test evaluation here.
"""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .nlp_data import MemeTextDataset, MAX_LENGTH
from .nlp_evaluation import TASKS, evaluate_model, rescue_hurt, write_json
from .source_model import SOURCE_ARMS

VARIANTS = ("B0", "B1", "B2", "M1", "M2", "M3")
TEXT_CONDITIONS = ("normal", "no_caption", "no_ocr")
MODEL_VARIANTS = VARIANTS + SOURCE_ARMS
SOURCE_FIELDS = ("source_ids", "source_mask", "source_present")


def validate_cp2_batch(batch: Mapping, variant: str, special_ids=()):
    if variant not in MODEL_VARIANTS:
        raise ValueError(f"Unknown CP2 variant: {variant}")
    forbidden = {"topic", "joint_label", "truth", "prediction", "y_hate", "y_appropriate"}
    if forbidden & set(batch):
        raise ValueError(f"Forbidden metadata/labels leaked to model input: {forbidden & set(batch)}")
    required = {"input_ids", "attention_mask", "content_mask", "labels", "image_path"}
    if required - set(batch):
        raise ValueError(f"Missing batch fields: {required - set(batch)}")
    ids, mask, content, labels = (batch[k] for k in
                                  ("input_ids", "attention_mask", "content_mask", "labels"))
    if not isinstance(ids, torch.Tensor) or ids.dtype != torch.long or ids.ndim != 2:
        raise ValueError("input_ids must be int64 [B,L]")
    n, length = ids.shape
    if n < 1 or length < 1 or length > MAX_LENGTH:
        raise ValueError(f"Token count must be 1..{MAX_LENGTH}")
    if not isinstance(mask, torch.Tensor) or mask.shape != ids.shape or mask.dtype != torch.long:
        raise ValueError("attention_mask must be int64 [B,L]")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("attention_mask must be binary")
    if not isinstance(content, torch.Tensor) or content.dtype != torch.bool or content.shape != ids.shape:
        raise ValueError("content_mask must be boolean [B,L]")
    if (content & (mask == 0)).any():
        raise ValueError("content_mask includes padding")
    special = tuple(special_ids)
    if special and (content & torch.isin(ids, torch.tensor(special, device=ids.device, dtype=ids.dtype))).any():
        raise ValueError("content_mask includes a special token / separator")
    if not isinstance(labels, torch.Tensor) or labels.shape != (n, 2) or labels.dtype != torch.long:
        raise ValueError("labels must be int64 [B,2]")
    if not torch.all((labels == 0) | (labels == 1)):
        raise ValueError("labels must be binary")
    if ((labels[:, 0] == 1) & (labels[:, 1] == 0)).any():
        raise ValueError("Gold labels violate hate=>inappropriate")
    image_paths = batch["image_path"]
    if not isinstance(image_paths, (tuple, list)) or len(image_paths) != n:
        raise ValueError("image_path must have exactly B entries")
    if any(not isinstance(i, str) or not i.strip() for i in image_paths) or len(set(image_paths)) != n:
        raise ValueError("Empty/duplicate image_path")
    if variant != "B0":
        pixels = batch.get("pixel_values")
        if not isinstance(pixels, torch.Tensor) or pixels.shape != (n, 3, 224, 224):
            raise ValueError("pixel_values must be [B,3,224,224]")
        if not pixels.dtype.is_floating_point or not torch.isfinite(pixels).all():
            raise ValueError("pixel_values must be finite floating point")
    if variant in SOURCE_ARMS:
        source_ids, source_mask, present = (batch.get(key) for key in SOURCE_FIELDS)
        source_shape = (n, 2, MAX_LENGTH // 2)
        if any(not isinstance(field, torch.Tensor) or field.shape != source_shape or field.dtype != torch.long
               for field in (source_ids, source_mask)):
            raise ValueError("Source tokens and masks must be int64 [B,2,128]")
        if not isinstance(present, torch.Tensor) or present.shape != (n, 2) or present.dtype != torch.bool:
            raise ValueError("Source availability must be boolean [B,2]")
    return n


def model_inputs(batch: Mapping, variant: str):
    if variant not in MODEL_VARIANTS:
        raise ValueError("Unknown variant")
    keys = ("input_ids", "attention_mask", "content_mask")
    if variant != "B0":
        keys += ("pixel_values",)
    if variant in SOURCE_ARMS:
        keys += SOURCE_FIELDS
    if set(keys) - set(batch):
        raise ValueError("Missing model inputs")
    return {key: batch[key] for key in keys}


class NLPCheckedModel(nn.Module):
    """Require exactly two task logits; visual_off must be explicit in CV model."""
    def __init__(self, model: nn.Module, variant: str, visual_off=False):
        super().__init__()
        if variant not in MODEL_VARIANTS or (variant in ("B0", *SOURCE_ARMS) and visual_off):
            raise ValueError("Unknown variant or visual_off on B0")
        self.model, self.variant, self.visual_off = model, variant, visual_off

    def forward(self, inputs):
        features = model_inputs(inputs, self.variant)
        if self.visual_off:
            try:
                outputs = self.model(features, visual_off=True)
            except TypeError as exc:
                raise TypeError("CV model must implement forward(..., visual_off=True)") from exc
        else:
            outputs = self.model(features)
        if not isinstance(outputs, Mapping) or set(outputs) != set(TASKS):
            raise ValueError("Model must return exactly hate and appropriate")
        n = features["input_ids"].shape[0]
        for task in TASKS:
            if not isinstance(outputs[task], torch.Tensor) or outputs[task].shape != (n, 2):
                raise ValueError(f"Invalid logits for {task}; expected [B,2]")
            if not torch.isfinite(outputs[task]).all():
                raise ValueError(f"Nonfinite logits for {task}")
        return dict(outputs)


class CheckedLoader:
    def __init__(self, loader, variant, special_ids=()):
        self.loader, self.variant, self.special_ids = loader, variant, tuple(special_ids)

    def __iter__(self):
        for batch in self.loader:
            validate_cp2_batch(batch, self.variant, self.special_ids)
            yield batch

    def __len__(self):
        return len(self.loader)


def evaluate_cv_model(model, loader, device, variant, run_id, seed, *, special_ids=(), visual_off=False,
                      split="validation"):
    wrapped = NLPCheckedModel(model, variant, visual_off=visual_off)
    checked = CheckedLoader(loader, variant, special_ids)
    return evaluate_model(wrapped, checked, device, run_id, seed, split=split)


def evaluate_robustness(model, variant, validation_frame, tokenizer, device,
                        run_id, seed, output_dir, *, images=None, processor=None):
    """Evaluate validation only; no retraining, official test, or model selection."""
    if variant not in VARIANTS:
        raise ValueError("Unknown variant")
    if variant != "B0" and (images is None or processor is None):
        raise ValueError("Multimodal evaluation requires image bytes and pinned processor")
    if validation_frame.empty or not validation_frame.image_path.is_unique:
        raise ValueError("Validation must be nonempty with unique image_path")
    out = Path(output_dir)
    if (out / "robustness_metrics.json").exists():
        raise FileExistsError("Robustness report already exists; choose new private output")
    out.mkdir(parents=True, exist_ok=True, mode=0o700)
    conditions = (*TEXT_CONDITIONS, *(("visual_off",) if variant != "B0" else ()))
    reports, predictions = {}, {}
    for condition in conditions:
        text_mode = condition if condition in TEXT_CONDITIONS else "normal"
        if variant == "B0":
            data = MemeTextDataset(validation_frame, tokenizer, mode=text_mode)
        else:
            from .mm_data import MemeMultimodalDataset
            data = MemeMultimodalDataset(validation_frame, images, tokenizer, processor, mode=text_mode)
        loader = DataLoader(data, batch_size=1, shuffle=False, num_workers=0)
        reports[condition], predictions[condition] = evaluate_cv_model(
            model, loader, device, variant, run_id, seed,
            special_ids=tokenizer.all_special_ids, visual_off=condition == "visual_off"
        )
        predictions[condition].to_csv(
            out / f"validation_{condition}_predictions_PRIVATE.csv", index=False
        )
    changes = {c: rescue_hurt(predictions["normal"], p) for c, p in predictions.items() if c != "normal"}
    write_json(out / "robustness_metrics.json", {
        "variant": variant, "run_id": run_id, "seed": seed,
        "split": "validation", "test_evaluated": False,
        "number_of_ids": len(validation_frame), "scores": reports,
        "rescue_hurt_vs_normal": changes,
    })
    return reports, changes
