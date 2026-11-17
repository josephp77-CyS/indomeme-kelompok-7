"""Two-task CP2 validation (raw checkpoint selection; constrained metrics secondary)."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score

TASKS = ("hate", "appropriate")


def task_scores(truth, prediction):
    y = np.asarray(truth, dtype=int)
    p = np.asarray(prediction, dtype=int)
    if y.shape != p.shape or not np.isin(y, [0, 1]).all() or not np.isin(p, [0, 1]).all():
        raise ValueError("Expected same-shape binary truth/prediction")
    if len(y) == 0:
        raise ValueError("Empty predictions")
    precision, recall, f1, support = precision_recall_fscore_support(
        y, p, labels=[0, 1], average=None, zero_division=0
    )
    return {
        "macro_f1": float(np.mean(f1)), "accuracy": float(accuracy_score(y, p)),
        "confusion_matrix_0_1": confusion_matrix(y, p, labels=[0, 1]).tolist(),
        "classes": {
            str(i): {"precision": float(precision[i]), "recall": float(recall[i]),
                     "f1": float(f1[i]), "support": int(support[i])}
            for i in (0, 1)
        },
    }


def evaluate_records(frame: pd.DataFrame):
    required = {"image_path", "task", "truth", "prediction_raw", "probability_positive"}
    if required - set(frame.columns):
        raise ValueError(f"Missing columns {required - set(frame.columns)}")
    if len(frame) == 0 or frame.duplicated(["image_path", "task"]).any():
        raise ValueError("Empty or duplicate (image_path,task) predictions")
    if set(frame.task.unique()) != set(TASKS):
        raise ValueError("Both tasks are mandatory")
    if frame["probability_positive"].isna().any() or not np.isfinite(frame["probability_positive"]).all():
        raise ValueError("NaN/inf probabilities")
    if not frame["probability_positive"].between(0, 1).all():
        raise ValueError("Probabilities outside [0,1]")
    h = frame[frame.task == "hate"].set_index("image_path")
    a = frame[frame.task == "appropriate"].set_index("image_path")
    if set(h.index) != set(a.index) or len(h) != len(a):
        raise ValueError("Unpaired task IDs")
    if not (h.truth.isin([0, 1]).all() and a.truth.isin([0, 1]).all()
            and h.prediction_raw.isin([0, 1]).all() and a.prediction_raw.isin([0, 1]).all()):
        raise ValueError("Invalid binary labels or predictions")
    a = a.loc[h.index]  # pairing is by image_path, not accidental row order
    if ((h.truth == 1) & (a.truth == 0)).any():
        raise ValueError("Ground-truth relation hate=>inappropriate violated")
    constrained = np.maximum(a.prediction_raw.to_numpy(), h.prediction_raw.to_numpy())
    result = frame.copy()
    result["prediction_constrained"] = result["prediction_raw"].astype(int)
    location = result.task.eq("appropriate")
    corr = pd.Series(constrained, index=a.index)
    result.loc[location, "prediction_constrained"] = result.loc[location, "image_path"].map(corr).to_numpy()
    scores = {
        "hate": task_scores(h.truth, h.prediction_raw),
        "appropriate": task_scores(a.truth, a.prediction_raw),
        "appropriate_constrained": task_scores(a.truth, constrained),
        "raw_constraint_violations": int(((h.prediction_raw == 1) & (a.prediction_raw == 0)).sum()),
        "constrained_constraint_violations": 0,
    }
    scores["selection_macro_f1_raw"] = (
        scores["hate"]["macro_f1"] + scores["appropriate"]["macro_f1"]
    ) / 2
    return scores, result


@torch.no_grad()
def evaluate_model(model, loader, device, run_id: str, seed: int, split="validation"):
    model.eval()
    records = []
    losses = torch.zeros(len(TASKS), device=device)
    for batch in loader:
        labels = batch["labels"].cpu()
        targets = labels.to(device)
        ids = batch["image_path"]
        inputs = {k: v.to(device) for k, v in batch.items() if k not in ("labels", "image_path")}
        logits = model(inputs)
        if set(logits) != set(TASKS):
            raise ValueError(f"Model harus memiliki dua head; got: {list(logits)}")
        for task_idx, task in enumerate(TASKS):
            if logits[task].ndim != 2 or logits[task].shape != (len(ids), 2):
                raise ValueError(f"Invalid logits shape for {task}")
            if not torch.isfinite(logits[task]).all():
                raise ValueError("Non-finite logits")
            losses[task_idx] += torch.nn.functional.cross_entropy(
                logits[task], targets[:, task_idx], reduction="sum",
            )
            probabilities = logits[task].softmax(-1).cpu().numpy()
            preds = probabilities.argmax(-1)
            for i, image_path in enumerate(ids):
                records.append({
                    "run_id": run_id, "seed": seed, "split": split,
                    "image_path": image_path, "task": task,
                    "truth": int(labels[i, task_idx]), "prediction_raw": int(preds[i]),
                    "probability_positive": float(probabilities[i, 1]),
                })
    scores, predictions = evaluate_records(pd.DataFrame(records))
    means = (losses / (len(records) // len(TASKS))).cpu().tolist()
    scores["loss"] = dict(zip(TASKS, means))
    scores["loss"]["mean"] = sum(means) / len(TASKS)
    return scores, predictions


def rescue_hurt(reference: pd.DataFrame, candidate: pd.DataFrame):
    """Compare per ID and per task. Never infer rescue/hurt from macro-F1 difference."""
    cols = ["image_path", "task", "truth", "prediction_raw"]
    for df in (reference, candidate):
        if df.duplicated(["image_path", "task"]).any():
            raise ValueError("Duplicate pair in rescue/hurt")
    ref = reference[cols].rename(columns={"prediction_raw": "pred_ref"})
    new = candidate[cols].rename(columns={"prediction_raw": "pred_new"})
    joined = ref.merge(new, on=["image_path", "task"], suffixes=("_ref", "_new"), validate="one_to_one")
    if len(joined) != len(ref) or len(joined) != len(new) or not (joined.truth_ref == joined.truth_new).all():
        raise ValueError("Missing/mismatched IDs or labels")
    results = {}
    for task in TASKS:
        part = joined[joined.task == task]
        truth = part.truth_ref
        results[task] = {
            "rescue": int(((part.pred_ref != truth) & (part.pred_new == truth)).sum()),
            "hurt": int(((part.pred_ref == truth) & (part.pred_new != truth)).sum()),
        }
    return results


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    tmp.replace(path)
