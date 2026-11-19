"""Offline human-review queue simulation from aligned predictions."""

import hashlib
import math

import numpy as np

from .nlp_evaluation import TASKS


def simulate_review(predictions, meme_ids, source_doubt):
    """Measure which gold positives and raw model errors a fixed review budget reaches."""
    ids = list(meme_ids)
    if len(set(ids)) != len(ids) or len(source_doubt) != len(ids):
        raise ValueError("Unpaired queue entries")

    def paired(column):
        table = predictions.pivot(index="image_path", columns="task", values=column)
        if set(table.index) != set(ids) or set(table.columns) != set(TASKS) or table.isna().any().any():
            raise ValueError("Incomplete prediction pairs in review queue")
        return table.loc[ids, list(TASKS)].to_numpy()

    probabilities = paired("probability_positive")
    gold = paired("truth").astype(bool)
    errors = paired("prediction_raw") != gold
    doubt = np.asarray(source_doubt)
    if doubt.shape != (len(ids),) or not np.isfinite(doubt).all() or not ((doubt >= 0) & (doubt <= 1)).all():
        raise ValueError("Invalid source-doubt scores")
    risk = probabilities.max(axis=1)
    uncertainty = (1 - np.abs(2 * probabilities - 1)).max(axis=1)
    policies = {
        "random_order_proxy": sorted(range(len(ids)), key=lambda i: hashlib.sha256(ids[i].encode()).digest()),
        "positive_score": np.argsort(-risk, kind="stable"),
        "score_and_uncertainty": np.argsort(-(0.75 * risk + 0.25 * uncertainty), kind="stable"),
        "score_uncertainty_and_source_doubt": np.argsort(
            -(0.50 * risk + 0.25 * uncertainty + 0.25 * doubt), kind="stable"),
    }
    result = {}
    for policy, order in policies.items():
        result[policy] = {}
        for fraction in (0.10, 0.20, 0.40):
            budget = math.ceil(len(ids) * fraction)
            selected = order[:budget]
            result[policy][str(fraction)] = {
                "reviewed": budget,
                "gold_positive_recall": dict(zip(TASKS, (gold[selected].sum(0) / np.maximum(gold.sum(0), 1)).tolist())),
                "model_error_recall": dict(zip(TASKS, (errors[selected].sum(0) / np.maximum(errors.sum(0), 1)).tolist())),
                "reviewed_gold_negative_both_tasks": int((~gold[selected].any(axis=1)).sum()),
            }
    return result
