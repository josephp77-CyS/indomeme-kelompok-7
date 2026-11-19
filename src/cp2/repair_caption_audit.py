"""Measure how generated descriptions differ by OCR overlap and existing INDOMEME topics."""

import argparse
import re

import numpy as np

from .caption_triage import SEEDS, load_views
from .nlp_data import get_partitions
from .nlp_evaluation import write_json
from .operations import private_path
from .recaption import restore_caption_rows
from .repair_router import error_cost


def words(text):
    return set(re.findall(r"[^\W_]+", text.casefold()))


def caption_audit(workspace, study, job):
    partitions, digest = get_partitions(workspace)
    result = {}
    for split, folder in (("validation", "benchmark"), ("test", "official")):
        frame = partitions[split].set_index("image_path").sort_index()
        artifact = job / f"{split}_captions_PRIVATE.json"
        captions = restore_caption_rows(artifact, frame.index, split, digest)
        available = np.array([bool(captions[image_id]) for image_id in frame.index])
        clean_overlap, generated_overlap, clean_length, generated_length = [], [], [], []
        for image_id, row in frame.iterrows():
            generated = words(captions[image_id])
            if not generated:
                continue
            clean, ocr = words(row.caption), words(row.ocr)
            clean_length.append(len(clean))
            generated_length.append(len(generated))
            if clean and ocr:
                clean_overlap.append(len(clean & ocr) / len(clean))
                generated_overlap.append(len(generated & ocr) / len(generated))

        lexical = {
            "paired_lexical_n": len(clean_length), "paired_nonempty_ocr_n": len(clean_overlap),
            "median_distinct_words_original": float(np.median(clean_length)),
            "median_distinct_words_generated": float(np.median(generated_length)),
            "mean_share_of_original_caption_words_also_in_ocr": float(np.mean(clean_overlap)),
            "mean_share_of_generated_caption_words_also_in_ocr": float(np.mean(generated_overlap)),
        }
        tag_sets = frame.topic.map(lambda tag: {item.strip() for item in tag.split(",") if item.strip()})
        groups = {"overall": np.ones(len(frame), dtype=bool), "no_topic": tag_sets.map(lambda tags: not tags).to_numpy()}
        for topic in sorted(set().union(*tag_sets)):
            groups[topic] = tag_sets.map(lambda tags: topic in tags).to_numpy()
        changes = {"normal": [], "caption_swap": []}
        for seed in SEEDS:
            run_id = f"S1_full_seed{seed}"
            ids, truth, views = load_views(study / folder / run_id, split, len(frame), seed, run_id)
            new_ids, new_truth, new = load_views(job / "router_" / split / run_id, split, len(frame), seed,
                                                 run_id, ("recaption",))
            if not ids.equals(frame.index) or not ids.equals(new_ids) or not np.array_equal(truth, new_truth):
                raise ValueError("Caption audit requires paired predictions and topic metadata")
            after = error_cost(truth, new["recaption"].predictions)
            for mode in changes:
                before = error_cost(truth, views[mode].predictions)
                changes[mode].append(np.where(available, after - before, 0))
        result[split] = {
            "lexical": lexical, "coverage": int(available.sum()),
            "groups": {
                name: {"n": int(mask.sum()), "n_usable_generated_captions": int((mask & available).sum()),
                       **{f"cost_delta_per_100_vs_{mode}": 100 * float(np.asarray(rows)[:, mask].mean())
                          for mode, rows in changes.items()}}
                for name, mask in groups.items() if mask.sum() >= 50
            },
        }
    return {
        "scope": "Descriptive S1 audit; apply available generated captions to all items; shared meme tags overlap.",
        "manifest_sha256": digest, "seeds": SEEDS, "splits": result,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--study", required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    workspace, study, job = map(private_path, (args.workspace, args.study, args.job))
    write_json(job / "caption_audit.json", caption_audit(workspace, study, job))


if __name__ == "__main__":
    main()
