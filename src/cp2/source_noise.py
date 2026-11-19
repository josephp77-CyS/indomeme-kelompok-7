"""Deterministic, label-blind changes to caption/OCR on paired INDOMEME records."""

import hashlib
import random

import numpy as np

from .nlp_data import clean_field


CONDITIONS = ("normal", "no_caption", "caption_swap", "no_ocr", "ocr_noise")
TRAIN_WEIGHTS = (0.60, 0.10, 0.10, 0.10, 0.10)
CONFUSABLES = dict(zip("oislbzeg", "01518239"))


def rng_for(seed, image_id, operation):
    key = f"{seed}/{operation}/{image_id}".encode()
    return random.Random(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))


def damage_ocr(text, rng):
    eligible = [i for i, char in enumerate(text) if char.lower() in CONFUSABLES]
    if not eligible:
        eligible = [i for i, char in enumerate(text) if char.isalnum()]
    if not eligible:
        return text
    chars = list(text)
    for pos in rng.sample(eligible, max(1, round(len(eligible) * 0.20))):
        chars[pos] = CONFUSABLES.get(chars[pos].lower(), "_")
    return "".join(chars)


def mixed_condition(seed, image_id):
    return rng_for(seed, image_id, "train").choices(CONDITIONS, weights=TRAIN_WEIGHTS, k=1)[0]


def noisy_sources(frame, condition, seed):
    """Return an aligned input frame and text-quality targets (-1 missing, 0 altered, 1 original)."""
    if condition not in (*CONDITIONS, "mixed"):
        raise ValueError(f"Unknown condition: {condition}")
    if frame.empty or not frame.image_path.is_unique:
        raise ValueError("Noise requires nonempty, unique meme IDs")
    data = frame.reset_index(drop=True).copy()
    ids = data.image_path.astype(str).tolist()
    captions, ocr = ([clean_field(value) for value in data[column]] for column in ("caption", "ocr"))
    donors = sorted((i for i, text in enumerate(captions) if text), key=lambda i: (len(captions[i]), ids[i]))
    ranks = {i: rank for rank, i in enumerate(donors)}
    modified_caption, modified_ocr = captions.copy(), ocr.copy()
    for i, image_id in enumerate(ids):
        mode = mixed_condition(seed, image_id) if condition == "mixed" else condition
        if mode == "no_caption" and ocr[i]:
            modified_caption[i] = ""
        elif mode == "caption_swap" and captions[i]:
            rank = ranks[i]
            others = [j for j in donors[max(0, rank - 12):rank + 13] if captions[j] != captions[i]]
            if not others:
                others = [j for j in donors if captions[j] != captions[i]]
            if not others:
                raise ValueError("Caption swap needs a different donor caption")
            modified_caption[i] = captions[rng_for(seed, image_id, "caption").choice(others)]
        elif mode == "no_ocr" and captions[i]:
            modified_ocr[i] = ""
        elif mode == "ocr_noise":
            modified_ocr[i] = damage_ocr(ocr[i], rng_for(seed, image_id, "ocr"))
    data["caption"], data["ocr"] = modified_caption, modified_ocr
    quality = np.array([
        [int(new_caption == caption) if new_caption else -1, int(new_ocr == text) if new_ocr else -1]
        for caption, text, new_caption, new_ocr in zip(captions, ocr, modified_caption, modified_ocr)
    ], dtype=np.int8)
    return data, quality


def repair_training_captions(frame, quality, captions, seed):
    for index, image_id in enumerate(frame.image_path.astype(str)):
        caption = captions[image_id]
        if caption and mixed_condition(seed, image_id) == "normal" and rng_for(seed, image_id, "recaption").random() < 0.5:
            frame.at[index, "caption"] = caption
            quality[index, 0] = 0
