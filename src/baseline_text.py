"""Baseline tambahan INDOMEME: validation saja, tanpa membaca label test."""
import csv
import hashlib
import importlib.metadata
import io
import json
import platform
import sys
import time
import urllib.request
import warnings
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold
from threadpoolctl import threadpool_limits

SEED = 42
DATA_URL = 'https://dfzljdn9uc3pi.cloudfront.net/2026/cs-3736/1/INDOMEME_Dataset.zip'
ZIP_SHA = 'b7131f28b96283d5beabcddf8e800bf19e48e18b9aae500e3cc6ee8cef635092'
VARIANTS = {
    'OCR': ['ocr'], 'Caption': ['caption'], 'Caption+OCR': ['caption', 'ocr'],
    'Caption+Topic': ['topic', 'caption'], 'OCR+Topic': ['topic', 'ocr'],
    'Caption+OCR+Topic': ['topic', 'caption', 'ocr'],
}
# Verified exact-image groups from the complete pinned-image audit, 18 October 2026.
# shortcut: these groups do not detect crops/recompression; extend with perceptual audit later.
TRAIN_DUPLICATES = [
    ['3910338112587181.jpeg', '3910336549254004.jpeg'],
    ['656359160541478.jpeg', '656357510541643.jpeg'],
    ['1384560702680330.jpeg', '2076707349531098.jpeg'],
]


def load_training(archive):
    archive = Path(archive)
    if not archive.exists():
        archive.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(DATA_URL, headers={'User-Agent': 'INDOMEME-reproduction/1.0'})
        with urllib.request.urlopen(request, timeout=120) as response:
            payload = response.read()
        assert hashlib.sha256(payload).hexdigest() == ZIP_SHA, 'Checksum sumber berubah; hentikan audit.'
        archive.write_bytes(payload)
    payload = archive.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == ZIP_SHA
    with zipfile.ZipFile(io.BytesIO(payload)) as zipped:
        candidates = [name for name in zipped.namelist()
                      if name.endswith('train_translated.csv') and not name.startswith('__MACOSX/')]
        assert len(candidates) == 1
        rows = list(csv.DictReader(io.StringIO(zipped.read(candidates[0]).decode('utf-8-sig'))))
    assert len(rows) == 4018
    assert len({row['image_path'] for row in rows}) == len(rows)
    assert Counter(row['hate_final'] for row in rows) == {'hate': 1386, 'not hate': 2632}
    assert Counter(row['appropriate_final'] for row in rows) == {'inappropriate': 1731, 'appropriate': 2287}
    for row in rows:
        for field in ['ocr', 'caption']:
            assert row[field].strip() and row[field].lower() != 'nan'
        if row['topic'].strip().lower() in {'', 'nan'}:
            row['topic'] = ''
    return rows


def partition(rows):
    names = {row['image_path'] for row in rows}
    assert all(set(pair) <= names for pair in TRAIN_DUPLICATES)
    duplicate_map = {name: pair[0] for pair in TRAIN_DUPLICATES for name in pair}
    groups = np.array([duplicate_map.get(row['image_path'], row['image_path']) for row in rows])
    joint = [row['hate_final'] + '|' + row['appropriate_final'] for row in rows]
    fit, val = next(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
                    .split(np.zeros(len(rows)), joint, groups))
    assert not set(groups[fit]) & set(groups[val]), 'Duplikat lintas train/validation.'
    assert len(fit) + len(val) == len(rows)
    assert set(joint[i] for i in fit) == set(joint[i] for i in val) == set(joint)
    return fit, val, groups


def text_input(row, fields):
    return ' [SEP] '.join(row[field] for field in fields if row[field].strip())


def metrics(truth, pred):
    p, r, f, _ = precision_recall_fscore_support(truth, pred, labels=[0, 1], average='macro', zero_division=0)
    return {'macro_precision': float(p), 'macro_recall': float(r), 'macro_f1': float(f),
            'accuracy': float(accuracy_score(truth, pred)),
            'confusion_matrix_labels_0_1': confusion_matrix(truth, pred, labels=[0, 1]).tolist()}


def run_baselines(archive='data/INDOMEME_Dataset.zip', output_root='runs'):
    rows = load_training(archive)
    fit, val, groups = partition(rows)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = Path(output_root) / ('baseline_text_' + run_id)
    output.mkdir(parents=True, exist_ok=False)
    with (output / 'split_manifest.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['image_path', 'partition', 'exact_image_group'])
        val_set = set(val.tolist())
        writer.writerows((row['image_path'], 'validation' if i in val_set else 'train', groups[i])
                         for i, row in enumerate(rows))
    config = {
        'purpose': 'Additional sanity baseline, not reproduction of a paper model',
        'evaluation': 'validation only; official 1005-row test never evaluated',
        'seed': SEED, 'split': 'First fold StratifiedGroupKFold(5, shuffle=True, seed=42); joint labels',
        'fit_rows': len(fit), 'validation_rows': len(val), 'data_url': DATA_URL, 'zip_sha256': ZIP_SHA,
        'image_audit_revision': '3e0cb8cae9cd726582f69afdb122163d0eab108b',
        'train_exact_duplicate_pairs': TRAIN_DUPLICATES,
        'variants': VARIANTS, 'separator': 'literal [SEP] between components; not transformer tokenization',
        'tfidf': {'analyzer': 'word', 'ngram_range': [1, 2], 'min_df': 2, 'max_features': 30000,
                  'sublinear_tf': True, 'lowercase': True},
        'logreg': {'C': 1.0, 'class_weight': 'balanced', 'solver': 'liblinear', 'max_iter': 1000},
        'selection': 'No hyperparameter search; 6 prespecified variants. Report all validation results.',
        'topic_note': 'Provided annotated topic is privileged input; not automatically available at deployment.',
        'label_map': {'hate': {'not hate': 0, 'hate': 1},
                      'appropriate': {'appropriate': 0, 'inappropriate': 1}},
        'versions': {pkg: importlib.metadata.version(pkg) for pkg in
                     ['numpy', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl']},
        'python': platform.python_version(), 'platform': platform.platform(), 'created_utc': run_id,
        'split_sha256': hashlib.sha256((output / 'split_manifest.csv').read_bytes()).hexdigest(),
        'class_distribution': {part: dict(Counter(rows[i]['hate_final'] + '|' + rows[i]['appropriate_final']
                                                for i in indices)) for part, indices in [('train', fit), ('validation', val)]},
    }
    if '__file__' in globals():
        config['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    results, predictions = [], []
    start = time.perf_counter()
    with warnings.catch_warnings(), threadpool_limits(limits=2):
        warnings.simplefilter('error', ConvergenceWarning)
        for task, column, positive in [('hate', 'hate_final', 'hate'),
                                       ('appropriate', 'appropriate_final', 'inappropriate')]:
            y = np.array([int(row[column] == positive) for row in rows])
            for variant in ['Majority'] + list(VARIANTS):
                begin = time.perf_counter()
                if variant == 'Majority':
                    model = DummyClassifier(strategy='most_frequent')
                    x_fit, x_val = np.zeros((len(fit), 1)), np.zeros((len(val), 1))
                else:
                    text = [text_input(row, VARIANTS[variant]) for row in rows]
                    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000,
                                                 sublinear_tf=True, dtype=np.float64)
                    x_fit = vectorizer.fit_transform([text[i] for i in fit])
                    x_val = vectorizer.transform([text[i] for i in val])
                    model = LogisticRegression(C=1.0, class_weight='balanced', solver='liblinear',
                                               max_iter=1000, random_state=SEED)
                model.fit(x_fit, y[fit])
                pred, prob = model.predict(x_val), model.predict_proba(x_val)[:, 1]
                assert np.isfinite(prob).all() and len(pred) == len(val)
                row_result = {'task': task, 'variant': variant, 'evaluation': 'validation',
                              **metrics(y[val], pred), 'seconds': time.perf_counter() - begin}
                results.append(row_result)
                predictions.extend({'task': task, 'variant': variant, 'image_path': rows[i]['image_path'],
                                    'truth': int(y[i]), 'prediction': int(p), 'probability_positive': float(q)}
                                   for i, p, q in zip(val, pred, prob))
                print(f"{task:11s} {variant:22s} macro-F1={row_result['macro_f1']:.4f}", flush=True)
    assert len(results) == 14 and len(predictions) == 14 * len(val)
    (output / 'metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    for filename, records in [('metrics.csv', [{k: v for k, v in r.items() if not isinstance(v, list)} for r in results]),
                              ('predictions.csv', predictions)]:
        with (output / filename).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=records[0].keys())
            writer.writeheader()
            writer.writerows(records)
    (output / 'completion.json').write_text(json.dumps({'success': True, 'conditions': len(results),
        'seconds': time.perf_counter() - start, 'test_evaluated': False}, indent=2), encoding='utf-8')
    print('Bukti hasil:', output.resolve())
    return output, results


def self_check():
    assert metrics([0, 0, 1, 1], [0, 0, 0, 0])['macro_f1'] == 1 / 3
    assert metrics([0, 1], [0, 1])['macro_f1'] == 1.0
    assert text_input({'topic': '', 'caption': 'a', 'ocr': 'b'}, ['topic', 'caption', 'ocr']) == 'a [SEP] b'
    assert list(VARIANTS.values())[-1] == ['topic', 'caption', 'ocr']


if __name__ == '__main__':
    self_check()
    archive = sys.argv[1] if len(sys.argv) > 1 else 'data/INDOMEME_Dataset.zip'
    run_baselines(archive)
