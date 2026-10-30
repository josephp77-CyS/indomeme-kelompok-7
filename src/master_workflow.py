"""Visible data-science steps used by the INDOMEME master notebook."""
import hashlib
import io
import json
import re
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

REVISION = '3e0cb8cae9cd726582f69afdb122163d0eab108b'
CSV_URL = 'https://dfzljdn9uc3pi.cloudfront.net/2026/cs-3736/1/INDOMEME_Dataset.zip'
CSV_SHA = 'b7131f28b96283d5beabcddf8e800bf19e48e18b9aae500e3cc6ee8cef635092'
PARQUET = {
    'train': ('train-00000-of-00001.parquet', 'dea87a3de2a8bcd4c7179ab5cca14ff320b7c32f028aefdf2635457846d59c4b'),
    'test': ('test-00000-of-00001.parquet', 'ab06c80b83e0128a686f9134da4e7e44611635e8b1af74671f5595a4907ee2e3'),
}
TEXT_MODELS = {'IndoBERT': 'indobenchmark/indobert-base-p1', 'XLM-R': 'FacebookAI/xlm-roberta-base',
               'IndoBERTweet': 'indolem/indobertweet-base-uncased'}
IMAGE_MODELS = {'SwinV2': 'microsoft/swinv2-tiny-patch4-window8-256',
                'CLIP': 'openai/clip-vit-base-patch32', 'ViT': 'google/vit-base-patch32-224-in21k'}
VARIANTS = {'OCR': ['ocr'], 'Caption': ['caption'], 'Caption+OCR': ['caption', 'ocr'],
            'Caption+Topic': ['topic', 'caption'], 'OCR+Topic': ['topic', 'ocr'],
            'Caption+OCR+Topic': ['topic', 'caption', 'ocr']}
MLLMS = ['GPT-4o', 'Gemini 2.5 Flash', 'Gemma3 27B', 'Qwen2-VL 2B', 'LLaVA OneVision 0.5B']
# Transcribed from the Indonesian templates in the cached primary article, RQ3.
RQ3_PROMPTS = {
    'hate': 'Anda adalah pendeteksi meme yang mengandung ujaran kebencian. Klasifikasikan sebagai “hate” jika meme berikut memuat penghinaan, pelecehan, stereotip negatif, atau sindiran yang merendahkan terhadap seseorang atau kelompok, baik secara langsung maupun tersirat. Klasifikasikan sebagai “not hate” jika tidak mengandung unsur-unsur tersebut. Berikan labelnya saja.',
    'appropriate': 'Anda adalah pendeteksi meme yang mengandung konten tidak pantas. Klasifikasikan sebagai “inappropriate” jika meme berikut memuat penghinaan, pelecehan, stereotip negatif, atau sindiran yang merendahkan terhadap seseorang atau kelompok, serta mengandung kata-kata tidak pantas, konteks berbau seksual, pembunuhan, atau hal-hal lain yang tidak sesuai untuk dilihat oleh publik umum baik secara teks maupun visual. Klasifikasikan sebagai “appropriate” jika tidak mengandung unsur-unsur tersebut. Berikan labelnya saja.',
}


def digest(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def workspace(root):
    root = Path(root)
    for name in ['data/raw', 'data/processed', 'eda', 'splits', 'runs', 'reports', 'scripts']:
        (root / name).mkdir(parents=True, exist_ok=True)
    return root


def download(url, path, expected_sha):
    path = Path(path)
    if path.exists():
        assert digest(path) == expected_sha, f'Cache checksum mismatch: {path}'
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print('Mengunduh sumber:', path.name, flush=True)
    temporary = path.with_suffix(path.suffix + '.part')
    request = urllib.request.Request(url, headers={'User-Agent': 'INDOMEME-reproduction/1.0'})
    with urllib.request.urlopen(request, timeout=180) as response, temporary.open('wb') as handle:
        received = 0
        while chunk := response.read(1024 * 1024):
            handle.write(chunk)
            previous = received // (100 * 1024 * 1024)
            received += len(chunk)
            if received // (100 * 1024 * 1024) > previous:
                print(path.name, ':', received // (1024 * 1024), 'MiB diterima', flush=True)
    assert digest(temporary) == expected_sha, 'Checksum unduhan berubah; hentikan proses.'
    temporary.replace(path)
    return path


def load_metadata(root):
    root = workspace(root)
    archive = download(CSV_URL, root / 'data/raw/INDOMEME_Dataset.zip', CSV_SHA)
    frames = []
    with zipfile.ZipFile(archive) as zipped:
        for split in ['train', 'test']:
            names = [n for n in zipped.namelist() if n.endswith(f'{split}_translated.csv') and not n.startswith('__MACOSX/')]
            assert len(names) == 1
            frame = pd.read_csv(io.BytesIO(zipped.read(names[0])), keep_default_na=False)
            assert len(frame) == {'train': 4018, 'test': 1005}[split]
            frame['split'] = split
            frame['topic'] = frame['topic'].replace('nan', '')
            frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    assert df['image_path'].is_unique
    assert set(df['hate_final']) == {'hate', 'not hate'}
    assert set(df['appropriate_final']) == {'appropriate', 'inappropriate'}
    assert not ((df.hate_final == 'hate') & (df.appropriate_final == 'appropriate')).any()
    df['joint_label'] = df.hate_final + ' | ' + df.appropriate_final
    df['y_hate'] = (df.hate_final == 'hate').astype(int)
    df['y_appropriate'] = (df.appropriate_final == 'inappropriate').astype(int)
    for col in ['ocr', 'caption']:
        assert df[col].str.strip().ne('').all()
        df[col + '_words'] = df[col].str.split().str.len()
    df.to_csv(root / 'data/processed/metadata.csv', index=False, lineterminator='\n')
    return df


def data_checks(df):
    observed = df.groupby(['split', 'hate_final', 'appropriate_final']).size()
    expected = {('train', 'hate', 'inappropriate'): 1386, ('train', 'not hate', 'inappropriate'): 345,
                ('train', 'not hate', 'appropriate'): 2287, ('test', 'hate', 'inappropriate'): 347,
                ('test', 'not hate', 'inappropriate'): 86, ('test', 'not hate', 'appropriate'): 572}
    assert observed.to_dict() == expected
    topic_rows = []
    for row in df.itertuples():
        topics = [part.strip() for part in row.topic.split(',')] if row.topic.strip() else ['none/others']
        topic_rows.extend({'topic': topic, 'joint_label': row.joint_label} for topic in topics)
    topic_table = pd.crosstab(pd.Series([r['joint_label'] for r in topic_rows]),
                              pd.Series([r['topic'] for r in topic_rows]))
    order = ['gender', 'individual', 'national origin', 'political', 'religion', 'institutions', 'social sub-groups', 'none/others']
    expected_topics = {'hate | inappropriate': [324,465,270,543,57,273,583,2],
                       'not hate | inappropriate': [210,22,11,15,6,12,191,33],
                       'not hate | appropriate': [319,178,65,110,178,147,2075,193]}
    for label, values in expected_topics.items():
        assert topic_table.loc[label, order].tolist() == values
    return pd.crosstab(df['split'], df['joint_label']), topic_table[order]


def plot_eda(df, root):
    import matplotlib.pyplot as plt
    counts, topics = data_checks(df)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    counts.plot.bar(ax=axes[0], rot=0, title='Label gabungan: split resmi')
    axes[0].set_ylabel('Jumlah meme')
    topics.sum().sort_values().plot.barh(ax=axes[1], title='Topik multilabel: seluruh data')
    axes[1].set_ylabel('')
    train = df[df.split == 'train']
    bins = np.linspace(0, max(train.ocr_words.max(), train.caption_words.max()) + 1, 41)
    for col in ['ocr_words', 'caption_words']:
        axes[2].hist(train[col], bins=bins, alpha=.55, label=col)
    axes[2].legend(); axes[2].set_title('Panjang teks TRAIN (kata, bukan token)')
    fig.tight_layout(); fig.savefig(Path(root) / 'eda/distribusi.png', dpi=150)
    plt.show()
    summary = {'rows': len(df), 'train': int((df.split == 'train').sum()), 'test': int((df.split == 'test').sum()),
               'hate_fraction_train': float(train.y_hate.mean()), 'inappropriate_fraction_train': float(train.y_appropriate.mean()),
               'blank_topic': int(df.topic.str.strip().eq('').sum()), 'tables_3_4_5_match': True,
               'topic_is_annotated': True}
    save_json(Path(root) / 'eda/summary.json', summary)
    return summary


def load_images(root, splits=('train',)):
    import pyarrow.parquet as pq
    images = {}
    for split in splits:
        name, sha = PARQUET[split]
        path = download(f'https://huggingface.co/datasets/aiatums/INDOMEME/resolve/{REVISION}/data/{name}',
                        Path(root) / 'data/raw' / name, sha)
        for batch in pq.ParquetFile(path).iter_batches(batch_size=32, columns=['image_path', 'image']):
            for row in batch.to_pylist():
                assert row['image_path'] not in images
                images[row['image_path']] = row['image']['bytes']
    return images


def audit_images(df, images, root):
    from PIL import Image
    rows = []
    for record in df[['split', 'image_path']].itertuples(index=False):
        payload = images[record.image_path]
        with Image.open(io.BytesIO(payload)) as opened:
            opened.load()
            rgb = opened.convert('RGB')
            pixel_hash = hashlib.sha256(f'{rgb.width}x{rgb.height}:'.encode() + rgb.tobytes()).hexdigest()
            rows.append({'split': record.split, 'image_path': record.image_path, 'width': rgb.width,
                         'height': rgb.height, 'pixel_sha256': pixel_hash, 'image_sha256': hashlib.sha256(payload).hexdigest()})
    manifest = pd.DataFrame(rows)
    duplicates = manifest[manifest.duplicated('pixel_sha256', keep=False)].copy()
    cross_hashes = duplicates.groupby('pixel_sha256')['split'].nunique()
    summary = {'decoded': len(manifest), 'exact_duplicate_groups': int(duplicates.pixel_sha256.nunique()),
               'cross_split_groups': int((cross_hashes > 1).sum()), 'near_duplicates_tested': False}
    assert summary['decoded'] == 5023 and summary['exact_duplicate_groups'] == 5 and summary['cross_split_groups'] == 2
    manifest.to_csv(Path(root) / 'data/processed/image_manifest.csv', index=False, lineterminator='\n')
    duplicates.to_csv(Path(root) / 'eda/duplicates.csv', index=False)
    save_json(Path(root) / 'eda/image_audit.json', summary)
    return manifest, duplicates, summary


def show_examples(df, images):
    import matplotlib.pyplot as plt
    from PIL import Image
    examples = df[df.split == 'train'].groupby('joint_label', sort=True).head(1)
    fig, axes = plt.subplots(1, len(examples), figsize=(15, 5))
    for ax, row in zip(np.atleast_1d(axes), examples.itertuples()):
        with Image.open(io.BytesIO(images[row.image_path])) as image:
            ax.imshow(image.convert('RGB'))
        ax.set_title(row.joint_label + '\n' + row.image_path, fontsize=9); ax.axis('off')
    fig.tight_layout(); plt.show()
    return examples[['image_path', 'joint_label', 'topic', 'ocr', 'caption']]


def make_split(df, root):
    from baseline_text import partition
    train = df[df.split == 'train'].reset_index(drop=True)
    fit, val, groups = partition(train.to_dict('records'))
    manifest = train[['image_path', 'joint_label']].copy()
    manifest['partition'] = 'train'; manifest.loc[val, 'partition'] = 'validation'
    manifest['exact_image_group'] = groups
    path = Path(root) / 'splits/train_validation.csv'
    content = manifest.to_csv(index=False, lineterminator='\n')
    if path.exists():
        assert path.read_text(encoding='utf-8') == content, 'Manifest lama berbeda; pakai workspace baru untuk protokol baru.'
    else:
        path.write_text(content, encoding='utf-8')
    image_manifest = Path(root) / 'data/processed/image_manifest.csv'
    if image_manifest.exists():
        joined = manifest.merge(pd.read_csv(image_manifest)[['image_path', 'pixel_sha256']], on='image_path', validate='one_to_one')
        assert joined.groupby('pixel_sha256')['partition'].nunique().max() == 1
    save_json(Path(root) / 'splits/protocol.json', {'seed': 42, 'method': 'StratifiedGroupKFold 5, first fold',
              'fit': len(fit), 'validation': len(val), 'sha256': digest(path), 'test_used_for_selection': False})
    return train.iloc[fit].reset_index(drop=True), train.iloc[val].reset_index(drop=True), df[df.split == 'test'].reset_index(drop=True)


def experiment_id(rq, kind, image='', text='', variant='', task='both'):
    return '__'.join(str(x).replace(' ', '_') for x in [rq, kind, image or '-', text or '-', variant or '-', task])


def experiment_plan():
    rows = []
    def add(rq, kind, image='', text='', variant='', task='both', table=0):
        rows.append({'experiment_id': experiment_id(rq, kind, image, text, variant, task), 'rq': rq, 'kind': kind,
                     'image': image, 'text': text, 'variant': variant, 'task': task, 'paper_table': table})
    for task in ['hate', 'appropriate']:
        for image in IMAGE_MODELS: add('RQ2', 'image', image=image, task=task, table=7 if task=='hate' else 8)
        for text in TEXT_MODELS:
            for variant in VARIANTS: add('RQ2', 'text', text=text, variant=variant, task=task, table=9 if task=='hate' else 10)
        for image in IMAGE_MODELS:
            for text in TEXT_MODELS: add('RQ2', 'multimodal', image=image, text=text, variant='Caption+OCR+Topic', task=task, table=11 if task=='hate' else 12)
    for image in IMAGE_MODELS: add('RQ4', 'image', image=image, table=14)
    for text in TEXT_MODELS:
        for variant in VARIANTS: add('RQ4', 'text', text=text, variant=variant, table=15)
    for image in IMAGE_MODELS:
        for text in TEXT_MODELS: add('RQ4', 'multimodal', image=image, text=text, variant='Caption+OCR+Topic', table=16)
    for model in MLLMS:
        for variant in ['Image', 'Image+Topic']:
            for task in ['hate', 'appropriate']: add('RQ3', 'zero-shot', image=model, variant=variant, task=task, table=13)
    plan = pd.DataFrame(rows)
    assert len(plan) == 110 and plan.experiment_id.is_unique
    assert plan.groupby('rq').size().to_dict() == {'RQ2': 60, 'RQ3': 20, 'RQ4': 30}
    return plan


def paper_references(table_file):
    evidence = json.loads(Path(table_file).read_text(encoding='utf-8'))
    if 'records' in evidence:
        refs = pd.DataFrame(evidence['records'])
        assert len(refs)==140 and not refs.duplicated(['experiment_id','task']).any()
        assert set(refs.experiment_id)==set(experiment_plan().experiment_id)
        return refs
    tables = evidence['tables']
    aliases = {'SwinTransformerV2': 'SwinV2', 'XLM-RoBERTa': 'XLM-R', 'Image Caption': 'Caption'}
    rows = []
    for table in tables:
        number = int(table['id'].split('-')[1])
        if not 7 <= number <= 16: continue
        task = 'hate' if number in [7,9,11] else 'appropriate' if number in [8,10,12] else None
        last_model = ''
        for line in table['text'].splitlines():
            if line == 'Hate speech task': task='hate'; last_model=''; continue
            if line == 'Appropriateness task': task='appropriate'; last_model=''; continue
            fields = line.split('\t')
            if len(fields) < 5: continue
            try: values = list(map(float, fields[-4:]))
            except ValueError: continue
            assert all(0 <= x <= 1 for x in values)
            parts = fields[:-4]
            if parts[0]: last_model = parts[0]
            model = aliases.get(last_model, last_model)
            image=text=variant=''
            rq='RQ4' if number>=14 else 'RQ3' if number==13 else 'RQ2'
            if number in [7,8,14]: kind='image'; image=model
            elif number in [9,10,15]: kind='text'; text=model; variant=aliases.get(parts[1],parts[1])
            elif number in [11,12]:
                kind='multimodal'; text,image=model.split('+'); text=aliases.get(text,text); image=aliases.get(image,image); variant='Caption+OCR+Topic'
            elif number==16:
                kind='multimodal'; text=model; image=aliases.get(parts[1].lstrip('+'),parts[1].lstrip('+')); variant='Caption+OCR+Topic'
            else: kind='zero-shot'; image=model; variant=parts[1]
            task_key='both' if rq=='RQ4' else task
            rows.append({'experiment_id': experiment_id(rq,kind,image,text,variant,task_key), 'task':task,
                         'paper_macro_precision':values[0], 'paper_macro_recall':values[1],
                         'paper_macro_f1':values[2], 'paper_accuracy':values[3], 'paper_table':number})
    refs=pd.DataFrame(rows)
    assert len(refs)==140 and not refs.duplicated(['experiment_id','task']).any()
    assert set(refs.experiment_id)==set(experiment_plan().experiment_id)
    return refs


def progress(root):
    plan = experiment_plan(); rows=[]
    for file in Path(root).glob('runs/*/result.json'):
        item=json.loads(file.read_text(encoding='utf-8'))
        tested=(file.parent/'test_metrics.json').exists()
        rows.append({'experiment_id':item['config']['experiment_id'], 'run_id':file.parent.name,
                     'scope':item['config']['scope'], 'evaluation':'test' if tested else 'validation',
                     'status':'test selesai; review deviasi' if tested else 'validation selesai'})
    for file in Path(root).glob('reports/rq3/*/metrics.json'):
        rows.append({'experiment_id':file.parent.name,'run_id':file.parent.name,'scope':'zero-shot',
                     'evaluation':'test','status':'prediksi eksternal dievaluasi; audit backend'})
    runs=pd.DataFrame(rows)
    if len(runs):
        plan=plan.merge(runs.groupby('experiment_id').size().rename('recorded_runs'),left_on='experiment_id',right_index=True,how='left')
    else: plan['recorded_runs']=0
    plan['recorded_runs']=plan.recorded_runs.fillna(0).astype(int)
    plan.to_csv(Path(root)/'reports/experiment_progress.csv',index=False)
    return plan, runs
