"""Audit portable research evidence and deliverable structure without training or network access."""
import csv
import importlib.util
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile
import hashlib
import re

ROOT = Path(__file__).resolve().parent
IDS = {'E005', 'E006', 'E041', 'E042', 'E059', 'E060', 'E101', 'E110'}


def scores(cm):
    assert len(cm) == 2 and all(len(row) == 2 for row in cm)
    assert all(isinstance(value, int) and value >= 0 for row in cm for value in row)
    precision, recall, f1 = [], [], []
    for label in [0, 1]:
        tp = cm[label][label]
        predicted = sum(row[label] for row in cm)
        support = sum(cm[label])
        p, r = tp / predicted if predicted else 0., tp / support if support else 0.
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / (p + r) if p + r else 0.)
    return dict(macro_precision=sum(precision) / 2, macro_recall=sum(recall) / 2,
                macro_f1=sum(f1) / 2, accuracy=(cm[0][0] + cm[1][1]) / sum(map(sum, cm)))


def main():
    assert scores([[2, 0], [0, 3]])['macro_f1'] == 1.
    assert scores([[0, 2], [3, 0]])['macro_f1'] == 0.
    audit = json.loads((ROOT / 'results/audit/local_results_audit.json').read_text())
    assert audit['passed'] and audit['configurations'] == 8 and audit['outcomes'] == 10
    rows = list(csv.DictReader((ROOT / 'results/hasil_utama.csv').open(encoding='utf-8')))
    assert len(rows) == 10 and {row['id'] for row in rows} == IDS
    records = {row['id']: row for row in audit['results']}
    pred_checks = 0
    for row in rows:
        identifier, task = row['id'], row['task']
        key = 'appropriate_constrained' if task == 'appropriate' and row['policy'] == 'constrained' else task
        cm = records[identifier]['test'][key]['confusion_matrix_labels_0_1']
        assert sum(map(sum, cm)) == 1005
        for metric, value in scores(cm).items():
            assert abs(float(row[metric]) - value) < 1e-12, (identifier, task, metric)
        path = ROOT / 'results/runs' / identifier / 'test_predictions.csv'
        if path.exists():
            data = [item for item in csv.DictReader(path.open(encoding='utf-8-sig')) if item['task'] == task]
            assert len(data) == 1005 and len({item['image_path'] for item in data}) == 1005
            actual = [[0, 0], [0, 0]]
            for item in data:
                pred = item['prediction_constrained'] if row['policy'] == 'constrained' else item['prediction']
                truth, pred = int(item['truth']), int(pred)
                assert truth in [0, 1] and pred in [0, 1]
                actual[truth][pred] += 1
            assert actual == cm, (identifier, task, 'predictions')
            pred_checks += 1
    spec = importlib.util.spec_from_file_location('reproduce_core', ROOT / 'src/reproduce_core.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert len([module.frozen_config(identifier) for identifier in IDS]) == 8
    notebook = json.loads((ROOT / 'notebooks/eksperimen_indomeme.ipynb').read_text(encoding='utf-8'))
    code = [cell for cell in notebook['cells'] if cell['cell_type'] == 'code']
    assert notebook['nbformat'] == 4 and code
    assert all(cell.get('execution_count') is not None for cell in code), 'Notebook has unexecuted code cells'
    assert not any(out.get('output_type') == 'error' for cell in code for out in cell.get('outputs', []))
    proof=json.loads((ROOT/'results/audit/provenance/migration_notebook.json').read_text())
    assert hashlib.sha256('\n'.join(''.join(cell['source']) for cell in code).encode()).hexdigest()==proof['code_cells_sha256']
    assert hashlib.sha256((ROOT/'notebooks/eksperimen_indomeme.ipynb').read_bytes()).hexdigest()==proof['submission_notebook_sha256']
    baselines=list(csv.DictReader((ROOT/'results/hasil_baseline.csv').open(encoding='utf-8')))
    assert len(baselines)==14
    for md in [ROOT/'README.md',*ROOT.glob('docs/*.md')]:
        for link in re.findall(r'\]\(([^)]+)\)',md.read_text(encoding='utf-8')):
            if not link.startswith(('http://','https://','#')):
                assert (md.parent/link.split('#')[0]).exists(),(str(md),link)
    pptx = ROOT / 'docs/INDOMEME_Checkpoint1_PitchDeck.pptx'
    assert pptx.stat().st_size < 100 * 2**20
    assert hashlib.sha256(pptx.read_bytes()).hexdigest()=='ef16029ae6db09094ddbe278effa8c4327afb0b800c5941abc0a787b54bfae91'
    with zipfile.ZipFile(pptx) as package:
        assert package.testzip() is None
        names = package.namelist()
        slides = [n for n in names if n.startswith('ppt/slides/slide') and n.endswith('.xml') and '/_rels/' not in n]
        assert len(slides) == 25
        assert not any('vbaProject' in n or n.startswith('ppt/embeddings/') and n.endswith('.bin') for n in names)
        for name in slides:
            ET.fromstring(package.read(name))
    print(f'PASS: 8 configs, 10 outcomes, {pred_checks} prediction audits, {len(code)} executed notebook cells, {len(slides)} PPTX slides.')
    if pred_checks == 0:
        print('Prediction CSVs omitted in GitHub checkout; aggregate confusion matrices were checked. Full audit ZIP is on Drive.')


if __name__ == '__main__':
    main()
