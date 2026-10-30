"""Inspect a frozen core configuration; training requires --run and a new workspace."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
IDS = ['E005', 'E006', 'E041', 'E042', 'E059', 'E060', 'E101', 'E110']


def frozen_config(identifier):
    config = json.loads((ROOT / 'results/runs' / identifier / 'config.json').read_text())
    source = ROOT / 'src'
    checks = {'master_training.py': config['script_sha256'], **config['support_sha256']}
    for name, expected in checks.items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == expected, name
    assert config['scope'] == 'full' and config['seed'] == 42
    assert config['max_train'] is None and config['max_validation'] is None
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--id', choices=IDS, required=True)
    parser.add_argument('--workspace', type=Path)
    parser.add_argument('--run', action='store_true', help='Download pinned inputs and train one condition.')
    args = parser.parse_args()
    config = frozen_config(args.id)
    print(json.dumps(config, indent=2))
    if not args.run:
        return
    if args.workspace is None or args.workspace.exists() or not args.workspace.parent.is_dir():
        parser.error('--run requires a new --workspace directory; existing results are never reused or overwritten.')
    import torch
    import master_workflow as workflow
    import master_training as training
    import local_storage
    assert torch.cuda.is_available(), 'A calibrated CUDA GPU is required.'
    local_storage.CACHE.mkdir(parents=True, exist_ok=True)
    budget = local_storage.check(config)
    assert budget['cache_free'] >= budget['cache_reserve'] + budget['estimated_download_peak']
    assert shutil.disk_usage(args.workspace.parent).free >= budget['project_reserve'] + budget['estimated_checkpoint_peak']
    torch.set_num_threads(2)
    root = workflow.workspace(args.workspace.resolve())
    frame = workflow.load_metadata(root)
    workflow.data_checks(frame)
    fit, validation, test = workflow.make_split(frame, root)
    split = root / 'splits/train_validation.csv'
    normalized = split.read_bytes().replace(b'\r\n', b'\n')
    choices = [normalized, normalized.replace(b'\n', b'\r\n')]
    matching = [content for content in choices if hashlib.sha256(content).hexdigest() == config['split_sha256']]
    assert matching, 'Released data or frozen split no longer matches.'
    split.write_bytes(matching[0])
    assert (len(fit), len(validation), len(test)) == (3214, 804, 1005)
    images = workflow.load_images(root, ('train',)) if config['image'] else {}
    _, run = training.run_training(root, config, fit, validation, images)
    training.lock_test_protocol(root, [run.name])
    if config['image']:
        images.update(workflow.load_images(root, ('test',)))
    scores = training.evaluate_official_test(root, run.name, test, images)
    print(json.dumps({'run_id': run.name, 'test_metrics': scores}, indent=2))


if __name__ == '__main__':
    main()
