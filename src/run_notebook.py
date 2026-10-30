"""Execute the complete analysis notebook in this isolated bundle and retain a truthful receipt."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import socket
import time
import argparse
import shutil
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    import nbformat
    from nbclient import NotebookClient
    from nbconvert import HTMLExporter
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-bundle',type=Path,required=True,help='Extracted complete private analysis bundle from Drive')
    args=parser.parse_args()
    bundle=args.data_bundle.resolve()
    required=['data','runs','results','evidence','src']
    if not all((bundle/name).is_dir() for name in required):parser.error('Bundle must contain data, runs, results, evidence and src folders.')
    if sha(bundle/'results/core_results.csv')!=sha(ROOT/'results/hasil_utama.csv'):parser.error('Bundle core results do not match this submission.')
    total=sum(p.stat().st_size for name in required for p in (bundle/name).rglob('*') if p.is_file())
    if total>250*2**20:parser.error('Use the small analysis bundle, not the model archive (limit 250 MiB).')
    parent=ROOT/'results/analysis_runs';parent.mkdir(exist_ok=True)
    workspace=Path(tempfile.mkdtemp(prefix='analysis_',dir=parent)).resolve()
    assert workspace.is_relative_to(parent.resolve())
    for name in required:shutil.copytree(bundle/name,workspace/name)
    source = ROOT / 'notebooks/eksperimen_indomeme.ipynb'
    target = ROOT / 'notebooks/eksperimen_indomeme.rerun.ipynb'
    receipt_path = workspace / 'runner_receipt.json'
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['OMP_NUM_THREADS'] = '2'
    os.environ['MKL_NUM_THREADS'] = '2'
    notebook = nbformat.read(source, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            cell.outputs = []
            cell.execution_count = None
    started = time.time()
    receipt = dict(status='running', started_at=datetime.now(timezone.utc).isoformat(),
                   hostname=socket.gethostname(), platform=platform.platform(),
                   input_sha256=sha(source), training_started=False,
                   purpose='Recompute analysis from retained experimental artifacts, not rerun training')
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    def cell_start(cell, cell_index, **kwargs):
        if cell.cell_type == 'code':
            print('Executing cell ' + str(cell_index), flush=True)
    try:
        client = NotebookClient(notebook, timeout=300, kernel_name='python3',
                                on_cell_start=cell_start, allow_errors=False,
                                resources={'metadata': {'path': str(workspace)}})
        client.execute()
        cells = [cell for cell in notebook.cells if cell.cell_type == 'code']
        assert cells and all(cell.execution_count is not None for cell in cells)
        assert not any(output.output_type == 'error' for cell in cells for output in cell.outputs)
        nbformat.write(notebook, target)
        body, _ = HTMLExporter().from_notebook_node(notebook)
        (ROOT / 'notebooks/eksperimen_indomeme.rerun.html').write_text(body, encoding='utf-8')
        receipt.update(status='passed', elapsed_seconds=round(time.time() - started, 2),
                       executed_code_cells=len(cells), output_sha256=sha(target),
                       output_bytes=target.stat().st_size,
                       visual_outputs=sum(output.output_type in ['display_data', 'execute_result']
                                          for cell in cells for output in cell.outputs))
    except Exception as exc:
        nbformat.write(notebook, workspace / 'analysis.failed.ipynb')
        receipt.update(status='failed', elapsed_seconds=round(time.time() - started, 2),
                       error_type=type(exc).__name__)
        raise
    finally:
        receipt_path.write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    main()
