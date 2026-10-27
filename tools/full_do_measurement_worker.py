#!/usr/bin/env python3
"""One frozen full measurement campaign on an already-owned DigitalOcean box."""
import glob
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time


def run(root):
    root = Path(root)
    cfg = json.loads((root/'config.json').read_text())
    source = root/'source'
    source.mkdir(exist_ok=True)
    with tarfile.open(root/'source.tar.gz') as archive:
        archive.extractall(source, filter='data')
    # Sparse source checkout at the real frozen main commit. Verify every
    # transferred file against Git's blob identity before hiding omitted data
    # and historical evidence paths with skip-worktree bits.
    with (root/'source-install.log').open('w') as log:
        for cmd in (['git', 'init', '-q'],
                    ['git', 'remote', 'add', 'origin', 'https://github.com/mojolearn/mojolearn.git'],
                    ['git', 'fetch', '--depth=1', '--filter=blob:none', 'origin', cfg['source_commit']],
                    ['git', 'update-ref', 'HEAD', 'FETCH_HEAD'], ['git', 'read-tree', 'HEAD']):
            subprocess.run(cmd, cwd=source, stdout=log, stderr=log, check=True)
        tree = subprocess.check_output(['git', 'ls-tree', '-rz', 'HEAD'], cwd=source).split(b'\0')
        omitted = []
        for entry in tree:
            if not entry:
                continue
            metadata, name = entry.split(b'\t', 1)
            mode, kind, blob = metadata.split()
            path = source/os.fsdecode(name)
            if not path.is_file():
                omitted.append(name)
                continue
            data = path.read_bytes()
            actual = hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest().encode()
            if actual != blob:
                raise RuntimeError('frozen source differs from main: '+os.fsdecode(name))
        subprocess.run(['git', 'update-index', '--skip-worktree', '-z', '--stdin'], cwd=source,
                       input=b'\0'.join(omitted)+b'\0', stdout=log, stderr=log, check=True)
    sys.path.insert(0, str(source/'tools'))
    import bench_board as B
    import bench_board_evidence as E
    import gate_smoke_box as Gate
    os.environ['MOJOLEARN_NUMERIC_MODE'] = 'identical'
    os.environ['GBM_BENCH_DATA'] = '/root/datasets/gbm-bench'
    os.environ.pop('PYTHONPATH', None)
    os.environ.pop('PYTHONHOME', None)
    if cfg.get('bootstrap'):
        setup = Gate.Gate(cfg['vendor'], cfg['version'], root/'runtime', profile='performance-campaign-v1')
        setup.setup()
        python, manifest, cache = setup.state['python'], setup.state['wheel_manifest'], setup.cache
    else:
        python, manifest, cache = cfg['python'], cfg['wheel_manifest'], cfg['cache']
    if cfg['vendor'] == 'amd':
        os.environ['LD_LIBRARY_PATH'] = ':'.join(glob.glob(cache+'/venv/lib/python*/site-packages/*/lib64'))
    # Prove remote retention works before any fit starts.
    proof = E.archive(root/'retention-preflight.tar.gz', {'run.json': E.encoded(cfg)})
    E.write(root/'retention-preflight.json', E.public_ref(E.upload(proof)))
    campaign = root/'campaign'
    cmd = [python, str(source/'tools/gpu_gap_campaign.py'), '--vendor', cfg['vendor'],
           '--manifest', str(root/'manifest.json'), '--out', str(campaign), '--cache', cache,
           '--python-env', python, '--wheel-manifest', manifest, '--cpu-fallback',
           '--opponent-store', str(root/'opponent-store.jsonl'), '--keep-going',
           '--stop-at', str(time.time()+cfg['hours']*3600)]
    E.write(root/'invocation.json', {'argv': cmd, 'frozen_main': cfg['source_commit'],
                                   'version': cfg['version'], 'whole_plan': True})
    small = root/'recipe-logs'
    small.mkdir(exist_ok=True)
    with (root/'campaign-controller.log').open('w') as log:
        proc = subprocess.Popen(cmd, cwd=source, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            log.write(line)
            log.flush()
            if not line.startswith('CAMPAIGN_RECIPE_DONE '):
                continue
            event = json.loads(line.split(' ', 1)[1])
            directory = Path(event['directory'])
            path = small/(event['id'].replace('/', '.')+'.tar.gz')
            with tarfile.open(path, 'w:gz', compresslevel=1) as archive:
                for f in directory.rglob('*'):
                    if f.is_file() and f.suffix in ('.json', '.log', '.txt', '.csv') and f.stat().st_size <= 50*1024*1024:
                        archive.add(f, arcname=str(f.relative_to(directory)), recursive=False)
                recipe_log = campaign/(event['id'].replace('/', '.')+'.log')
                if recipe_log.exists():
                    archive.add(recipe_log, arcname='recipe-controller.log')
            event['local_archive'] = str(path)
            # A completion event reaches the Mac only after R2 retention.
            print('CAMPAIGN_RECIPE_DONE '+json.dumps(event), flush=True)
        rc = proc.wait()
    gate = Gate.Gate(cfg['vendor'], cfg['version'], root/'runtime', profile='performance-campaign-v1')
    merged = gate.merge_boards(campaign)
    E.write(root/'exit.json', {'returncode': rc, 'finished': time.time(), 'board': merged})
    return rc


if __name__ == '__main__':
    raise SystemExit(run(sys.argv[1]))
