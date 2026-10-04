#!/usr/bin/env python3
"""Untimed existing LU/PCA bit checks with per-process source provenance."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--vendor', choices=('cuda', 'hip'), required=True)
    a = p.parse_args()
    if sys.platform != 'linux':
        p.error('run only on the authorized Linux boxes')
    root = a.source.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=root, check=True)
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    report = {'source_sha': sha, 'source': str(root), 'purpose': 'untimed bit checks',
              'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'fixture_sha256': hashlib.sha256(Path(__file__).with_name('idn_all_checks.py').read_bytes()).hexdigest(),
              'status': 'RUNNING', 'checks': {}}
    receipt = a.out / 'report.json'

    def save():
        tmp = receipt.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2) + '\n')
        tmp.replace(receipt)

    # Run the existing fixtures unchanged, then inspect that very process's
    # loaded libraries. A parent import alone cannot prove child provenance.
    child = '''import json, runpy, sys
from pathlib import Path
root, harness, gate, vendor = sys.argv[1:]
sys.path.insert(0, str(Path(root) / 'python'))
sys.path.insert(0, harness)
from identical_wave_worker import source_provenance
sys.argv = [str(Path(harness) / 'idn_all_checks.py'), gate, '--child']
try:
    runpy.run_path(sys.argv[0], run_name='__main__')
except SystemExit as exc:
    if exc.code not in (None, 0):
        raise
print('SOURCE_PROVENANCE ' + json.dumps(source_provenance(Path(root), vendor)))
'''
    save()
    for gate in ('lu-nan', 'pca-id'):
        row = {'status': 'RUNNING', 'columns': {}}
        report['checks'][gate] = row
        save()
        for vendor in (a.vendor, 'cpu'):
            env = {k: v for k, v in os.environ.items() if not k.startswith(('MOJOLEARN_', 'MODULAR_MOJO_'))}
            env.update(MOJOLEARN_VENDOR=vendor, MOJOLEARN_NUMERIC_MODE='identical',
                       PYTHONPATH=str(root / 'python'), OMP_NUM_THREADS='1',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', PYTHONUNBUFFERED='1',
                       LD_LIBRARY_PATH=str(root / 'python/mojolearn/.libs') + ':' + env.get('LD_LIBRARY_PATH', ''))
            log = a.out / (gate + '-' + vendor + '.log')
            with log.open('w') as out:
                proc = subprocess.Popen([sys.executable, '-c', child, str(root),
                                         str(Path(__file__).resolve().parent), gate, vendor],
                                        cwd=root, env=env, stdout=out, stderr=subprocess.STDOUT)
                try:
                    rc = proc.wait(timeout=1800)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                    rc = 124
            lines = log.read_text().splitlines()
            digests = [s.split()[1:] for s in lines if s.startswith('DIGEST ')]
            provenance = [json.loads(s.removeprefix('SOURCE_PROVENANCE '))
                          for s in lines if s.startswith('SOURCE_PROVENANCE ')]
            row['columns'][vendor] = {'rc': rc, 'log': str(log),
                                      'digest': digests[-1] if digests else None,
                                      'provenance': provenance[-1] if provenance else None}
            save()
        columns = list(row['columns'].values())
        valid = all(c['rc'] == 0 and c['digest'] and c['provenance'] for c in columns)
        if valid:
            valid = columns[0]['digest'] == columns[1]['digest']
            if gate == 'lu-nan':
                valid = valid and columns[0]['digest'][1] != 'nan_words=0'
        row['status'] = 'PASS' if valid else 'FAILED'
        save()
        print('SOURCE_BIT_CHECK', gate, row['status'], flush=True)
    report['status'] = 'PASS' if all(c['status'] == 'PASS' for c in report['checks'].values()) else 'FAILED'
    save()
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    sys.exit(main())
