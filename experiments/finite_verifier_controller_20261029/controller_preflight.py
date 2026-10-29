"""Fail closed before installed verification; no product execution in these checks."""
import json
from pathlib import Path
import subprocess


def frozen_inputs(root, package, ledger, revision):
    if len(revision) != 40 or any(c not in '0123456789abcdef' for c in revision):
        raise ValueError('exact source commit required')
    actual = subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(root),'status','--porcelain','--untracked-files=no'],text=True).strip()
    if actual != revision or dirty:
        raise ValueError('source checkout differs or has tracked modifications')
    if (Path(package)/'identity_columns/COMMIT').read_text().strip() != revision:
        raise ValueError('installed verifier payload source differs')
    doc = json.loads(Path(ledger).read_text())
    if doc['source_revision'] != revision:
        raise ValueError('central ledger has another source freeze')
    return doc


def exact_begin(directory, vendor, lane, revision, document):
    paths = list(Path(directory).glob('identity-launch-*.json'))
    if len(paths) != 1:
        raise ValueError('expected exactly one new launch record')
    launch = json.loads(paths[0].read_text());rid=f'identity-{vendor}/{lane}'
    if (launch.get('phase')!='before-execution' or launch.get('vendor')!=vendor
            or launch.get('source_revision')!=revision or launch.get('lanes') != [lane]
            or set(launch.get('runs',{}))!={rid}):
        raise ValueError('lane was not begun exactly once at the frozen source')
    run=launch['runs'][rid];wanted=document['requirements'][rid]
    if (run.get('source_revision')!=revision or not run.get('run_id')
            or run.get('fingerprint')!=wanted['fingerprint']):
        raise ValueError('begun run fingerprint differs')
    return paths[0]
