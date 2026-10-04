"""Prepare an exact six-wheel AMD diagnostic bundle; renting requires --rent."""
import argparse
import base64
import csv
import email
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import tarfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PROJECTS = {'mojolearn', 'mojolearn-amd', 'mojolearn-nvidia',
            'mojolearn-amd-gfx942', 'mojolearn-nvidia-sm89', 'mojolearn-nvidia-sm90'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def inventory(wheels, commit):
    if not re.fullmatch('[0-9a-f]{40}', commit):
        raise ValueError('Full source commit required')
    rows = []
    for path in wheels:
        path = Path(path).resolve()
        if not re.fullmatch(r'[A-Za-z0-9_.-]+\.whl', path.name):
            raise ValueError('Unsafe wheel filename')
        with zipfile.ZipFile(path) as z:
            names = z.namelist()
            metadata = [n for n in names if n.endswith('.dist-info/METADATA')]
            if len(metadata) != 1 or len(names) != len(set(names)):
                raise ValueError('Ambiguous wheel metadata/members')
            dist = metadata[0].rsplit('/', 1)[0]
            m = email.message_from_bytes(z.read(metadata[0]))
            source = json.loads(z.read(dist + '/LINUX_PAYLOAD.json'))
            if source.get('source_commit') != commit:
                raise ValueError('Wheel source commit differs')
            record = dict((name, (sha, size)) for name, sha, size in
                          csv.reader(io.StringIO(z.read(dist + '/RECORD').decode())))
            for name in names:
                if name.endswith('/') or name == dist + '/RECORD':
                    continue
                data = z.read(name)
                expected = 'sha256=' + base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip('=')
                if record.get(name) != (expected, str(len(data))):
                    raise ValueError('Wheel RECORD mismatch: ' + name)
            if m['Name'] == 'mojolearn' and z.read('mojolearn/identity_columns/COMMIT').decode().strip() != commit:
                raise ValueError('Core COMMIT differs')
            rows.append(dict(file=path.name, path=str(path), distribution=m['Name'],
                             version=m['Version'], sha256=digest(path), source_commit=commit))
    if len(rows) != 6 or {r['distribution'] for r in rows} != PROJECTS or len({r['version'] for r in rows}) != 1:
        raise ValueError('Exactly six matching released projects required')
    return rows


def prepare(wheels, script, commit, directory):
    rows = inventory(wheels, commit)
    from wheel_api_audit import split_audit
    report = split_audit(wheels)
    if report['problems']:
        raise ValueError('Wheel ownership/dependencies failed: ' + '; '.join(report['problems']))
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    stage = directory / 'stage'
    stage.mkdir()
    files = {r['file']: Path(r['path']) for r in rows}
    files.update({'diagnose.py': Path(script), 'amd_serial_guard.py': ROOT / 'tools/amd_serial_guard.py',
                  'nvidia_serial_guard.py': ROOT / 'tools/nvidia_serial_guard.py'})
    for name, path in files.items():
        (stage / name).write_bytes(path.read_bytes())
    plan = dict(schema='mojolearn.amd-diagnostic-plan.v1', source_commit=commit,
                wheels=rows, script_sha256=digest(script), lease_minutes=30, cap_cents=300,
                diagnostic_seconds=1200, release_qualified=False)
    (directory / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    (stage / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    (stage / 'run.sh').write_text('''#!/bin/bash
set -euo pipefail
mkdir -p results
cp plan.json SHA256SUMS results/
python3 -m venv venv > results/setup.log 2>&1 || { apt-get -o DPkg::Lock::Timeout=60 update -qq >> results/setup.log 2>&1; apt-get -o DPkg::Lock::Timeout=60 install -y -qq python3-venv >> results/setup.log 2>&1; python3 -m venv venv >> results/setup.log 2>&1; }
timeout -k 10 180 venv/bin/pip install --disable-pip-version-check --retries 1 --timeout 30 ./*.whl numpy >> results/setup.log 2>&1
export MOJOLEARN_NUMERIC_MODE=identical PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
venv/bin/python - <<'PY'
import json, pathlib, importlib.metadata as md, mojolearn as m
from mojolearn import _backend
p=json.loads(pathlib.Path('plan.json').read_text())
assert m.vendor()=='hip' and _backend.gpu_arch()=='gfx942'
assert m.numeric_mode()=='identical'
assert pathlib.Path(m.__file__).parent.joinpath('identity_columns/COMMIT').read_text().strip()==p['source_commit']
versions={r['distribution']:md.version(r['distribution']) for r in p['wheels']}
assert all(versions[r['distribution']]==r['version'] for r in p['wheels'])
plugin=_backend.gpu_plugin()
assert plugin['code_format']=='native' and plugin['payloads']==['mojolearn-amd-gfx942']
pathlib.Path('results/installed.json').write_text(json.dumps(dict(vendor=m.vendor(), architecture=_backend.gpu_arch(), plugin=plugin, source_commit=p['source_commit'], distributions=versions, package=m.__file__),indent=2))
PY
commit=$(venv/bin/python -c 'import json;print(json.load(open("plan.json"))["source_commit"])')
venv/bin/python amd_serial_guard.py --seconds 1200 --rss-gib 12 --cores 2 -- venv/bin/python diagnose.py --source-commit "$commit" --out "$PWD/results/diagnostic" > results/diagnostic.log 2>&1
''')
    (stage / 'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.name}\n' for p in sorted(stage.iterdir())))
    bundle = directory / 'bundle.tgz'
    with tarfile.open(bundle, 'w:gz') as archive:
        for p in sorted(stage.iterdir()):
            archive.add(p, arcname=p.name)
    return bundle


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--wheel', action='append', type=Path, required=True)
    p.add_argument('--script', type=Path, required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--rent', action='store_true')
    a = p.parse_args()
    bundle = prepare(a.wheel, a.script, a.source_commit, a.out)
    return subprocess.run(['bash', str(ROOT / 'tools/amd_diagnostic_lease.sh'), str(bundle),
                           str(a.out / 'lease'), 'rent' if a.rent else 'dry-run']).returncode


if __name__ == '__main__':
    raise SystemExit(main())
