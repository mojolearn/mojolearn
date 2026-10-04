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
AMD_POOL = {
    'MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_SIZE': '1073741824',
    'MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_ONLY': 'true',
    'MODULAR_DEVICE_CONTEXT_MEMORY_MANAGER_CHUNK_PERCENT': '100',
}


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


def prepare(wheels, script, commit, directory, *, patch_manifest=None, patch_binding=None, patch_proof=None):
    rows = inventory(wheels, commit)
    patch = None
    if any(p is not None for p in (patch_manifest, patch_binding, patch_proof)):
        if not all(p is not None for p in (patch_manifest, patch_binding, patch_proof)):
            raise ValueError('Patch manifest, binding and proof must be supplied together')
        from amd_diagnostic_patch import validate
        patch = validate(patch_manifest, patch_proof, commit, binding=patch_binding, wheels=rows)
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
                  'nvidia_serial_guard.py': ROOT / 'tools/nvidia_serial_guard.py',
                  'amd_diagnostic_patch.py': ROOT / 'tools/amd_diagnostic_patch.py'})
    if patch:
        files.update({'patch-manifest.json': Path(patch_manifest), 'patch-proof.json': Path(patch_proof),
                      'patched-binding.so': Path(patch_binding)})
    for name, path in files.items():
        (stage / name).write_bytes(path.read_bytes())
    plan = dict(schema='mojolearn.amd-diagnostic-plan.v1', source_commit=commit,
                wheels=rows, script_sha256=digest(script), lease_minutes=30, cap_cents=300,
                diagnostic_seconds=1200, amd_runtime_pool=AMD_POOL, diagnostic_patch=patch,
                release_qualified=False)
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
# The existing AMD campaign pool (gemm_remote_leg.sh) avoids the runtime's
# near-device-sized reservation. This does not change the external 85% guard.
eval "$(venv/bin/python - <<'POOL'
import json, pathlib, shlex
pool=json.loads(pathlib.Path('plan.json').read_text())['amd_runtime_pool']
for name, value in pool.items():
    print('export '+name+'='+shlex.quote(value))
pathlib.Path('results/runtime-pool.json').write_text(json.dumps(pool,indent=2)+'\\n')
POOL
)"
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
if [[ ! -f patch-manifest.json ]]; then
    venv/bin/python amd_serial_guard.py --seconds 1200 --rss-gib 12 --cores 2 -- venv/bin/python diagnose.py --source-commit "$commit" --out "$PWD/results/diagnostic" > results/diagnostic.log 2>&1
else
    # Separate installs and fresh processes; original wheel files and original
    # installed binding remain unchanged. Total guarded work <=1200 seconds.
    venv/bin/python amd_serial_guard.py --seconds 450 --rss-gib 12 --cores 2 -- venv/bin/python diagnose.py --source-commit "$commit" --out "$PWD/results/original" > results/original.log 2>&1
    python3 -m venv fixed
    timeout -k 10 180 fixed/bin/pip install --disable-pip-version-check --retries 1 --timeout 30 ./*.whl numpy >> results/setup-fixed.log 2>&1
    fixed/bin/python - <<'PATCH'
import json, pathlib, importlib.metadata as md
from amd_diagnostic_patch import apply
plan=json.loads(pathlib.Path('plan.json').read_text())
rows=[dict(row,path=str(pathlib.Path(row['file']).resolve())) for row in plan['wheels']]
parent=md.distribution('mojolearn').locate_file('')
witness=apply('patch-manifest.json','patch-proof.json','patched-binding.so',parent,plan['source_commit'],rows)
pathlib.Path('results/patch-install.json').write_text(json.dumps(witness,indent=2)+'\\n')
PATCH
    fixed/bin/python amd_serial_guard.py --seconds 450 --rss-gib 12 --cores 2 -- fixed/bin/python diagnose.py --source-commit "$commit" --out "$PWD/results/fixed" --patch-manifest patch-manifest.json --patch-proof patch-proof.json > results/fixed.log 2>&1
    # This is the original installed failing lane, still using its frozen
    # harness; the fixed diagnostic records the actual loaded patch hash.
    export MOJOLEARN_COMMIT="$commit"
    fixed/bin/python amd_serial_guard.py --seconds 300 --rss-gib 12 --cores 2 -- fixed/bin/python -m mojolearn._identity_break --lanes trees-oob-cv-link --json "$PWD/results/fixed-lane.json" --fixtures base,denormal,odd --repeats 1 --fail-on-refused --require-backend hip --no-batch --no-rlpair > results/fixed-lane.log 2>&1
    fixed/bin/python - <<'WITNESS'
import json, pathlib
patch=json.loads(pathlib.Path('patch-manifest.json').read_text())
column=json.loads(pathlib.Path('results/fixed-lane.json').read_text())
loaded=[r for r in column['package']['bindings'] if r['module']=='_mojolearn_x_trees']
assert len(loaded)==1 and loaded[0]['sha256']==patch['patched_sha256']
assert column['complete'] and len(column['cells'])==3
assert all(row['verdict']=='STABLE' for row in column['cells'].values())
pathlib.Path('results/fixed-lane-witness.json').write_text(json.dumps(dict(loaded_binding=loaded[0],patch=patch,qualification=False),indent=2)+'\\n')
WITNESS
fi
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
    p.add_argument('--patch-manifest', type=Path)
    p.add_argument('--patch-binding', type=Path)
    p.add_argument('--patch-proof', type=Path)
    a = p.parse_args()
    bundle = prepare(a.wheel, a.script, a.source_commit, a.out,
                     patch_manifest=a.patch_manifest, patch_binding=a.patch_binding, patch_proof=a.patch_proof)
    return subprocess.run(['bash', str(ROOT / 'tools/amd_diagnostic_lease.sh'), str(bundle),
                           str(a.out / 'lease'), 'rent' if a.rent else 'dry-run']).returncode


if __name__ == '__main__':
    raise SystemExit(main())
