#!/usr/bin/env python3
"""End-to-end automatic PTX fallback stage for a native-absent NVIDIA device.

Internal to tools/nvidia_baseline_gpu_batch.py --fallback-stage. It never
rents, connects to or releases anything; tools/nvidia_baseline_gpu_lease.sh
does the transport inside the one existing rental.

  plan     before the rental: validate every retained input and pin its hash
  prepare  on the orchestrator's machine, after collection and while the pod
           is still leased: generate the admission from this rental's PTX
           receipt and witness plus the retained native-device evidence, pack
           the NVIDIA vendor wheel with --bundle-ptx-admission, and stage the
           pod body
  body     what the pod runs: core plus that vendor wheel in a fresh venv with
           no forcing variable; the selection receipt, the canonical
           three-fixture column against the Apple reference, and two negative
           installs that must refuse with GpuPluginError and never use CPU

A failed comparison or admission writes no admission and stages nothing.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import nvidia_baseline_qualification as baseline

ROOT = Path(__file__).resolve().parents[1]
WITNESS = ROOT / 'tools/cuda_runtime_config_witness.py'
E2E = ROOT / 'tools/nvidia_ptx_fallback_e2e.py'
PACKER = ROOT / 'packaging/linux/pack_wheel.py'
MANIFEST_MEMBER = 'mojolearn/cuda_ptx/sm_80/PTX_BASELINE.json'
ADMISSION_MEMBER = 'mojolearn/cuda_ptx/sm_80/PTX_IDENTITY_ADMISSION.json'
# The stage owns these packer flags; everything else (sets, profile, proofs,
# the Linux math helper) is the orchestrator's retained build description.
OWNED_PACK_FLAGS = ('--wheels', '--bundle-ptx-admission', '--bundle-ptx', '--out')
# Sum of the body's own step bounds; the lease refuses to start it with less.
BODY_SECONDS = 2200
require = baseline.require
sha = baseline.sha


def pinned(path):
    path = Path(path).resolve()
    require(path.is_file(), 'Missing retained input: ' + str(path))
    return dict(file=str(path), sha256=sha(path))


def wheel_named(wheels, prefix):
    matches = [w for w in wheels if w.name.split('-')[0] == prefix]
    require(len(matches) == 1, 'Exactly one ' + prefix + ' wheel required')
    return matches[0]


def plan(commit, wheels, gpu_name, receipts, witnesses, apple, amd, apple_sha256, amd_sha256, pack_argv):
    """Pre-rental validation. Raises before any pod exists."""
    core, vendor, ptx = (wheel_named(wheels, name) for name in ('mojolearn', 'mojolearn_nvidia', 'mojolearn_nvidia_ptx80'))
    with zipfile.ZipFile(core) as z:
        require('mojolearn/ptx_admission.py' in z.namelist(),
                'Core wheel has no admitted-fallback loader (mojolearn/ptx_admission.py); '
                'the fallback stage needs wheels built from a source that has it')
    with zipfile.ZipFile(vendor) as z:
        names = z.namelist()
        require(any(n.startswith('mojolearn/cuda_native/') for n in names),
                'NVIDIA vendor wheel carries no native sets; the unbundled negative case needs the current layout')
        require(not any(n.startswith('mojolearn/cuda_ptx/') for n in names),
                'The staged NVIDIA vendor wheel must be the unbundled one')
    paired = {}
    baseline_uuids = set()
    require(receipts, 'Native-device receipts are required')
    for path in receipts:
        receipt = baseline.read(path)
        cap = tuple(receipt.get('hardware', {}).get('compute_capability', ()))
        require(receipt.get('schema') == baseline.SCHEMA and receipt.get('source_commit') == commit,
                'Receipt source or schema differs: ' + str(path))
        require(len(cap) == 2 and baseline.native_supported(cap),
                'Retained receipts must come from natively supported devices: ' + str(path))
        require(sha(Path(path).parent / receipt['column_file']) == receipt.get('column_sha256'),
                'Retained column hash differs: ' + str(path))
        paired.setdefault(cap, set()).add(receipt.get('role'))
        if receipt.get('role') == 'baseline':
            baseline_uuids.add(receipt['hardware']['uuid'])
    require(sum(roles == {'baseline', 'native-reference'} for roles in paired.values()) >= 2,
            'Need PTX and native receipts on two natively supported capabilities')
    witnessed = {baseline.read(path).get('device', {}).get('uuid') for path in witnesses}
    require(witnessed == baseline_uuids and len(witnesses) == len(witnessed),
            'One configuration witness per retained PTX receipt is required')
    references = {}
    for vendor_name, path, digest in (('apple', apple, apple_sha256), ('amd', amd, amd_sha256)):
        require(path is not None and digest and sha(path) == digest, 'Pinned reference hash differs: ' + vendor_name)
        require(baseline.read(path).get('commit') == commit, 'Reference column source differs: ' + vendor_name)
        references[vendor_name] = dict(file=str(Path(path).resolve()), sha256=digest)
    require(isinstance(pack_argv, list) and pack_argv and all(isinstance(x, str) for x in pack_argv)
            and '--set' in pack_argv, 'Packer argv must be a JSON list of strings naming --set directories')
    require(not any(x.split('=')[0] in OWNED_PACK_FLAGS for x in pack_argv),
            'Packer argv must not set ' + ', '.join(OWNED_PACK_FLAGS))
    return dict(source_commit=commit, gpu=gpu_name, body_seconds=BODY_SECONDS,
                core_wheel=core.name, plain_vendor_wheel=vendor.name,
                ptx_wheel=pinned(ptx), receipts=[pinned(p) for p in receipts],
                witnesses=[pinned(p) for p in witnesses], references=references,
                witness_script=pinned(WITNESS), e2e_script=pinned(E2E), pack_argv=pack_argv,
                negative_cases=['negative-config', 'negative-unbundled'])


def unchanged(row):
    require(sha(row['file']) == row['sha256'], 'Retained input changed since the plan: ' + row['file'])
    return Path(row['file'])


def negative_admission(record):
    """The same evidence without the native-absent device: a real record that
    admits only the natively supported configurations, so this device refuses."""
    absent = record['coverage']['native_absent']['configurations']
    document = copy.deepcopy(record)
    del document['coverage']['native_absent']
    document['configurations'] = [row for row in record['configurations'] if row not in absent]
    require(document['configurations'] and len(document['configurations']) < len(record['configurations']),
            'Negative admission must drop exactly the native-absent configurations')
    return document


def encoded(document):
    return (json.dumps(document, indent=2, sort_keys=True) + '\n').encode()


def packed_wheel(directory, admission_bytes, manifest_bytes, version):
    wheels = sorted(Path(directory).glob('mojolearn_nvidia-*.whl'))
    require(len(wheels) == 1 and wheels[0].name.split('-')[1] == version,
            'Packer did not emit exactly one same-version NVIDIA vendor wheel')
    with zipfile.ZipFile(wheels[0]) as z:
        require(z.read(ADMISSION_MEMBER) == admission_bytes and z.read(MANIFEST_MEMBER) == manifest_bytes,
                'Packed vendor wheel does not carry this admission and manifest')
    return wheels[0]


def body(document):
    return r'''#!/bin/bash
# Automatic IDENTICAL PTX fallback, end to end, on a device with no native payload.
set -euo pipefail
cd /root/ptx-batch
mkdir -p results/fallback
exec > results/fallback/body.log 2>&1
trap 'rc=$?; echo "$rc" > results/fallback/body.exit' EXIT
ROOT=$PWD
OUT=$ROOT/results/fallback
FB=$ROOT/fallback
(cd "$FB" && sha256sum -c SHA256SUMS)
cp "$FB/plan.json" "$FB/expect.json" "$OUT/"
# Nothing may force or hint a code path: the loader must choose by itself.
unset PYTHONPATH MOJOLEARN_CUDA_PATH MOJOLEARN_EXPERIMENTAL_PTX MOJOLEARN_GPU_ARCH MOJOLEARN_VENDOR MOJOLEARN_NUMERIC_MODE
export PYTHONNOUSERSITE=1 MOJOLEARN_COMMIT=@SHA@
install() {
    # A fresh venv holding the core and exactly one NVIDIA vendor wheel.
    local name=$1 vendor=$2
    timeout -k 20 60 python3 -m venv "fb-$name"
    timeout -k 20 180 "fb-$name/bin/python" -m pip install --no-input --only-binary=:all: "wheels/@CORE@" "$vendor" numpy > "$OUT/install-$name.log" 2>&1
    "fb-$name/bin/python" -m pip freeze > "$OUT/pip-freeze-$name.txt"
    if grep -i -E '^mojolearn[-_](nvidia[-_]ptx80|amd)' "$OUT/pip-freeze-$name.txt"; then return 1; fi
}
install positive "fallback/wheels-positive/@VENDOR@"
install negative-config "fallback/wheels-negative-config/@VENDOR@"
install negative-unbundled "wheels/@PLAIN@"
cd "$OUT"
# 1. The unforced import selects the admitted fallback.
timeout -k 20 140 "$ROOT/fb-positive/bin/python" "$FB/e2e.py" select --expect "$FB/expect.json" --out selection.json > selection.log 2>&1
# 2. The canonical three-fixture column through the fallback equals Apple's.
timeout -k 20 120 "$ROOT/fb-positive/bin/python" "$ROOT/source/tools/verify_lanes.py" --gpu-pass cuda --all --write-selection lanes.json > lanes.log 2>&1
LANES=$("$ROOT/fb-positive/bin/python" -c 'import json,sys; print(",".join(json.load(open(sys.argv[1]))["lanes"]))' lanes.json)
timeout -k 20 920 "$ROOT/venv/bin/python" "$ROOT/source/tools/nvidia_serial_guard.py" --seconds 900 --rss-gib 12 --cores 2 -- \
    "$ROOT/fb-positive/bin/python" -m mojolearn._identity_break --lanes "$LANES" --json column-cuda.json \
    --repeats 1 --fixtures base,denormal,odd --fail-on-refused --require-backend cuda --no-batch --no-rlpair > column.log 2>&1
cp "$FB/reference.json" reference.json
"$ROOT/fb-positive/bin/python" "$ROOT/source/tools/identity_break.py" --diff reference.json column-cuda.json \
    --require-columns 2 --lanes "$LANES" --json diff-ref-cuda.json > diff-ref-cuda.txt 2>&1
"$ROOT/fb-positive/bin/python" "$FB/e2e.py" column --expect "$FB/expect.json" --column column-cuda.json --out column-check.json > column-check.log 2>&1
# 3. A device configuration the admission does not name, and a vendor wheel
# with no bundled PTX: both refuse by name and never select CPU.
for case in @NEGATIVES@; do
    timeout -k 20 140 "$ROOT/fb-$case/bin/python" "$FB/e2e.py" refuse --case "$case" --out "$case.json" > "$case.log" 2>&1
done
echo FALLBACK_E2E_PASSED > passed.txt
'''.replace('@SHA@', document['source_commit']).replace('@CORE@', document['core_wheel']) \
        .replace('@VENDOR@', document['vendor_wheel']).replace('@PLAIN@', document['plain_vendor_wheel']) \
        .replace('@NEGATIVES@', ' '.join(document['negative_cases']))


def prepare(document, results, stage, *, build=None, run=subprocess.run):
    """Mid-rental, local only. Emits the stage directory last."""
    import admit_nvidia_ptx as admit
    build = build or admit.build
    api = admit.admission_api()
    results, stage = Path(results).resolve(), Path(stage).resolve()
    work = stage.parent / 'fallback-work'
    require(not stage.exists() and not work.exists(), 'Refusing to overwrite a retained fallback stage')
    for name in ('core_wheel', 'plain_vendor_wheel'):
        require(baseline.re.fullmatch('[A-Za-z0-9_.+-]+', document[name]), 'Unsafe wheel filename')
    # 1. The audited payload beside its manifest, equal to what the pod installed.
    payload = work / 'ptx-payload'
    payload.mkdir(parents=True)
    with zipfile.ZipFile(unchanged(document['ptx_wheel'])) as z:
        z.extractall(payload)
    manifest = payload / MANIFEST_MEMBER
    require(sha(manifest) == sha(results / 'PTX_BASELINE.json'), 'Pod installed another PTX manifest')
    # 2. Admission from this rental's receipt and witness plus retained evidence.
    receipts = [unchanged(row) for row in document['receipts']] + [results / 'full-baseline.json']
    witnesses = [unchanged(row) for row in document['witnesses']] + [results / 'cuda-runtime-config-ptx.json']
    references = {vendor: unchanged(row) for vendor, row in document['references'].items()}
    hashes = {vendor: row['sha256'] for vendor, row in document['references'].items()}
    record, reports = build(manifest, receipts, references, hashes, witnesses, unchanged(document['witness_script']))
    absent = record['coverage'].get('native_absent', {}).get('configurations', [])
    require(len(absent) == 1 and absent[0]['device_name'] == document['gpu']
            and not baseline.native_supported(absent[0]['compute_capability']),
            'Admission does not name exactly this native-absent device')
    negative = negative_admission(record)
    api.validate_admission(negative, source_commit=record['source_commit'], manifest_sha256=record['manifest_sha256'])
    admissions = {}
    for name, decision in (('positive', record), ('negative-config', negative)):
        directory = work / ('admission-' + name)
        directory.mkdir()
        if name == 'positive':
            for report_name, report in reports.items():
                (directory / report_name).write_bytes(encoded(report))
        (directory / api.ADMISSION_FILE).write_bytes(encoded(decision))
        admissions[name] = directory / api.ADMISSION_FILE
    # 3. Two vendor wheels from the same sets; only the bundled admission differs.
    version = document['core_wheel'].split('-')[1]
    packed = {}
    for name, admission in admissions.items():
        out = work / 'packed' / name
        command = [sys.executable, str(PACKER), *document['pack_argv'], '--wheels', 'nvidia',
                   '--bundle-ptx-admission', str(admission), '--out', str(out)]
        with (work / ('pack-' + name + '.log')).open('w') as log:
            done = run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
        require(done.returncode == 0, 'Packing the ' + name + ' vendor wheel failed; see pack-' + name + '.log')
        packed[name] = packed_wheel(out, admission.read_bytes(), manifest.read_bytes(), version)
    require(packed['positive'].name == packed['negative-config'].name, 'Packed wheel names differ')
    # 4. The pod stage, written last.
    staged = dict(document, vendor_wheel=packed['positive'].name,
                  admission_sha256=sha(admissions['positive']),
                  negative_admission_sha256=sha(admissions['negative-config']),
                  vendor_wheel_sha256={name: sha(path) for name, path in packed.items()})
    expect = dict(source_commit=record['source_commit'], manifest_sha256=record['manifest_sha256'],
                  admission_sha256=staged['admission_sha256'], configuration=absent[0])
    build_dir = work / 'stage'
    build_dir.mkdir()
    for name, path in packed.items():
        (build_dir / ('wheels-' + name)).mkdir()
        shutil.copyfile(path, build_dir / ('wheels-' + name) / path.name)
    shutil.copyfile(unchanged(document['e2e_script']), build_dir / 'e2e.py')
    shutil.copyfile(references['apple'], build_dir / 'reference.json')
    (build_dir / 'plan.json').write_bytes(encoded(staged))
    (build_dir / 'expect.json').write_bytes(encoded(expect))
    (build_dir / 'body.sh').write_text(body(staged))
    files = sorted(p for p in build_dir.rglob('*') if p.is_file())
    (build_dir / 'SHA256SUMS').write_text(''.join(f'{sha(p)}  {p.relative_to(build_dir)}\n' for p in files))
    build_dir.rename(stage)
    return staged


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('prepare')
    for name in ('plan', 'results', 'stage'):
        command.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    staged = prepare(baseline.read(args.plan), args.results, args.stage)
    print(json.dumps(dict(stage=str(args.stage), admission_sha256=staged['admission_sha256'],
                          vendor_wheel=staged['vendor_wheel'])))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, subprocess.SubprocessError, zipfile.BadZipFile) as exc:
        raise SystemExit('REFUSED: ' + str(exc))
