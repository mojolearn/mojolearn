#!/usr/bin/env python3
"""Guarded experimental native/PTX comparison on ONE GPU. Dry-run by default.

Run twice with the same wheel directory and source: --gpu ada, then --gpu hopper.
All seven exact wheels are staged; no index copy of mojolearn is installed.
This does not grant release admission or universal IDENTICAL qualification.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[1]
GPUS = {'ada': 'NVIDIA GeForce RTX 4090', 'hopper': 'NVIDIA H100 80GB HBM3'}
DISTS = {'mojolearn', 'mojolearn_nvidia', 'mojolearn_amd', 'mojolearn_nvidia_sm89',
         'mojolearn_nvidia_sm90', 'mojolearn_amd_gfx942', 'mojolearn_nvidia_ptx80'}
LANES = 'kmeans,gemm-pinned,ridge,gbdt-symmetric,mamba2,transformer,svc'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def artifacts(directory, commit):
    require(re.fullmatch('[0-9a-f]{40}', commit), 'Full source SHA required')
    wheels = sorted(Path(directory).resolve().glob('*.whl'))
    require(len(wheels) == 7 and {p.name.split('-')[0] for p in wheels} == DISTS,
            'Exactly six native release wheels plus experimental PTX wheel required')
    require(len({p.name.split('-')[1] for p in wheels}) == 1, 'Wheel versions differ')
    manifest = None
    for wheel in wheels:
        require(re.fullmatch('[A-Za-z0-9_.+-]+', wheel.name), 'Unsafe wheel filename')
        with zipfile.ZipFile(wheel) as z:
            name = wheel.name.split('-')[0]
            if name == 'mojolearn_nvidia_ptx80':
                matches = [p for p in z.namelist() if p.endswith('/PTX_BASELINE.json')]
                require(matches == ['mojolearn/cuda_ptx/sm_80/PTX_BASELINE.json'],
                        'Baseline manifest missing, ambiguous or outside its registered directory')
                manifest = json.loads(z.read(matches[0]))
                require(manifest.get('source_commit') == commit and manifest.get('source_dirty') is False
                        and manifest.get('code_format') == 'ptx-baseline', 'Baseline source/format differs')
                prefix = matches[0].rsplit('/', 1)[0] + '/'
                require(manifest.get('files'), 'Empty baseline payload')
                for row in manifest['files']:
                    require(hashlib.sha256(z.read(prefix + row['file'])).hexdigest() == row['sha256'],
                            'Baseline wheel payload does not match manifest')
            else:
                inventories = [p for p in z.namelist() if p.endswith('.dist-info/LINUX_PAYLOAD.json')]
                require(len(inventories) == 1, 'Native wheel lacks release inventory')
                require(json.loads(z.read(inventories[0])).get('source_commit') == commit,
                        'Native inventory source differs')
                if name == 'mojolearn':
                    require(z.read('mojolearn/identity_columns/COMMIT').decode().strip() == commit,
                            'Installed core source differs')
    return wheels, manifest


def box_body(commit, full=True, extra_capture=False):
    # All interpolated values are validated SHA or constants. No remote credentials.
    return '''#!/bin/bash
set -euo pipefail
cd /root/ptx-batch
mkdir -p results
exec > results/body.log 2>&1
trap 'rc=$?; echo "$rc" > results/body.exit' EXIT
sha256sum -c SHA256SUMS
cp plan.json results/plan.json
nvidia-smi > results/nvidia-smi.txt
python3 -m venv venv
timeout -k 20 300 venv/bin/python -m pip install --no-input wheels/*.whl numpy > results/install.log 2>&1
venv/bin/python -m pip freeze > results/pip-freeze.txt
git init source
git -C source remote add origin https://github.com/mojolearn/mojolearn.git
git -C source sparse-checkout init --no-cone
printf '/*\\n!/bench/results/\\n!/bench/mamba/corpus/\\n!/bench/oracle_*/\\n!/bench/minentropy_oracle.txt\\n' | git -C source sparse-checkout set --no-cone --stdin
timeout -k 20 300 git -C source fetch --filter=blob:none --depth 1 origin @SHA@
timeout -k 20 180 git -C source checkout --detach FETCH_HEAD
test "$(git -C source rev-parse HEAD)" = @SHA@
# Resolve installed payload without importing mojolearn or preselecting a backend.
MANIFEST=$(venv/bin/python - <<'INNER'
import pathlib,sysconfig
p=pathlib.Path(sysconfig.get_paths()['purelib'])/'mojolearn/cuda_ptx/sm_80/PTX_BASELINE.json'
if not p.is_file(): raise SystemExit('Installed baseline manifest missing from its registered directory')
print(p)
INNER
)
cp "$MANIFEST" results/PTX_BASELINE.json
export MOJOLEARN_NUMERIC_MODE=identical PYTHONNOUSERSITE=1
unset PYTHONPATH MOJOLEARN_CUDA_PATH MOJOLEARN_EXPERIMENTAL_PTX
@EXTRA@
collect() {
    local scope=$1 role=$2 bound=$3
    shift 3
    local cmd=(venv/bin/python source/tools/nvidia_baseline_qualification.py collect --role "$role" --manifest "$MANIFEST" --out "results/$scope-$role.json" "$@")
    if [ "$role" = baseline ]; then
        timeout -k 20 "$bound" env MOJOLEARN_CUDA_PATH=ptx-baseline MOJOLEARN_EXPERIMENTAL_PTX=1 "${cmd[@]}" > "results/$scope-$role.log" 2>&1
    else
        timeout -k 20 "$bound" "${cmd[@]}" > "results/$scope-$role.log" 2>&1
    fi
}
for role in native-reference baseline; do
    collect prototype "$role" 300 --lanes @LANES@ --fixtures base,denormal,odd
done
venv/bin/python source/tools/nvidia_baseline_qualification.py check --prototype --manifest "$MANIFEST" --out results/prototype-comparison.json results/prototype-native-reference.json results/prototype-baseline.json
@FULL@
test "$EXTRA_FAILED" = 0
''' .replace('@SHA@', commit).replace('@LANES@', LANES).replace('@EXTRA@', '''EXTRA_FAILED=0
for role in native-reference baseline; do
    cmd=(venv/bin/python source/tools/nvidia_serial_guard.py --seconds 120 --rss-gib 12 --cores 2 -- venv/bin/python extra-wrapper.py --script extra-capture.py --source-tools source/tools --manifest "$MANIFEST" --role "$role" --out "results/extra-$role.json")
    if [ "$role" = baseline ]; then
        timeout -k 20 140 env MOJOLEARN_CUDA_PATH=ptx-baseline MOJOLEARN_EXPERIMENTAL_PTX=1 "${cmd[@]}" > "results/extra-$role.log" 2>&1 || EXTRA_FAILED=1
    else
        timeout -k 20 140 "${cmd[@]}" > "results/extra-$role.log" 2>&1 || EXTRA_FAILED=1
    fi
done''' if extra_capture else 'EXTRA_FAILED=0').replace('@FULL@', '''for role in native-reference baseline; do
    collect full "$role" 2400
done
# Single-pod comparison is deliberately prototype-labelled; cross-pod full
# admission check happens locally against the complete original payload set.
venv/bin/python source/tools/nvidia_baseline_qualification.py check --prototype --manifest "$MANIFEST" --out results/full-local-comparison.json results/full-native-reference.json results/full-baseline.json''' if full else '')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('commit')
    p.add_argument('--wheels', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gpu', choices=GPUS, required=True)
    p.add_argument('--prototype-only', action='store_true')
    p.add_argument('--rent', action='store_true')
    p.add_argument('--extra-capture', type=Path, help='Supplemental script, native and PTX, 120 seconds each')
    p.add_argument('--tooling-commit', help='Explicit full clean runner SHA when supplementary tooling differs from payload source')
    args = p.parse_args()
    wheels, manifest = artifacts(args.wheels, args.commit)
    require(not args.out.exists(), 'Output already exists')
    actual = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    tooling = args.tooling_commit or args.commit
    require(re.fullmatch('[0-9a-f]{40}', tooling) and actual == tooling,
            'Run the runner from the frozen candidate checkout or name its exact tooling commit')
    require(subprocess.run(['git', '-C', str(ROOT), 'diff', '--quiet', 'HEAD']).returncode == 0,
            'Tracked source dirty')
    refs = subprocess.check_output(['git', 'ls-remote', '--refs', 'https://github.com/mojolearn/mojolearn.git'],
                                   text=True, timeout=60)
    require(any(line.startswith(args.commit + '\t') for line in refs.splitlines()), 'Push frozen source first')
    require(any(line.startswith(tooling + '\t') for line in refs.splitlines()), 'Push frozen tooling first')
    plan = dict(source_commit=args.commit, gpu=GPUS[args.gpu], lease_minutes=120,
                work_seconds=6300, full=not args.prototype_only,
                tooling_commit=tooling, tooling_dirty=False,
                extra_capture_sha256=sha(args.extra_capture) if args.extra_capture else None,
                extra_wrapper_sha256=sha(ROOT / 'tools/nvidia_extra_capture.py') if args.extra_capture else None,
                extra_capture_seconds_per_role=120 if args.extra_capture else 0,
                wheels={w.name: sha(w) for w in wheels}, identical_qualified=False)
    print(json.dumps(plan, indent=2), flush=True)
    if not args.rent:
        return 0
    args.out.mkdir(parents=True)
    (args.out / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    with tempfile.TemporaryDirectory(prefix='ptx-gpu-stage-') as tmp:
        stage = Path(tmp)
        (stage / 'wheels').mkdir()
        import shutil
        for wheel in wheels:
            shutil.copyfile(wheel, stage / 'wheels' / wheel.name)
            require(sha(stage / 'wheels' / wheel.name) == plan['wheels'][wheel.name],
                    'Wheel bytes changed while staging; refuse before rental')
        artifacts(stage / 'wheels', args.commit)  # revalidate the exact staged bytes before creating a pod
        (stage / 'body.sh').write_text(box_body(args.commit, not args.prototype_only, bool(args.extra_capture)))
        files = sorted((stage / 'wheels').glob('*.whl')) + [stage / 'body.sh']
        (stage / 'plan.json').write_text(json.dumps(plan, indent=2) + '\n')
        files.append(stage / 'plan.json')
        if args.extra_capture:
            for source, name, digest in [(args.extra_capture, 'extra-capture.py', plan['extra_capture_sha256']),
                                         (ROOT / 'tools/nvidia_extra_capture.py', 'extra-wrapper.py', plan['extra_wrapper_sha256'])]:
                shutil.copyfile(source, stage / name)
                require(sha(stage / name) == digest, 'Supplemental capture changed while staging')
                files.append(stage / name)
        (stage / 'SHA256SUMS').write_text(''.join(f'{sha(f)}  {f.relative_to(stage)}\n' for f in files))
        return subprocess.run(['bash', str(ROOT / 'tools/nvidia_baseline_gpu_lease.sh'),
                               str(stage), str(args.out.resolve()), GPUS[args.gpu]], check=False).returncode


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
        raise SystemExit('REFUSED: ' + str(exc))
