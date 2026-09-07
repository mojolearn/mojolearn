#!/usr/bin/env bash
# ROOT ONLY on an already rented/prepared Linux GPU. No install, rental,
# transport, benchmark, packing or publication. One actual architecture/build.
# bash tools/release061_remote_build.sh cuda sm_89 /absolute/NEW_OUT
# Required: MOJOLEARN_COMMIT(full SHA), MOJOLEARN_PYTHON(existing absolute Python),
# MOJOLEARN_RELEASE_BUILD_SECONDS(120..2400, already excludes lease fetch reserve).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
[[ $(uname -s) == Linux && $# == 3 ]] || { echo 'Remote Linux: vendor arch NEW_OUT required' >&2; exit 2; }
vendor=$1 arch=$2 OUT=$3
case "$vendor:$arch" in
    cuda:sm_89|cuda:sm_90) guard=tools/nvidia_serial_guard.py; column=nvidia ;;
    hip:gfx942) guard=tools/amd_serial_guard.py; column=amd ;;
    *) echo 'Only cuda:sm_89, cuda:sm_90 or hip:gfx942' >&2; exit 2 ;;
esac
PY=${MOJOLEARN_PYTHON:?existing absolute stdlib Python executable required}
commit=${MOJOLEARN_COMMIT:?full frozen source commit required}
seconds=${MOJOLEARN_RELEASE_BUILD_SECONDS:?remaining work seconds, excluding fetch reserve}
[[ "$PY" = /* && -x "$PY" && "$commit" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "$seconds" =~ ^[0-9]+$ ]] && ((seconds >= 120 && seconds <= 2400)) || exit 2
[[ "$OUT" = /* && ! -e "$OUT" && ! -L "$OUT" ]] || { echo 'New absolute output directory required' >&2; exit 2; }
command -v taskset >/dev/null
command -v pixi >/dev/null
command -v objdump >/dev/null
command -v patchelf >/dev/null || { echo 'Prepare patchelf before the build campaign; implicit installation refused' >&2; exit 2; }
[[ -x "$ROOT/.pixi/envs/default/bin/mojo" && -x "$ROOT/.pixi/envs/default/bin/python" ]] || {
    echo 'Prepare the locked Pixi default environment before this campaign' >&2; exit 2;
}
export MOJOLEARN_BUILD_PIXI_ENV=default MOJOLEARN_QUALIFY_PYTHON="$PY"
export MOJOLEARN_PACKAGE_BYTE_LM=1
export MOJOLEARN_GPU_ARCHS="$arch" MOJOLEARN_COMMIT="$commit"
export MOJOLEARN_TARGET_COLUMN="$column" MOJOLEARN_LINUX_CPU=x86-64-v3
unset MOJOLEARN_GPU_ARCH MOJOLEARN_VENDOR PYTHONHOME PYTHONPATH
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
export MOJOLEARN_BUILD_JOBS=1 MOJOLEARN_COMPILE_JOBS=2 MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
export MOJOLEARN_CPU_THREADS=2 CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=2
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 OMP_MAX_ACTIVE_LEVELS=1
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MAKEFLAGS=-j2 MFLAGS=-j2 GNUMAKEFLAGS=
cores=$("$PY" -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:2])))')
[[ -n "$cores" ]] || exit 2
taskset -pc "$cores" $$
"$PY" -c 'import os; assert 1 <= len(os.sched_getaffinity(0)) <= 2'
umask 077
mkdir "$OUT"
deadline=$(($(date +%s) + seconds - 30))
active_guard=
cleanup() {
    local status=$?
    trap - EXIT
    trap '' TERM HUP INT
    if [[ -n "$active_guard" ]]; then
        kill -TERM "$active_guard" 2>/dev/null || true
        wait "$active_guard" 2>/dev/null || true
    fi
    (set -o noclobber; printf '%s\n' "$status" > "$OUT/exit_code")
    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 129' HUP
trap 'exit 130' INT
: > "$OUT/results.tsv"
cp "$ROOT/tools/release061_remote_build.sh" "$OUT/campaign-source.sh"
printf '%s\n' "vendor=$vendor" "architecture=$arch" "source_commit=$commit" \
    "cpu_affinity=$cores" "work_seconds=$seconds" 'rss_gib=12' 'pixi_environment=default' \
    "kernel_column=$column" 'linux_cpu=x86-64-v3' "patchelf=$(command -v patchelf)" \
    'scope=one architecture full46 build; byte LM IDENTICAL only; no installed wheel or numerical admission' > "$OUT/campaign.txt"
run() {
    local name=$1 cap=$2 remaining status
    shift 2
    remaining=$((deadline - $(date +%s)))
    if ((remaining < 15)); then
        printf '%s\t124\tSKIPPED_DEADLINE\n' "$name" >> "$OUT/results.tsv"
        return 124
    fi
    ((cap <= remaining)) || cap=$remaining
    printf '%q ' "$PY" "$guard" --seconds "$cap" --rss-gib 12 -- "$@" > "$OUT/$name.command.txt"
    printf '\n' >> "$OUT/$name.command.txt"
    "$PY" "$guard" --seconds "$cap" --rss-gib 12 -- "$@" > "$OUT/$name.log" 2>&1 &
    active_guard=$!
    status=0
    wait "$active_guard" || status=$?
    active_guard=
    printf '%s\t%s\n' "$name" "$status" >> "$OUT/results.tsv"
    return "$status"
}

cat > "$OUT/preflight.py" <<'PYPROBE'
import importlib.util, json, os, pathlib, subprocess, sys
root, out = map(pathlib.Path, sys.argv[1:3])
vendor, arch, commit = sys.argv[3:]
sys.path.insert(0, str(root / 'tools'))
from check_linux_release_qualification import native_inventory, inventory_digest
if (root / '.git').exists():
    actual = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True, timeout=10).strip()
    if actual != commit:
        raise SystemExit('Checkout commit differs from root pin')
    # Tracked source must be clean; output evidence is retained separately.
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--'], cwd=root, timeout=15, check=True)
else:
    if (root / 'commit.txt').read_text().strip() != commit:
        raise SystemExit('Archive commit witness differs from root pin')
spec = importlib.util.spec_from_file_location('release_backend_probe', root / 'python/mojolearn/_backend.py')
backend = importlib.util.module_from_spec(spec)
spec.loader.exec_module(backend)
actual_arch, how = backend._device_arch(vendor)
if actual_arch != arch:
    raise SystemExit('Physical device does not match requested architecture: ' + repr((actual_arch, arch, how)))
inventory = native_inventory(root)
record = dict(schema='mojolearn.release061.build-preflight.v1', source_commit=commit,
              source_inventory=inventory, source_sha256=inventory_digest(inventory),
              vendor=vendor, device_architecture=actual_arch, architecture_probe=how,
              cpu_affinity=sorted(os.sched_getaffinity(0)), pixi_environment='default',
              scope='Physical device and source witness; no installed qualification')
(out / 'preflight.json').write_text(json.dumps(record, indent=2) + '\n')
PYPROBE
run resource-prefix-tests 45 "$PY" "$ROOT/tools/test_linux_surface_resource_caps.py"
run physical-source-preflight 60 "$PY" "$OUT/preflight.py" "$ROOT" "$OUT" "$vendor" "$arch" "$commit"
run full46-build 2400 bash "$ROOT/tools/linux_surface_qualification.sh" build "$OUT/build"
cat > "$OUT/postflight.py" <<'PYPOST'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
before = json.loads((out / 'preflight.json').read_text())
proof = json.loads((out / 'build/build-provenance.json').read_text())
if (proof.get('complete') is not True or proof.get('build_exit') != 0
        or proof.get('source_commit') != before['source_commit']
        or proof.get('source_inventory') != before['source_inventory']
        or proof.get('source_sha256') != before['source_sha256']):
    raise SystemExit('Build proof differs from preflight or is incomplete')
prefix = 'mojolearn/' + before['vendor'] + '/' + before['device_architecture'] + '/'
if len(proof.get('extensions', {})) != 46 or not all(p.startswith(prefix) for p in proof['extensions']):
    raise SystemExit('Built architecture differs from physical GPU witness')
byte_members = {p for p in proof['extensions'] if p.endswith('/_mojolearn_byte_lm.so')}
if byte_members != {prefix + 'identical/_mojolearn_byte_lm.so'}:
    raise SystemExit('Byte LM must exist exactly once, in IDENTICAL')
print(json.dumps(dict(status='BUILT_NOT_INSTALLED', extensions=46,
                     vendor=before['vendor'], architecture=before['device_architecture'])))
PYPOST
run build-proof-check 30 "$PY" "$OUT/postflight.py" "$OUT"
