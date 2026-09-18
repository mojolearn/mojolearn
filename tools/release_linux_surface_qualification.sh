#!/usr/bin/env bash
# Installed release-wheel qualification; no native builds.
# The original combined build/qualification script remains frozen in build
# provenance. This successor is independently hashed in qualification-sources.
# Qualify an already packed,
# repaired wheel with COMPLETE advertised vendor sets. Never publishes or rents.
#
# tools/release_linux_surface_qualification.sh qualify-release-linux3 \
#   WHEEL SHA256 cuda|hip OUT PROOF_DIRECTORY ARCH
# Proofs: cuda-sm_89.json, cuda-sm_90a.json, hip-gfx942.json.
# Parent must impose the lease/work timeout and retain its fetch reserve.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ACTION=${1:?qualify-release-linux3}
shift
PY=${MOJOLEARN_QUALIFY_PYTHON:-python3}
[[ $(uname -s) = Linux ]] || { echo 'Qualification requires remote Linux' >&2; exit 2; }
command -v taskset >/dev/null || { echo 'taskset required for CPU cap' >&2; exit 2; }
# Fixed limits override inherited broad job settings. Child builds inherit this
# affinity; build_sets.sh selects a subset of it and cannot expand it itself.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export OMP_THREAD_LIMIT=1 OMP_MAX_ACTIVE_LEVELS=1 BLIS_NUM_THREADS=1 NUMEXPR_MAX_THREADS=1
export MOJOLEARN_COMPILE_JOBS=2 MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2 MOJOLEARN_CPU_THREADS=2
# DEVIATION 2501: the build action runs MOJOLEARN_BUILD_JOBS extension
# builds at a time (default 4) on 2 x jobs cores; each build keeps the
# two-worker, one-BLAS-thread caps above. Qualification stays serial on two.
BUILD_JOBS=1
if [[ "$ACTION" = build ]]; then BUILD_JOBS=${MOJOLEARN_BUILD_JOBS:-4}; fi
[[ "$BUILD_JOBS" =~ ^[1-9][0-9]?$ && "$BUILD_JOBS" -le 16 ]] || { echo 'MOJOLEARN_BUILD_JOBS must be 1..16' >&2; exit 2; }
BUILD_CORES=$((2 * BUILD_JOBS))
export MOJOLEARN_BUILD_JOBS=$BUILD_JOBS CARGO_BUILD_JOBS=2 RAYON_NUM_THREADS=2
export MAKEFLAGS=-j2 MFLAGS=-j2 GNUMAKEFLAGS=
cores=$("$PY" -c 'import os, sys; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:int(sys.argv[1])])))' "$BUILD_CORES")
[[ -n "$cores" ]] || { echo 'Empty CPU affinity' >&2; exit 2; }
taskset -pc "$cores" $$
"$PY" - "$cores" "$BUILD_CORES" <<'PYCAP'
import os, sys
expected = {int(cpu) for cpu in sys.argv[1].split(',')}
actual = os.sched_getaffinity(0)
if not 1 <= len(actual) <= int(sys.argv[2]) or actual != expected:
    raise SystemExit('CPU affinity cap failed')
PYCAP
# RESOURCE_CAPS_END: source-only test extracts only the admission prefix.
# DEVIATION 2290: the three-architecture action is qualify-release-linux3; the
# name it was authored under, qualify-release-0.6.1, is a deprecated alias.
if [[ "$ACTION" = qualify-release-0.6.1 ]]; then ACTION=qualify-release-linux3; fi
[[ "$ACTION" = qualify-release-linux3 ]] || { echo 'Unknown qualification action'; exit 2; }
WHEEL=${1:?wheel} EXPECTED=${2:?sha256} VENDOR=${3:?cuda or hip} DEST=${4:?artifact directory}
PROVENANCE=${5:?build-provenance.json from the complete vendor build}
[[ "$VENDOR" = cuda || "$VENDOR" = hip ]] || exit 2
WHEEL=$(cd "$(dirname "$WHEEL")" && pwd)/$(basename "$WHEEL")
mkdir -p "$DEST"
DEST=$(cd "$DEST" && pwd)
[[ -z $(ls -A "$DEST") ]] || { echo 'Refusing nonempty qualification directory'; exit 2; }
# An interrupted/preflight-failed run must never leave a successful marker.
printf '1\n' > "$DEST/exit_code"
"$PY" "$ROOT/tools/verify_linux_surface_qualification.py" snapshot "$ROOT" "$DEST"

if [[ "$ACTION" = qualify-release-linux3 ]]; then
    ARCH=${6:?actual runtime architecture sm_89, sm_90, sm_90a or gfx942}
    "$PY" - "$ROOT" "$WHEEL" "$EXPECTED" "$PROVENANCE" "$VENDOR/$ARCH" "$DEST" <<'PYMULTI'
import json, pathlib, shutil, sys
root, wheel, expected, proof_root, key, out = sys.argv[1:]
sys.path.insert(0, str(pathlib.Path(root) / 'tools'))
from check_linux_release_qualification import release_audit
audit = release_audit(wheel, root, proof_root, key)
if audit['sha256'] != expected:
    raise SystemExit('Wheel SHA mismatch')
dest = pathlib.Path(out)
(dest / 'wheel-audit.json').write_text(json.dumps(audit, indent=2) + '\n')
shutil.copyfile(pathlib.Path(proof_root) / (key.replace('/', '-') + '.json'), dest / 'build-provenance.json')
PYMULTI
fi

timeout -k 10 60 "$PY" -m venv "$DEST/venv"
VPY="$DEST/venv/bin/python"
timeout -k 10 180 "$VPY" -m pip install --disable-pip-version-check --only-binary=:all: \
    "$WHEEL" > "$DEST/install.log" 2>&1
"$VPY" -m pip check > "$DEST/dependency-check.log" 2>&1
"$VPY" -m pip freeze > "$DEST/installed-dependencies.txt"
(cd "$DEST" && env -u PYTHONPATH -u PYTHONHOME "$VPY" \
    "$ROOT/tools/check_numpy_free_runtime.py" --installed) \
    > "$DEST/numpy-free-runtime.log" 2>&1
# Only after the dependency-free installed GPU path passes, add the oracle.
timeout -k 10 180 "$VPY" -m pip install --disable-pip-version-check --only-binary=:all: \
    'numpy>=1.24' > "$DEST/oracle-install.log" 2>&1
"$VPY" -m pip freeze > "$DEST/test-dependencies.txt"

cat > "$DEST/run_installed.py" <<'PYRUN'
import hashlib, json, os, pathlib, runpy, sys
import mojolearn
audit = json.loads(pathlib.Path(os.environ['MOJOLEARN_WHEEL_AUDIT']).read_text())
installed = pathlib.Path(mojolearn.__file__).resolve()
assert installed.is_relative_to(pathlib.Path(sys.prefix).resolve()), ('Checkout shadowing', installed)
assert 'site-packages' in installed.parts
assert mojolearn.vendor() == os.environ['MOJOLEARN_EXPECT_VENDOR']
assert mojolearn.numeric_mode() == os.environ['MOJOLEARN_NUMERIC_MODE']
print(json.dumps({'package': str(installed), 'version': mojolearn.__version__,
                  'vendor': mojolearn.vendor(), 'mode': mojolearn.numeric_mode()}), flush=True)
from mojolearn import _backend
architecture = {}
# DEVIATION 2290: release_audit writes assembly_profile 'release-linux3';
# 'release-0.6.1' is the deprecated alias retained audits may still carry.
release_profile = audit.get('assembly_profile') in ('release-linux3', 'release-0.6.1')
if release_profile:
    assert not os.environ.get('MOJOLEARN_GPU_ARCH'), 'Architecture override forbidden'
    device_arch, probe = _backend._device_arch(os.environ['MOJOLEARN_EXPECT_VENDOR'])
    selected = _backend.gpu_arch()
    # DEVIATION 2293: the selected binding may be the architecture-specific
    # build for this exact device (device sm_90 -> selected sm_90a), which is
    # what _backend.py prefers. The audit's runtime_architecture is the name
    # the wheel carries, so it must equal the SELECTED one; the device must be
    # the chip that selection targets. Anything else is still refused.
    assert selected == audit['runtime_architecture'], ('Wrong actual GPU architecture', device_arch, selected)
    assert selected in (device_arch, device_arch + 'a'), ('Wrong actual GPU architecture', device_arch, selected)
    architecture = dict(device_architecture=device_arch, selected_architecture=selected,
                        architecture_probe=probe, architecture_override_absent=True)
# Older bindings expose vendor but no tier getter; report that gap explicitly.
getters = {'_mojolearn': 'mojolearn_numeric_mode', '_mojolearn_gbdt': 'gbdt_numeric_mode',
           '_mojolearn_svm': 'svm_numeric_mode', '_mojolearn_metrics': 'umap_numeric_mode',
           '_mojolearn_linalg': 'linalg_numeric_mode', '_mojolearn_arima': 'arima_numeric_mode',
           '_mojolearn_training': 'training_numeric_mode', '_mojolearn_gp': 'gp_numeric_mode',
           '_mojolearn_mamba': 'mamba_numeric_mode', '_mojolearn_transformer': 'transformer_numeric_mode'}
if release_profile and mojolearn.numeric_mode() == 'identical':  # DEVIATION 2290
    getters['_mojolearn_byte_lm'] = 'byte_lm_numeric_mode'
# Read the same complete, tier-specific inventory required by admission.
# The legacy fixed list both loaded forbidden lower-tier modules and omitted
# seven newer IDENTICAL bindings.
sys.path.insert(0, str(pathlib.Path(os.environ['MOJOLEARN_REPO']) / 'tools'))
from verify_linux_surface_qualification import expected_bindings
sys.path.pop(0)
all_bindings = expected_bindings(mojolearn.numeric_mode(), release_profile)
readback = {}
for name in sorted(all_bindings):
    binding = _backend.binding(name)
    path = pathlib.Path(binding.__file__).resolve()
    assert path.is_relative_to(installed.parent), ('Noninstalled extension', path)
    assert _backend.read_vendor(binding) == os.environ['MOJOLEARN_EXPECT_VENDOR']
    row = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    member = path.relative_to(installed.parent).as_posix()
    if architecture:
        assert member.startswith(os.environ['MOJOLEARN_EXPECT_VENDOR'] + '/' + architecture['selected_architecture'] + '/')
    assert audit['extension_hashes'].get(member) == row['sha256'], ('Installed binary differs from wheel', member)
    if name in getters:
        row['mode_code'] = int(getattr(binding, getters[name])())
        assert row['mode_code'] == {'fast': 0, 'identical': 1, 'deterministic': 2}[os.environ['MOJOLEARN_NUMERIC_MODE']]
    else:
        row['mode_readback'] = 'unavailable; build provenance and functional gate required'
    readback[name] = row
record = {'package': str(installed), 'version': mojolearn.__version__,
          'vendor': mojolearn.vendor(), 'mode': mojolearn.numeric_mode(),
          'wheel_sha256': audit['sha256'], 'installed_bindings': readback}
# DEVIATION 2680: the CPU TRAINING binding, under its own key. It cannot join
# `readback` above, whose loop asserts read_vendor() == the GPU vendor of this
# leg; this one reads back 'cpu' by design and is reached through its own path
# helper rather than _backend.binding() for the same reason.
if release_profile and mojolearn.numeric_mode() == 'identical':
    from mojolearn import _byte_lm_host
    host_path = pathlib.Path(_byte_lm_host.binary_path()).resolve()
    assert host_path.is_relative_to(installed.parent), ('Noninstalled CPU training binding', host_path)
    host_member = host_path.relative_to(installed.parent).as_posix()
    host_digest = hashlib.sha256(host_path.read_bytes()).hexdigest()
    assert audit.get('host_extension_hashes', {}).get(host_member) == host_digest, \
        ('Installed CPU training binding differs from wheel', host_member)
    host_module = _byte_lm_host._load()
    assert int(host_module.byte_lm_host_numeric_mode()) == 1, 'CPU training binding is not IDENTICAL'
    assert str(host_module.byte_lm_host_vendor()) == 'cpu', 'CPU training binding vendor read-back'
    assert callable(getattr(host_module, 'byte_lm_host_train_step', None)), \
        'CPU training binding has no train step; the wheel carries an inference-only build'
    record['installed_host_binding'] = {'path': str(host_path), 'sha256': host_digest,
                                        'numeric_mode': 1, 'vendor': 'cpu'}
    # EVERY host binding the manifest ships (the packaging lane, 2026-09-14),
    # each loaded through the package's own host loader and read back as
    # vendor cpu, IDENTICAL and the CPU column, each with the digest of the
    # installed file, which must be the audited wheel's member.
    from mojolearn import _backend, host_surface
    rows = {}
    for name in host_surface.wheel_bindings():
        path = pathlib.Path(_backend.host_module_path(name)).resolve()
        assert path.is_relative_to(installed.parent), ('Noninstalled host binding', path)
        member = path.relative_to(installed.parent).as_posix()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        assert audit.get('host_extension_hashes', {}).get(member) == digest, ('Installed host binding differs from wheel', member)
        module = _backend.load_host_module(name)
        prefix = name[len('_mojolearn_'):]
        rows[name] = {'path': str(path), 'sha256': digest,
                      'numeric_mode': int(getattr(module, prefix + '_numeric_mode')()),
                      'vendor': str(getattr(module, prefix + '_vendor')()),
                      'column': str(getattr(module, prefix + '_column')())}
        assert rows[name]['numeric_mode'] == 1 and rows[name]['vendor'] == 'cpu' and rows[name]['column'] == 'cpu', ('Host binding read-back', name, rows[name])
    record['installed_host_bindings'] = rows
record.update(architecture)
pathlib.Path(os.environ['MOJOLEARN_INSTALLED_RECORD']).write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps(record), flush=True)
target = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(target, run_name='__main__')
PYRUN
rc=0
: > "$DEST/results.tsv"
unset PYTHONPATH PYTHONHOME MOJOLEARN_VENDOR MOJOLEARN_ARIMA_GATE_QUICK MOJOLEARN_ARIMA_GATE_N_OBS
export MOJOLEARN_WHEEL_AUDIT="$DEST/wheel-audit.json"
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 MOJOLEARN_EXPECT_VENDOR="$VENDOR" MOJOLEARN_REPO="$ROOT"
cd "$DEST"
for mode in fast deterministic identical; do
    export MOJOLEARN_NUMERIC_MODE="$mode"
    # DEVIATION 2490: the job set per tier comes from the verifier's
    # expected_jobs so the on-box loop and the admission side cannot drift.
    read -r -a surfaces <<< "$("$PY" - "$mode" "$ACTION" <<'PYJOBS'
import os, sys
sys.path.insert(0, os.environ['MOJOLEARN_REPO'] + '/tools')
from verify_linux_surface_qualification import RELEASE_PROFILE, expected_jobs
mode, action = sys.argv[1:3]
audit = {'assembly_profile': RELEASE_PROFILE} if action == 'qualify-release-linux3' else {}
order = ('smoke', 'umap', 'umap-transform', 'umap-quality', 'ordered-rmse', 'mamba', 'transformer', 'arima', 'byte-lm')
print(' '.join(s for s in order if (s, mode) in expected_jobs(audit)))
PYJOBS
)"
    for surface in "${surfaces[@]}"; do
        export MOJOLEARN_INSTALLED_RECORD="$DEST/$surface-$mode.installed.json"
        args=()
        if [[ "$surface" = smoke ]]; then
            test="$ROOT/tools/release_linux_smoke.py"
            args=(--vendor "$VENDOR" --json "$DEST/smoke-$mode.json")
        elif [[ "$surface" = byte-lm ]]; then
            test="$ROOT/tools/byte_lm_installed_step.py"
        elif [[ "$surface" = ordered-rmse ]]; then
            test="$ROOT/tools/ordered_rmse_surface_check.py"
        elif [[ "$surface" = umap-transform ]]; then
            test="$ROOT/python/mojolearn/tests/test_umap_transform.py"
        elif [[ "$surface" = umap-quality ]]; then
            test="$ROOT/tools/umap_transform_quality_check.py"
            args=(--mode "$mode" --device "${MOJOLEARN_QUALIFY_DEVICE:-$VENDOR}" --profile expanded
                  --source-root "$ROOT" --output "$DEST/$surface-$mode.json")
        else
            test="$ROOT/python/mojolearn/tests/test_${surface}_surface.py"
        fi
        status=0
        timeout -k 10 "${MOJOLEARN_SURFACE_TIMEOUT:-300}" "$VPY" \
            "$DEST/run_installed.py" "$test" "${args[@]}" > "$DEST/$surface-$mode.log" 2>&1 || status=$?
        printf '%s\t%s\t%s\n' "$surface" "$mode" "$status" | tee -a "$DEST/results.tsv"
        [[ "$status" = 0 ]] || rc=1
    done
done
"$PY" "$ROOT/tools/verify_linux_surface_qualification.py" verify "$ROOT" "$DEST" || rc=1
printf '%s\n' "$rc" > "$DEST/exit_code"
exit "$rc"
