#!/usr/bin/env bash
# Main-lane only. Build a COMPLETE vendor set, or qualify an already packed,
# repaired wheel with COMPLETE advertised vendor sets. Never publishes or rents.
#
# build: tools/linux_surface_qualification.sh build /absolute/artifacts
# build-tier: tools/linux_surface_qualification.sh build-tier OUT fast|deterministic|identical
# qualify: tools/linux_surface_qualification.sh qualify WHEEL SHA256 cuda|hip OUT BUILD_PROVENANCE_JSON
# Parent must impose the lease/work timeout and retain its fetch reserve.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ACTION=${1:?build or qualify}
shift
PY=${MOJOLEARN_QUALIFY_PYTHON:-python3}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2 MOJOLEARN_CPU_THREADS=2
cores=$("$PY" -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:2])))')
taskset -pc "$cores" $$
if [[ "$ACTION" = build || "$ACTION" = build-tier ]]; then
    DEST=${1:?artifact directory}
    mkdir -p "$DEST"
    DEST=$(cd "$DEST" && pwd)
    cd "$ROOT"
    # build_sets only distinguishes <=1 versus >1; setting2 launches THREE
    # tier jobs. Use1 for one compiler at a time, all45 outputs required.
    export MOJOLEARN_BUILD_JOBS=1
    unset MOJOLEARN_BUILD_SCRIPTS MOJOLEARN_BUILD_TIERS
    if [[ "$ACTION" = build-tier ]]; then
        tier=${2:?tier}
        case "$tier" in fast|deterministic|identical) ;; *) exit 2 ;; esac
        export MOJOLEARN_BUILD_TIERS="$tier"
        echo 'PARTIAL SINGLE TIER: retain for assembly; never publish or claim installed-wheel qualification' > "$DEST/PARTIAL.txt"
    fi
    commit=${MOJOLEARN_COMMIT:-}
    if [[ -z "$commit" && -s "$ROOT/commit.txt" ]]; then read -r commit < "$ROOT/commit.txt"; fi
    if [[ -z "$commit" ]]; then commit=$(git rev-parse HEAD); fi
    [[ "$commit" =~ ^[0-9a-f]{40}$ ]] || { echo 'Full source commit SHA required'; exit 2; }
    {
        echo "source_commit=$commit"
        echo 'expected_bindings_per_tier=15'
        echo "expected_tiers=${MOJOLEARN_BUILD_TIERS:-fast deterministic identical}"
        echo 'build_jobs=1 cpu_affinity=2'
        echo "artifact=$ACTION vendor set; not a wheel or publication"
    } > "$DEST/source-provenance.txt"
    command -v objdump >/dev/null || { echo 'objdump required for CPU ISA qualification'; exit 2; }
    "$PY" - "$ROOT" "$DEST" <<'PYPROVENANCE'
import hashlib, json, os, pathlib, sys
root, out = map(pathlib.Path, sys.argv[1:])
files = []
for directory, dirs, names in os.walk(root):
    dirs[:] = sorted(d for d in dirs if d not in ('.git', '.pixi', '.venv', '__pycache__', 'results', 'dist', 'archive', 'upstream') and not d.startswith('.'))
    for name in sorted(names):
        path = pathlib.Path(directory) / name
        rel = path.relative_to(root).as_posix()
        if (name.endswith('.mojo') or rel.startswith(('bindings/', 'packaging/linux/', 'python/mojolearn/')) and name.endswith(('.py', '.sh'))
                or rel in ('pixi.toml', 'pixi.lock', 'tools/linux_surface_qualification.sh')):
            files.append([rel, hashlib.sha256(path.read_bytes()).hexdigest()])
files.sort()
(out / 'source-inventory.json').write_text(json.dumps(files, indent=2) + '\n')
PYPROVENANCE
    status=0
    bash packaging/linux/build_sets.sh "$DEST" || status=$?
    "$PY" - "$ROOT" "$DEST" "$status" "$ACTION" <<'PYBUILT'
import hashlib, json, pathlib, sys
root, out = map(pathlib.Path, sys.argv[1:3])
status, action = int(sys.argv[3]), sys.argv[4]
files = json.loads((out / 'source-inventory.json').read_text())
assert all(hashlib.sha256((root / p).read_bytes()).hexdigest() == h for p, h in files), 'Sources changed during build'
outputs = {}
for path in sorted((out / 'sets').rglob('_mojolearn*.so')):
    outputs['mojolearn/' + path.relative_to(out / 'sets').as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
commit = next(line.split('=', 1)[1] for line in (out / 'source-provenance.txt').read_text().splitlines() if line.startswith('source_commit='))
record = dict(schema='mojolearn.linux.build-provenance.v1', source_commit=commit, action=action, build_exit=status,
              source_inventory=files, source_sha256=hashlib.sha256(json.dumps(files, separators=(',', ':')).encode()).hexdigest(),
              extensions=outputs, complete=(status == 0 and action == 'build' and len(outputs) == 45))
(out / 'build-provenance.json').write_text(json.dumps(record, indent=2) + '\n')
assert status == 0, 'Build/staging failed; retained provenance is not admissible'
PYBUILT
    exit "$status"
fi
[[ "$ACTION" = qualify ]] || { echo 'action must be build or qualify'; exit 2; }
WHEEL=${1:?wheel} EXPECTED=${2:?sha256} VENDOR=${3:?cuda or hip} DEST=${4:?artifact directory}
PROVENANCE=${5:?build-provenance.json from the complete vendor build}
[[ "$VENDOR" = cuda || "$VENDOR" = hip ]] || exit 2
WHEEL=$(cd "$(dirname "$WHEEL")" && pwd)/$(basename "$WHEEL")
mkdir -p "$DEST"
DEST=$(cd "$DEST" && pwd)
[[ ! -e "$DEST/venv" ]] || { echo 'Refusing reused qualification venv'; exit 2; }

# Do not call a three-extension probe a complete vendor artifact.
# Audit every RECORD hash and require each embedded architecture to have all
# 15 extension names in all three modes. A CUDA-only/HIP-only wheel is
# admitted for that vendor and labelled explicitly; no universal claim.
"$PY" - "$WHEEL" "$EXPECTED" "$ROOT" "$VENDOR" "$PROVENANCE" > "$DEST/wheel-audit.json" <<'PYAUDIT'
import base64, csv, hashlib, io, json, pathlib, re, sys, zipfile
wheel, expected, root, required_vendor, provenance = sys.argv[1:]
wheel, root = pathlib.Path(wheel), pathlib.Path(root)
assert re.fullmatch(r'[0-9a-f]{64}', expected), 'Invalid expected SHA256'
assert hashlib.sha256(wheel.read_bytes()).hexdigest() == expected, 'Wheel SHA mismatch'
assert 'manylinux_' in wheel.name, 'Require repaired manylinux wheel, not raw linux_x86_64'
names = {'_mojolearn', '_mojolearn_gbdt', '_mojolearn_estimators', '_mojolearn_rf',
         '_mojolearn_trees', '_mojolearn_svm', '_mojolearn_solver', '_mojolearn_metrics',
         '_mojolearn_tsa', '_mojolearn_linalg', '_mojolearn_arima', '_mojolearn_training',
         '_mojolearn_gp', '_mojolearn_mamba', '_mojolearn_transformer'}
sets = {}
proof_path = pathlib.Path(provenance)
proof = json.loads(proof_path.read_text())
assert proof.get('schema') == 'mojolearn.linux.build-provenance.v1' and proof.get('complete') is True
assert proof.get('build_exit') == 0 and proof.get('action') == 'build'
assert re.fullmatch(r'[0-9a-f]{40}', proof.get('source_commit', '')), 'Missing source commit'
inventory = proof['source_inventory']
assert proof['source_sha256'] == hashlib.sha256(json.dumps(inventory, separators=(',', ':')).encode()).hexdigest()
assert len(inventory) == len({p for p, h in inventory}) and inventory, 'Invalid source inventory'
for p, h in inventory:
    assert not pathlib.PurePosixPath(p).is_absolute() and '..' not in pathlib.PurePosixPath(p).parts
    assert hashlib.sha256((root / p).read_bytes()).hexdigest() == h, ('Source differs from build', p)
assert len(proof['extensions']) == 45, 'Provenance is not a complete vendor set'
with zipfile.ZipFile(wheel) as z:
    paths = z.namelist()
    assert len(paths) == len(set(paths)), 'Duplicate wheel member'
    records = [p for p in paths if p.endswith('.dist-info/RECORD')]
    assert len(records) == 1
    declared = set()
    for path, hashed, size in csv.reader(io.StringIO(z.read(records[0]).decode())):
        assert path not in declared, ('Duplicate RECORD row', path)
        declared.add(path)
        if path == records[0]:
            continue
        assert hashed.startswith('sha256='), ('Missing SHA256', path)
        value = z.read(path)
        actual = base64.urlsafe_b64encode(hashlib.sha256(value).digest()).rstrip(b'=').decode()
        assert hashed == 'sha256=' + actual and int(size) == len(value), ('Bad RECORD', path)
    assert declared == set(paths), 'RECORD does not cover whole archive'
    for p in paths:
        match = re.fullmatch(r'mojolearn/(cuda|hip)/(sm_[0-9]+a?|gfx[0-9a-f]+)/(?:(deterministic|identical)/)?(_mojolearn[^/]*)\.so', p)
        if match:
            vendor, arch, mode, extension = match.groups()
            if vendor == required_vendor:
                assert p in proof['extensions'], ('Unproven extension', p)
                assert hashlib.sha256(z.read(p)).hexdigest() == proof['extensions'][p], ('Built/wheel binary differs', p)
            sets.setdefault((vendor, arch, mode or 'fast'), set()).add(extension)
        elif p.endswith('.so') and '/_mojolearn' in p:
            raise AssertionError('Unexpected extension location: ' + p)
    vendors = {v for v, _, _ in sets}
    assert required_vendor in vendors, 'Wheel lacks requested vendor'
    for vendor, arch in {(v, a) for v, a, _ in sets}:
        for mode in ('fast', 'deterministic', 'identical'):
            assert sets.get((vendor, arch, mode)) == names, ('Incomplete set', vendor, arch, mode)
    assert set(proof['extensions']) == {p for p in paths if p.startswith('mojolearn/' + required_vendor + '/') and p.endswith('.so') and '/_mojolearn' in p}, 'Provenance/wheel vendor coverage differs'
    # Current wrappers must match the source whose tests will execute.
    for source in (root / 'python/mojolearn').glob('*.py'):
        assert z.read('mojolearn/' + source.name) == source.read_bytes(), ('Stale wrapper', source.name)
print(json.dumps({'sha256': expected, 'wheel': str(wheel),
                  'advertised_vendors': sorted(vendors),
                  'qualification_vendor': required_vendor,
                  'build_provenance_sha256': hashlib.sha256(proof_path.read_bytes()).hexdigest(),
                  'source_sha256': proof['source_sha256'],
                  'sets': {'/'.join(k): len(v) for k, v in sorted(sets.items())}}, indent=2))
PYAUDIT

timeout -k 10 60 "$PY" -m venv "$DEST/venv"
VPY="$DEST/venv/bin/python"
timeout -k 10 180 "$VPY" -m pip install --disable-pip-version-check --only-binary=:all: \
    "$WHEEL" > "$DEST/install.log" 2>&1
"$VPY" -m pip freeze > "$DEST/installed-dependencies.txt"

cat > "$DEST/run_installed.py" <<'PYRUN'
import hashlib, json, os, pathlib, runpy, sys
import mojolearn
installed = pathlib.Path(mojolearn.__file__).resolve()
assert installed.is_relative_to(pathlib.Path(sys.prefix).resolve()), ('Checkout shadowing', installed)
assert 'site-packages' in installed.parts
assert mojolearn.vendor() == os.environ['MOJOLEARN_EXPECT_VENDOR']
assert mojolearn.numeric_mode() == os.environ['MOJOLEARN_NUMERIC_MODE']
print(json.dumps({'package': str(installed), 'version': mojolearn.__version__,
                  'vendor': mojolearn.vendor(), 'mode': mojolearn.numeric_mode()}), flush=True)
from mojolearn import _backend
# Older bindings expose vendor but no tier getter; report that gap explicitly.
getters = {'_mojolearn': 'mojolearn_numeric_mode', '_mojolearn_gbdt': 'gbdt_numeric_mode',
           '_mojolearn_svm': 'svm_numeric_mode', '_mojolearn_metrics': 'umap_numeric_mode',
           '_mojolearn_linalg': 'linalg_numeric_mode', '_mojolearn_arima': 'arima_numeric_mode',
           '_mojolearn_training': 'training_numeric_mode', '_mojolearn_gp': 'gp_numeric_mode',
           '_mojolearn_mamba': 'mamba_numeric_mode', '_mojolearn_transformer': 'transformer_numeric_mode'}
all_bindings = set(getters) | {'_mojolearn_estimators', '_mojolearn_rf', '_mojolearn_trees', '_mojolearn_solver', '_mojolearn_tsa'}
readback = {}
for name in sorted(all_bindings):
    binding = _backend.binding(name)
    path = pathlib.Path(binding.__file__).resolve()
    assert path.is_relative_to(installed.parent), ('Noninstalled extension', path)
    assert _backend.read_vendor(binding) == os.environ['MOJOLEARN_EXPECT_VENDOR']
    row = {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    if name in getters:
        row['mode_code'] = int(getattr(binding, getters[name])())
        assert row['mode_code'] == {'fast': 0, 'identical': 1, 'deterministic': 2}[os.environ['MOJOLEARN_NUMERIC_MODE']]
    else:
        row['mode_readback'] = 'unavailable; build provenance and functional gate required'
    readback[name] = row
print(json.dumps({'installed_bindings': readback}), flush=True)
target = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(target, run_name='__main__')
PYRUN
rc=0
: > "$DEST/results.tsv"
unset PYTHONPATH MOJOLEARN_ARIMA_GATE_QUICK MOJOLEARN_ARIMA_GATE_N_OBS
export PYTHONNOUSERSITE=1 MOJOLEARN_EXPECT_VENDOR="$VENDOR" MOJOLEARN_REPO="$ROOT"
cd "$DEST"
for mode in fast deterministic identical; do
    export MOJOLEARN_NUMERIC_MODE="$mode"
    for surface in smoke umap mamba transformer arima; do
        args=()
        if [[ "$surface" = smoke ]]; then
            test="$ROOT/packaging/linux/smoke.py"
            args=(--vendor "$VENDOR" --json "$DEST/smoke-$mode.json")
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
printf '%s\n' "$rc" > "$DEST/exit_code"
exit "$rc"
