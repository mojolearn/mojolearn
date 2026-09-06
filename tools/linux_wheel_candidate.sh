#!/usr/bin/env bash
# Main operator only, on a leased Linux GPU. Build, audit and qualify one
# complete vendor candidate. Parent owns timeout, fetch and deletion; no publish.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
DEST=${1:?fresh absolute artifact directory}
mkdir -p "$DEST"
DEST=$(cd "$DEST" && pwd)
[[ ! -e "$DEST/candidate-status.json" ]] || { echo 'Refusing reused candidate directory'; exit 2; }
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MOJOLEARN_BUILD_JOBS=1 MOJOLEARN_COMPILE_JOBS=2 MAX_JOBS=2
[[ $(uname -s) = Linux ]] || { echo 'Remote Linux GPU host required'; exit 2; }
cores=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:4])))')
taskset -pc "$cores" $$
cd "$ROOT"
export PATH="$HOME/.pixi/bin:$PATH"
pixi run -e gbmbench python3 -m venv "$DEST/tool-venv"
PY="$DEST/tool-venv/bin/python"
"$PY" -m pip install --disable-pip-version-check --only-binary=:all: auditwheel patchelf twine > "$DEST/tool-install.log" 2>&1
export PATH="$DEST/tool-venv/bin:$PATH"
export MOJOLEARN_QUALIFY_PYTHON="$PY"
BASE_PY=$(pixi run -e gbmbench python3 -c 'import sys; print(sys.executable)' | tail -1)
phase=build
finish() {
    rc=$?
    "$BASE_PY" - "$DEST" "$phase" "$rc" <<'PYEND'
import json, pathlib, sys
from datetime import datetime, timezone
out, phase, rc = sys.argv[1:]
pathlib.Path(out, 'candidate-status.json').write_text(json.dumps(dict(
    status='PASSED' if rc == '0' else 'FAILED', phase=phase, exit_code=int(rc),
    finished_utc=datetime.now(timezone.utc).isoformat(),
    scope='Single-vendor installed candidate; not publication or cross-vendor identity'), indent=2)+'\n')
PYEND
    rm -rf "$DEST/tool-venv" "$DEST/qualification/venv"
}
trap finish EXIT
bash tools/linux_surface_qualification.sh build "$DEST/build" > "$DEST/build.log" 2>&1
phase=pack
shopt -s nullglob
sets=("$DEST/build/sets/cuda" "$DEST/build/sets/hip")
args=()
for setdir in "${sets[@]}"; do [[ ! -d "$setdir" ]] || args+=(--set "$setdir"); done
[[ ${#args[@]} = 2 ]] || { echo 'Expected exactly one vendor set'; exit 2; }
vendor=$(basename "${args[1]}")
"$PY" packaging/linux/pack_wheel.py "${args[@]}" --out "$DEST/dist" > "$DEST/pack.log" 2>&1
phase=audit
"$PY" - "$DEST" "${args[1]}" <<'PYAUDIT'
import json, pathlib, re, subprocess, sys, zipfile
out, vendor = map(pathlib.Path, sys.argv[1:])
wheels = list((out / 'dist').glob('*.whl'))
assert len(wheels) == 1
wheel = wheels[0]
show = subprocess.run(['auditwheel', 'show', str(wheel)], capture_output=True, text=True)
(out / 'audit-show.log').write_text(show.stdout + show.stderr)
show.check_returncode()
tags = set(re.findall(r'manylinux_(\d+)_(\d+)_x86_64', show.stdout + show.stderr))
assert tags, 'auditwheel must determine the manylinux floor'
major, minor = max(tags, key=lambda t: tuple(map(int, t)))
platform = f'manylinux_{major}_{minor}_x86_64'
exclude = set()
for manifest in vendor.glob('*/manifest.json'):
    data = json.loads(manifest.read_text())
    exclude.update(data['driver_libs_not_staged'])
    exclude.update(row['name'] for row in data['staged_libs'])
command = ['auditwheel', 'repair', '--plat', platform, '-w', str(out / 'repaired')]
for name in sorted(exclude): command.extend(['--exclude', name])
command.append(str(wheel))
with (out / 'audit-repair.log').open('w') as log:
    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
repaired = list((out / 'repaired').glob('*.whl'))
assert len(repaired) == 1
with zipfile.ZipFile(repaired[0]) as archive:
    assert not any(n.startswith('mojolearn.libs/') for n in archive.namelist()), 'Duplicated runtime closure'
with (out / 'twine.log').open('w') as log:
    subprocess.run(['twine', 'check', str(repaired[0])], stdout=log, stderr=subprocess.STDOUT, check=True)
PYAUDIT
phase=normalize
wheels=("$DEST"/repaired/*.whl)
[[ ${#wheels[@]} = 1 ]] || exit 2
"$PY" tools/normalize_wheel_directories.py "${wheels[0]}" \
    "$DEST/normalized/$(basename "${wheels[0]}")" > "$DEST/normalization.json"
phase=qualify
wheels=("$DEST"/normalized/*.whl)
sha=$("$PY" - "${wheels[0]}" <<'PYSHA'
import hashlib,sys
print(hashlib.file_digest(open(sys.argv[1], 'rb'), 'sha256').hexdigest())
PYSHA
)
bash tools/linux_surface_qualification.sh qualify "${wheels[0]}" "$sha" "$vendor" "$DEST/qualification" "$DEST/build/build-provenance.json" > "$DEST/qualification.log" 2>&1
phase=complete
# EXIT retains wheels, raw logs and provenance and removes disposable environments.
