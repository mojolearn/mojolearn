#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MOJOLEARN_BUILD_JOBS=1 MOJOLEARN_COMPILE_JOBS=2 MAX_JOBS=2
cores=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:4])))')
taskset -pc "$cores" $$
ROOT=/root/mojolearn-frozen
D=/root/installed-candidate
mkdir -p "$ROOT" "$D"
cd "$ROOT"
git init -q
git remote add origin https://github.com/mojolearn/mojolearn.git
git -c protocol.version=2 fetch -q --depth 1 origin eb835021dcd79a59a7e8f78c754a75db3c1fea83
git sparse-checkout set --no-cone '/*' '!bench/results' >/dev/null 2>&1
git checkout -q --detach FETCH_HEAD
[[ $(git rev-parse HEAD) = eb835021dcd79a59a7e8f78c754a75db3c1fea83 ]]
printf '%s\n' eb835021dcd79a59a7e8f78c754a75db3c1fea83 > "$D/commit.txt"
export PATH="$HOME/.pixi/bin:$PATH"
if ! command -v pixi >/dev/null; then curl -fsSL https://pixi.sh/install.sh | bash; fi
pixi install > "$D/pixi-install.log" 2>&1
rc=0
timeout -k 15 2100 bash tools/linux_wheel_candidate.sh "$D/candidate" > "$D/candidate.log" 2>&1 || rc=$?
printf '%s\n' "$rc" > "$D/original-candidate.exit"
[[ "$rc" = 1 ]] || { echo "Unexpected candidate status $rc; inspect before recovery"; exit 1; }
PY="$ROOT/.pixi/envs/gbmbench/bin/python3"
"$PY" - "$D/candidate/candidate-status.json" <<'PY'
import json,sys
j=json.load(open(sys.argv[1])); assert j['phase']=='qualify' and j['exit_code']==1,j
PY
shopt -s nullglob
wheels=("$D/candidate"/repaired/*.whl)
[[ ${#wheels[@]} = 1 ]]
W="$D/candidate/normalized/$(basename "${wheels[0]}")"
"$PY" /tmp/mojolearn-normalize-wheel.py "${wheels[0]}" "$W" > "$D/candidate/normalization.json"
S=$("$PY" - "$W" <<'PY'
import hashlib,sys
print(hashlib.file_digest(open(sys.argv[1], 'rb'), 'sha256').hexdigest())
PY
)
export MOJOLEARN_QUALIFY_PYTHON="$PY"
rc=0
timeout -k 15 600 bash tools/linux_surface_qualification.sh qualify "$W" "$S" cuda "$D/candidate/qualification-normalized" "$D/candidate/build/build-provenance.json" > "$D/candidate/qualification-normalized.log" 2>&1 || rc=$?
printf '%s\n' "$rc" > "$D/candidate/qualification-normalized.exit"
cp /tmp/mojolearn-normalize-wheel.py "$D/candidate/normalizer.py"
cp /tmp/mojolearn-nvidia-candidate-recovery.sh "$D/candidate/qualification-command.sh"
rm -rf "$D/candidate/qualification-normalized/venv"
exit "$rc"
