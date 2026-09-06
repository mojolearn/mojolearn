#!/usr/bin/env bash
set -euo pipefail
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MOJOLEARN_BUILD_JOBS=1 MOJOLEARN_COMPILE_JOBS=2 MAX_JOBS=2
cores=$(python3 -c 'import os; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:4])))')
taskset -pc "$cores" $$
ROOT=/root/mojolearn-frozen
D=/root/installed-candidate
cd "$ROOT"
# The original candidate continues unchanged. Only its redundant shorter
# timeout is removed; the on-pod 60-minute deletion watchdog remains armed.
# Bound this continuation too, leaving time to fetch before the lease expires.
while [[ ! -f "$D/candidate/candidate-status.json" ]]; do
  if [[ $(date +%s) -ge 1788690600 ]]; then
    echo '10:30 UTC candidate deadline reached; preserving partial evidence'
    kill -TERM -- -416 || true
    exit 124
  fi
  sleep 5
done
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
cp /tmp/mojolearn-nvidia-candidate-recovery.sh "$D/candidate/qualification-command-original.sh"
cp /tmp/mojolearn-nvidia-finish.sh "$D/candidate/qualification-command.sh"
rm -rf "$D/candidate/qualification-normalized/venv"
exit "$rc"
