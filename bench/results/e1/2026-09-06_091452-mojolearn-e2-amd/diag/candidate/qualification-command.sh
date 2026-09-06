#!/usr/bin/env bash
set -euo pipefail
cd /root/mojolearn
D=/root/mojolearn/bench/results/e1/2026-09-06_091452-mojolearn-e2-amd/diag/candidate
W="$D/normalized/mojolearn-0.6.0-py3-none-manylinux_2_35_x86_64.whl"
S=$(python3 - "$W" <<'PY'
import hashlib,sys
print(hashlib.file_digest(open(sys.argv[1], 'rb'), 'sha256').hexdigest())
PY
)
export MOJOLEARN_QUALIFY_PYTHON=/root/mojolearn/.pixi/envs/gbmbench/bin/python3
rc=0
timeout -k 15 1500 bash tools/linux_surface_qualification.sh qualify "$W" "$S" hip "$D/qualification-normalized" "$D/build/build-provenance.json" > "$D/qualification-normalized.log" 2>&1 || rc=$?
printf '%s\n' "$rc" > "$D/qualification-normalized.exit"
rm -rf "$D/qualification-normalized/venv"
exit "$rc"
