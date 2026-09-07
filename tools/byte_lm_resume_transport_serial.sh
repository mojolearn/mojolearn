#!/usr/bin/env bash
# Transport-only bootstrap: pinned runtime, NumPy and retained binding; no build.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
OUT=${1:?new setup output required}
seconds=${2:?remaining deadline seconds required}
[[ ! -e "$OUT" && ! -L "$OUT" && "$seconds" =~ ^[0-9]+$ ]] || exit 2
((seconds >= 180 && seconds <= 2700)) || exit 2
mkdir "$OUT"
deadline=$(($(date +%s) + seconds - 30))
vendor=${MOJOLEARN_BYTE_LM_EXPECT_VENDOR:?}
case "$vendor" in cuda) guard=tools/nvidia_serial_guard.py ;; hip) guard=tools/amd_serial_guard.py ;; *) exit 2 ;; esac
PY=${MOJOLEARN_PYTHON:?}
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 NUMEXPR_NUM_THREADS=2
export MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2 MOJOLEARN_COMPILE_JOBS=2
active=
cleanup() {
    status=$?
    trap - EXIT
    trap '' TERM HUP INT
    if [[ -n "$active" ]]; then kill -TERM "$active" 2>/dev/null || true; wait "$active" 2>/dev/null || true; fi
    printf '%s\n' "$status" > "$OUT/exit_code"
    exit "$status"
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 129' HUP
trap 'exit 130' INT
: > "$OUT/results.tsv"
run() {
    local name=$1 cap=$2 remaining status=0
    shift 2
    remaining=$((deadline - $(date +%s)))
    ((remaining >= 15)) || return 124
    ((cap <= remaining)) || cap=$remaining
    printf '%q ' "$PY" "$guard" --seconds "$cap" --rss-gib 12 -- "$@" > "$OUT/$name.command.txt"
    "$PY" "$guard" --seconds "$cap" --rss-gib 12 -- "$@" > "$OUT/$name.log" 2>&1 &
    active=$!
    wait "$active" || status=$?
    active=
    printf '%s\t%s\n' "$name" "$status" >> "$OUT/results.tsv"
    return "$status"
}
VENV=/root/byte-lm-resume-venv
[[ ! -e "$VENV" && ! -L "$VENV" ]] || exit 2
# The frozen resume driver requires Linux memfd support. Some Conda builds
# omit os.memfd_create despite running on Linux. Fail before capture, and use
# Ubuntu's system interpreter explicitly on the HIP image when requested.
run venv 300 bash -c '
set -euo pipefail
py=$1 destination=$2 action=$3
if [ "$action" = resume128 ]; then
    "$py" -c "import os, fcntl; assert hasattr(os, \"memfd_create\") and hasattr(os, \"MFD_ALLOW_SEALING\") and hasattr(fcntl, \"F_ADD_SEALS\"), \"frozen resume requires a Python with Linux memfd/seal APIs\""
fi
if [ "$py" = /usr/bin/python3 ] && ! "$py" -c "import ensurepip" >/dev/null 2>&1; then
    export DEBIAN_FRONTEND=noninteractive
    apt-get -o Acquire::Retries=1 -o Acquire::http::Timeout=30 update
    apt-get -o Acquire::Retries=1 -o Acquire::http::Timeout=30 install -y --no-install-recommends python3-venv
fi
exec "$py" -m venv "$destination"
' byte-lm-venv "$PY" "$VENV" "$MOJOLEARN_BYTE_LM_RESUME_ACTION"
PY="$VENV/bin/python"
export MOJOLEARN_PYTHON="$PY"
run numpy 180 "$PY" -m pip install --disable-pip-version-check --no-input --only-binary=:all: numpy==1.26.4
run binding 60 "$PY" tools/byte_lm_handoff_transport.py install /root/byte-lm-handoffs/baseline \
    --sha256 "$MOJOLEARN_BYTE_LM_BASELINE_HANDOFF_SHA256" --vendor "$vendor" --kind baseline128 \
    --output "$ROOT/python/mojolearn/identical/_mojolearn_byte_lm.so"
remaining=$((deadline - $(date +%s)))
((remaining >= 120)) || exit 124
((remaining <= 2400)) || remaining=2400
export MOJOLEARN_BYTE_LM_RESUME_SECONDS=$remaining
args=("$MOJOLEARN_BYTE_LM_RESUME_ACTION" /root/byte-lm-handoffs/baseline "${OUT%/*}/byte-lm-resume")
if [[ "$MOJOLEARN_BYTE_LM_RESUME_ACTION" == resume128 ]]; then args+=(/root/byte-lm-handoffs/foreign); fi
# The helper owns the serial guard for every model job; do not nest GPU locks.
bash tools/byte_lm_resume_compact_serial.sh "${args[@]}" > "$OUT/compact-console.log" 2>&1 &
active=$!
status=0
wait "$active" || status=$?
active=
printf 'compact\t%s\n' "$status" >> "$OUT/results.tsv"
exit "$status"
