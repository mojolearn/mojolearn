#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
#
# THE NVIDIA OR AMD RELEASE COLUMN, ON THE BOX THAT JUST BUILT THE SET
# (lane/release-gpu-columns, 2026-09-22). RUNS ON THE RENTED BOX, after
# tools/release061_remote_build.sh finished, in the same lease:
#
#   bash tools/release_gpu_column.sh <cuda|hip> <release-build dir> <selection.json> <NEW_OUT> <seconds>
#
# <selection.json> was written on the Mac by
#   python3 tools/verify_lanes.py --gpu-pass <cuda|hip> --write-selection <file>
# (the lanes changed since the last finished pass on that backend, else since
# the newest v* tag, with every lane that cannot run on one GPU left out by
# name). This script runs exactly those lanes, base,denormal,odd, every cell
# fitted ONCE, from the binaries in <release-build>/build/sets/<vendor>, and
# writes the pass records (manifest.json, run-summary.json, column.json) under
# <NEW_OUT>/<cuda|hip>/. The leg fetches <NEW_OUT>; the Mac diffs the column
# against the CPU column of the same commit (tools/release_gpu_columns.py).
#
# MOJOLEARN_RELEASE_COLUMN_CPU=1 also records THIS box's CPU column for the
# same lanes (those with a CPU route) under <NEW_OUT>/cpu-box/, concurrently,
# from the set's own host bindings: a second CPU reference from the same
# rental, never a replacement for the Mac's.
#
# Exit 0 only when the GPU column is complete. Nothing here rents, fetches or
# deletes anything; the leg owns the box.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT" || exit 2
[[ $# == 5 ]] || { echo 'usage: release_gpu_column.sh <cuda|hip> <release-build dir> <selection.json> <NEW_OUT> <seconds>' >&2; exit 2; }
backend=$1 RB=$2 SEL=$3 OUT=$4 seconds=$5
case "$backend" in cuda|hip) ;; *) echo 'backend must be cuda or hip' >&2; exit 2 ;; esac
[[ "$seconds" =~ ^[0-9]+$ ]] && ((seconds >= 60)) || { echo 'seconds must be an integer >= 60' >&2; exit 2; }
[[ -d "$RB/build/sets/$backend" ]] || { echo "no $backend set under $RB/build/sets" >&2; exit 2; }
[[ -f "$SEL" ]] || { echo "no selection file $SEL" >&2; exit 2; }
[[ "$OUT" = /* && ! -e "$OUT" ]] || { echo 'NEW_OUT must be a new absolute directory' >&2; exit 2; }
PY=${MOJOLEARN_COLUMN_PYTHON:-$ROOT/.pixi/envs/default/bin/python}
[[ -x "$PY" ]] || { echo "no Python at $PY (the locked default environment the set was built with)" >&2; exit 2; }
commit=${MOJOLEARN_COMMIT:-$(cat "$ROOT/commit.txt" 2>/dev/null)}
[[ "$commit" =~ ^[0-9a-f]{40}$ ]] || { echo 'MOJOLEARN_COMMIT or commit.txt must hold the 40-hex commit' >&2; exit 2; }
export MOJOLEARN_COMMIT=$commit PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1
unset PYTHONHOME MOJOLEARN_HOST_DIR MOJOLEARN_GPU_ARCH MOJOLEARN_TARGET_COLUMN
mkdir -p "$OUT"
# One scheduler per box, outside the repository.
export MOJOLEARN_MAC_SLOT_BASE=$OUT/.slot MOJOLEARN_METAL_LOCK=$OUT/.gpu-lock MOJOLEARN_METAL_QUEUE=$OUT/.gpu-queue
ncpu=$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)
cpu_jobs=$(( ncpu / 2 )); ((cpu_jobs > 8)) && cpu_jobs=8; ((cpu_jobs < 1)) && cpu_jobs=1
export MAC_SLOTS=$(( cpu_jobs + 1 ))
started=$(date +%s)
{
  echo "backend=$backend"; echo "commit=$commit"; echo "selection_sha256=$(sha256sum "$SEL" | cut -d' ' -f1)"
  echo "release_build=$RB"; echo "seconds=$seconds"; echo "started=$(date -u +%FT%TZ)"
  "$PY" -c 'import json,sys; d=json.load(open(sys.argv[1])); print("summary=" + d.get("summary", "")); print("lanes=" + ",".join(d["lanes"]))' "$SEL"
} > "$OUT/column.txt"
cp "$SEL" "$OUT/selection.json"

cpu_pid=
if [[ "${MOJOLEARN_RELEASE_COLUMN_CPU:-0}" = 1 ]]; then
  host=$(ls -d "$RB"/build/sets/"$backend"/*/host 2>/dev/null | head -1)
  cpu_lanes=$("$PY" - "$SEL" <<'PYCPU'
import json, sys
sys.path.insert(0, "tools")
import lane_applicability
d = json.load(open(sys.argv[1]))
skip = lane_applicability.degenerate("cpu-host")
print(",".join(n for n in d["lanes"] if n not in skip))
PYCPU
)
  if [[ -n "$cpu_lanes" && -n "$host" ]]; then
    ( timeout -k 30 "$seconds" "$PY" tools/verify_lanes.py --lanes "$cpu_lanes" --backend cpu --host-dir "$host" \
        --fixtures base,denormal,odd --repeats 1 --shards "$cpu_jobs" --jobs "$cpu_jobs" \
        --budget $((seconds - 60)) --timeout $((seconds - 60)) --wait-timeout $((seconds - 60)) \
        --out "$OUT/cpu-box" > "$OUT/cpu-box.log" 2>&1; echo $? > "$OUT/cpu-box.exit" ) &
    cpu_pid=$!
    echo "cpu_box_lanes=$(tr ',' '\n' <<< "$cpu_lanes" | grep -c .)" >> "$OUT/column.txt"
  else
    echo "cpu_box=skipped (no lane with a CPU route, or no host bindings in the set)" >> "$OUT/column.txt"
  fi
fi

timeout -k 30 "$seconds" "$PY" tools/verify_lanes.py --gpu-pass "$backend" --selection "$SEL" \
    --gpu-set "$RB/build/sets/$backend" --budget $((seconds - 60)) --out "$OUT/$backend" > "$OUT/$backend.log" 2>&1
rc=$?
echo "$rc" > "$OUT/$backend.exit"
[[ -n "$cpu_pid" ]] && wait "$cpu_pid"
{
  echo "gpu_exit=$rc"
  [[ -f "$OUT/cpu-box.exit" ]] && echo "cpu_box_exit=$(cat "$OUT/cpu-box.exit")"
  echo "seconds_used=$(( $(date +%s) - started ))"
  echo "finished=$(date -u +%FT%TZ)"
} >> "$OUT/column.txt"
grep '^# verdict\|^# FAIL\|^# nothing to check\|^# lanes selected' "$OUT/$backend.log" | tail -8
exit "$rc"
