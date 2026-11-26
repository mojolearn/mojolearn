#!/bin/bash
# tools/ops/proving_job.sh -- the installed-wheel proving run of a measurement build on a GPU box (2026-11-22): one
# venv from the staged candidate wheels, then tests/proving (CPU and GPU routes; pytest-xdist over every core), then
# tools/ops/feature_proof_job.sh on the same venv, then checks/device_context_concurrent_repro.mojo from the tree.
# Diagnostic tooling: times nothing, decides no release. Each step prints `PROVING <name> PASS|FAIL rc=<n> <s>s` and
# keeps its full output in logs/<name>.log; the job exits 1 when any step failed. The steps share the one GPU, so they
# run one after another; inside the pytest step the test files spread over every core (one worker per allocated core).
#
#   tools/nvidia_central.sh submit <lane> --cap 90 --results box-out/proving-<sha12> \
#       --env PV_WHEEL_DIR=/root/pc-in/<sha12>/nvidia tools/ops/proving_job.sh
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/../.." && pwd)"
RES="${MOJOLEARN_RESULTS_DIR:?proving_job.sh needs MOJOLEARN_RESULTS_DIR (set by tools/box_persist_run.sh)}"
mkdir -p "$RES/logs" && RES="$(cd "$RES" && pwd)"
WHEELS="${PV_WHEEL_DIR:?proving_job.sh needs PV_WHEEL_DIR (the staged candidate wheels)}"
VENV="$RES/venv"
# Workers follow the allocation, not the host: xdist's auto count read 128 host CPUs on a 14-core cgroup and every
# worker's import aborted (2026-11-22, nvc1-0002).
. "$HERE/box_cores_lib.sh"; W=$(alloc_cores)
fails=0
step() { # name, command...
    local n=$1; shift; local t0=$SECONDS
    "$@" > "$RES/logs/$n.log" 2>&1; local rc=$?
    if [ $rc = 0 ]; then echo "PROVING $n PASS rc=0 $((SECONDS - t0))s"
    else echo "PROVING $n FAIL rc=$rc $((SECONDS - t0))s: $(grep -v '^\s*$' "$RES/logs/$n.log" | tail -n 2 | tr '\n' ' ' | cut -c1-240)"; fails=$((fails + 1)); fi
}
install() {
    python3 -m venv "$VENV" && "$VENV/bin/python" -m pip install -q --upgrade pip &&
        "$VENV/bin/python" -m pip install -q "$WHEELS"/*.whl scikit-learn pandas pytest pytest-xdist
}
step install install || true
[ -x "$VENV/bin/python" ] || { echo "PROVING setup FAIL no venv"; exit 4; }
"$VENV/bin/python" -I -c "import mojolearn as ml; print('PROVING wheel', ml.__version__)" 2>&1 | tail -n 1
cd "$REPO" || exit 4
# The model-selection case shards write their logs under the pytest base temp (tests/proving/
# test_proving_model_selection.py, proving-arms/<device>/<arm>/shard<N>.log: the CASE lines and the faulthandler
# frames of a case process that died). Each pytest step gets its own base temp, and every shard log is copied into
# logs/shards/<step>/ so it leaves the box with the results (L40S 2026-11-26: four cases died and their logs stayed
# in /tmp on a box that was deleted).
BT="$(mktemp -d "${TMPDIR:-/tmp}/proving-bt.XXXXXX")"
keep_shards() {
    local n=$1 root="$BT/$1" f
    [ -d "$root" ] || return 0
    find "$root" -path '*/proving-arms/*' -name 'shard*.log' -print0 | while IFS= read -r -d '' f; do
        mkdir -p "$RES/logs/shards/$n/$(dirname "${f#"$root"/}")" && cp "$f" "$RES/logs/shards/$n/${f#"$root"/}"
    done
    echo "PROVING $n shard logs kept: $(find "$RES/logs/shards/$n" -name 'shard*.log' 2>/dev/null | wc -l | tr -d ' ')"
}
step proving "$VENV/bin/python" -m pytest tests/proving -q -n "$W" -p no:cacheprovider --basetemp "$BT/proving"
keep_shards proving
step proving_gpu "$VENV/bin/python" -m pytest tests/proving -q -m gpu -n "$W" -p no:cacheprovider --basetemp "$BT/proving_gpu"
keep_shards proving_gpu
step feature_proof env FP_VENV="$VENV" MOJOLEARN_RESULTS_DIR="$RES/feature-proof" bash tools/ops/feature_proof_job.sh
step concurrent_repro bash -c "export PATH=\$PATH:/root/.pixi/bin; pixi run -q mojo run -I . checks/device_context_concurrent_repro.mojo"
echo "PROVING done fails=$fails"
[ $fails = 0 ]
