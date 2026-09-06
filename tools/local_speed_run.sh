#!/bin/sh
# THE FAST-PATH BOARD, RUN ON THE MACHINE YOU ARE ON.
#
#   sh tools/local_speed_run.sh forest     [outdir]
#   sh tools/local_speed_run.sh classical  [outdir]
#   sh tools/local_speed_run.sh gemmseq    [outdir]
#
# It was written for this desk (the Apple third of the board) and is used
# unchanged on a rented AMD droplet by tools/do_speed_leg.sh, which is the
# reason every machine-specific fact in it -- the device name, the opponent
# interpreters, the per-arm budget -- is asked of the box rather than
# assumed. RunPod's own speed payload (tools/gemm_remote_leg.sh --speed) is
# the NVIDIA third and is NOT replaced by this file.
#
# It is the Apple transcription of the `speed` payload inside
# tools/gemm_remote_leg.sh: same drivers, same FSPEED contract, same
# one-process-per-lane rule, same build-once-run-many, so that
# tools/fast_speed_table.py reads a Mac run and a rented run with one parser.
#
# THREE THINGS DIFFER FROM THE RENTED PAYLOAD AND EACH IS DELIBERATE.
#  1. `nice 19` on every arm (the standing no-heavy-local-compute rule).
#  2. NO `timeout(1)` -- macOS has none, so the per-arm budget is a
#     background watchdog that TERMs the arm's own process group.
#  3. The opponents are whatever runs on Apple silicon: sklearn/scipy and
#     CatBoost's CPU learner, torch on MPS. cuBLAS/cuML/cuVS arms refuse by
#     name here and that refusal is recorded, never hidden.
#
# THE MODE WITNESS IS THE POINT OF THE LAST BLOCK. Every `ours` header has
# to read `mode=FAST`; one reading IDENTICAL is a correctly-labelled
# measurement of the wrong arm.
set -u
FAMILY="${1:?forest|classical|gemmseq}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date +%Y-%m-%d_%H%M%S)"
OUT="${2:-$REPO/bench/results/fast_speed/mac-$STAMP-$FAMILY}"
LOGS="$OUT/logs"
mkdir -p "$LOGS" "$OUT/dump"
cd "$REPO" || exit 1

ROUNDS="${MOJOLEARN_SPEED_ROUNDS:-3}"
SIZE="${MOJOLEARN_SPEED_SIZE:-shipped}"
ARMBUDGET="${MOJOLEARN_ARM_BUDGET:-2400}"
# WHICH INTERPRETER RUNS EACH OPPONENT. On this desk they are pixi
# environments that were solved for this project; on a rented box they are
# whatever the image has, and every opponent that is not there REFUSES BY
# NAME rather than falling back to a CPU library -- the standing rule that
# a GPU vendor's box is compared against that vendor's GPU arm or nothing.
VENDOR_PY="${MOJOLEARN_VENDOR_PY:-pixi run -e bench python3}"
TORCH_PY="${MOJOLEARN_TORCH_PY:-pixi run -e skgpu python3}"
export MOJOLEARN_SPEED_ROUNDS="$ROUNDS"
export MOJOLEARN_SPEED_SIZE="$SIZE"
# THE DEVICE NAME COMES FROM THE BOX, NOT FROM AN ASSUMPTION. This runner
# is used on this desk AND, through tools/do_speed_leg.sh, on a rented AMD
# droplet -- so the name is asked of whatever answers, in the order the
# three columns exist, and never guessed.
if [ -z "${MOJOLEARN_SPEED_DEVICE:-}" ]; then
    MOJOLEARN_SPEED_DEVICE="$(sysctl -n machdep.cpu.brand_string 2>/dev/null | tr ' ' '_')"
    [ -n "$MOJOLEARN_SPEED_DEVICE" ] || MOJOLEARN_SPEED_DEVICE="$(rocm-smi --showproductname 2>/dev/null | grep -im1 'card series' | sed 's/.*: *//' | tr ' ' '_')"
    [ -n "$MOJOLEARN_SPEED_DEVICE" ] || MOJOLEARN_SPEED_DEVICE="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '_')"
    [ -n "$MOJOLEARN_SPEED_DEVICE" ] || MOJOLEARN_SPEED_DEVICE="unknown"
fi
export MOJOLEARN_SPEED_DEVICE
export MOJOLEARN_SPEED_DUMP="$OUT/dump"
export MOJOLEARN_SPEED_DUMP_DIR="$OUT/dump"

{
  echo "family=$FAMILY"
  echo "commit=$(git rev-parse --short HEAD) parent $(git rev-parse --short HEAD^ 2>/dev/null)"
  echo "rounds=$ROUNDS"
  echo "size=$SIZE"
  echo "device=$MOJOLEARN_SPEED_DEVICE"
  echo "started=$(date -u +%FT%TZ)"
} > "$OUT/leg.txt"

say() { printf '[%s %s] %s\n' "$(date +%T)" "$FAMILY" "$*"; }

# THE PER-ARM BUDGET, WITHOUT coreutils. The arm runs in its own process
# group; the watchdog kills the GROUP, because `mojo run` and `pixi run`
# both fork and killing only the parent leaves the compile behind.
runarm() {
    _log="$1"; shift
    say "arm $_log"
    printf '\n=== %s :: %s ===\n' "$_log" "$*" >> "$OUT/console.log"
    ( nice -n 19 "$@" > "$LOGS/$_log" 2>&1 ) &
    _pid=$!
    ( sleep "$ARMBUDGET"; kill -TERM "$_pid" 2>/dev/null ) &
    _wd=$!
    wait "$_pid"; _rc=$?
    kill "$_wd" 2>/dev/null
    echo "arm_exit ${_log}=$_rc" >> "$OUT/leg.txt"
    # WHICH EXIT CODES MEAN "THE BUDGET KILLED IT". The old test was
    # `-ge 143` alone, which catches SIGTERM (143) and misses the two that
    # matter most: 137 is SIGKILL, which on these boxes is the OOM killer,
    # and 124 is coreutils `timeout` on the rented payload. Both were
    # landing in the board as ordinary empty arms.
    case "$_rc" in
        124|137|143|152) echo "FSPEED-NOTE lane=${_log} arm=- KILLED BY THE PER-ARM BUDGET (${ARMBUDGET}s), exit $_rc" >> "$LOGS/$_log" ;;
        *) [ "$_rc" -ge 143 ] 2>/dev/null && echo "FSPEED-NOTE lane=${_log} arm=- KILLED BY THE PER-ARM BUDGET (${ARMBUDGET}s), exit $_rc" >> "$LOGS/$_log" ;;
    esac
    tail -3 "$LOGS/$_log" >> "$OUT/console.log" 2>&1
    return 0
}

# WHAT THE LEG INTENDED TO RUN, WRITTEN BEFORE IT TRIES.
#
# `builtok X && runarm ...` skips the arm SILENTLY when a build failed, and
# a skipped arm leaves no log and no arm_exit line, so nothing downstream can
# tell it apart from an arm that was never part of the plan. That is the
# 2026-08-25 gemm_nt_gram case: an IDENTICAL build that had not compiled for
# three days, and three days of boards that looked complete.
#
# `expect_arm <log> build=<binary>` is the plan, recorded first and read by
# tools/leg_status.py. An arm that appears here and nowhere else is reported
# as NOT_RUN against the build it was waiting on.
expectarm() {
    echo "expect_arm $1 build=${2:--}" >> "$OUT/leg.txt"
}

# expectarm + the builtok gate + runarm, so the three cannot drift apart.
armbuilt() {
    _bin="$1"; _log="$2"; shift 2
    expectarm "$_log" "$_bin"
    if builtok "$_bin"; then
        runarm "$_log" "$@"
    else
        say "SKIP $_log -- $_bin did not build"
    fi
}

buildone() {
    _name="$1"; _src="$2"
    say "build $_name"
    nice -n 19 sh tools/with_build_lock.sh pixi run -- mojo build -I . -o "$OUT/bin_$_name" "$_src" \
        > "$LOGS/build.$_name.log" 2>&1
    echo "build_exit $_name=$?" >> "$OUT/leg.txt"
}
builtok() { [ -x "$OUT/bin_$1" ]; }

case "$FAMILY" in
gemmseq)
    LANES="${MOJOLEARN_SPEED_LANES:-gemm transformer attention mlp rmsnorm mamba selective_scan}"
    buildone gemmspeed bench/speed/gemm_speed_main.mojo
    buildone seqspeed  bench/speed/seq_speed_main.mojo
    for L in $LANES; do
        MOJOLEARN_SPEED_LANE="$L"; export MOJOLEARN_SPEED_LANE
        case "$L" in
            gemm) armbuilt gemmspeed "gemm.gemm.ours.log" "$OUT/bin_gemmspeed" ;;
            *)    armbuilt seqspeed  "seq.$L.ours.log"    "$OUT/bin_seqspeed" ;;
        esac
    done
    # the two FAST correctness gates that DEVIATION 1876 put at risk, and
    # the same two under IDENTICAL, exactly as the rented payload runs them
    expectarm "verify.transformer_block.fast.log"
    runarm "verify.transformer_block.fast.log" \
        pixi run mojo run -I . transformer/checks/transformer_check.mojo
    expectarm "verify.mamba_block.fast.log"
    runarm "verify.mamba_block.fast.log" \
        pixi run mojo run -I . mamba/checks/mamba_check.mojo
    expectarm "verify.transformer_block.identical.log"
    runarm "verify.transformer_block.identical.log" \
        sh tools/with_identical_mode.sh pixi run mojo run -I . transformer/checks/transformer_check.mojo
    expectarm "verify.mamba_block.identical.log"
    runarm "verify.mamba_block.identical.log" \
        sh tools/with_identical_mode.sh pixi run mojo run -I . mamba/checks/mamba_check.mojo
    # the vendor arms: cuBLAS refuses here by name, torch runs on MPS
    expectarm "gemm.gemm.cublas.log"
    runarm "gemm.gemm.cublas.log" $TORCH_PY tools/speed_gemm_arm.py --rounds "$ROUNDS"
    for L in $LANES; do
        [ "$L" = "gemm" ] && continue
        expectarm "seq.$L.torch.log"
        runarm "seq.$L.torch.log" $TORCH_PY tools/speed_torch_seq.py \
            --lane "$L" --rounds "$ROUNDS" --dump-dir "$MOJOLEARN_SPEED_DUMP"
    done
    ;;
classical)
    LANES="${MOJOLEARN_SPEED_LANES:-kmeans dbscan pca ols knn cd kde linkage svm metrics ivf hdbscan cholesky gmm gp krr nystroem rbfsampler resample spectral holtwinters kpss}"
    buildone classicalspeed bench/speed/classical_speed_main.mojo
    for L in $LANES; do
        MOJOLEARN_SPEED_LANE="$L"; export MOJOLEARN_SPEED_LANE
        armbuilt classicalspeed "classical.$L.ours.log" "$OUT/bin_classicalspeed"
        expectarm "classical.$L.vendor.log"
        runarm "classical.$L.vendor.log" $VENDOR_PY tools/speed_cuml_arm.py
    done
    ;;
forest)
    LANES="${MOJOLEARN_SPEED_LANES:-gbdt-symmetric gbdt-depthwise gbdt-lossguide rf et iforest}"
    DS="${MOJOLEARN_SPEED_DATASET:-}"
    ROWS="${MOJOLEARN_SPEED_ROWS:-}"
    FOREST_PY="$REPO/.pixi/envs/gbmbench/bin/python"
    [ -x "$FOREST_PY" ] || FOREST_PY="$REPO/.pixi/envs/default/bin/python"
    [ -x "$FOREST_PY" ] || FOREST_PY=python3
    echo "forest_py=$FOREST_PY" >> "$OUT/leg.txt"
    for L in $LANES; do
        _dsflag=""
        if [ -n "$DS" ] && [ "$L" != "iforest" ]; then _dsflag="--dataset $DS"; fi
        if [ -z "$ROWS" ]; then
            expectarm "forest.$L.log"
            runarm "forest.$L.log" "$FOREST_PY" bench/speed/forest_speed_arm.py --lane "$L" $_dsflag
        else
            for R in $ROWS; do
                expectarm "forest.$L.r$R.log"
                runarm "forest.$L.r$R.log" "$FOREST_PY" bench/speed/forest_speed_arm.py --lane "$L" $_dsflag --rows "$R"
            done
        fi
    done
    ;;
*) echo "family must be forest|classical|gemmseq" >&2; exit 2 ;;
esac

{
  echo "finished=$(date -u +%FT%TZ)"
  echo "fspeed_lines=$(cat "$LOGS"/*.log 2>/dev/null | grep -c '^FSPEED')"
  echo "fspeed_rounds=$(cat "$LOGS"/*.log 2>/dev/null | grep -c '^FSPEED ')"
  echo "fspeed_refused=$(cat "$LOGS"/*.log 2>/dev/null | grep -c '^FSPEED-REFUSED')"
  echo "ours_headers_fast=$(grep -h '^FSPEED-HEADER' "$LOGS"/*.log 2>/dev/null | grep -c 'arm=ours mode=FAST')"
  echo "ours_headers_identical=$(grep -h '^FSPEED-HEADER' "$LOGS"/*.log 2>/dev/null | grep -c 'arm=ours mode=IDENTICAL')"
} >> "$OUT/leg.txt"

# THE LEG NOW JUDGES ITSELF, AND ITS EXIT CODE SAYS WHAT IT FOUND.
#
# Until 2026-08-30 this script exited 0 whatever happened -- `runarm` ends in
# `return 0` by design, so one arm's failure could not take the other five
# down with it, which is right. What was missing is the accounting AFTER the
# last arm. A leg that lost six of its arms and one that lost none were
# indistinguishable to every caller, including a human reading the tail.
#
#   0  every arm measured or refused for a reason we meant
#   1  OUR side broke, or a failure nobody has classified yet
#   3  the BOX broke (budget, OOM, driver, lease, network), or a row is
#      weaker than it looks
#
# 3 rather than 1 for the box, so `tools/do_speed_leg.sh` and a human can
# tell "re-run this on another droplet" from "fix the code first" without
# reading a log. Neither is 0: an incomplete board is never a pass.
python3 "$REPO/tools/leg_status.py" "$OUT" --out "$OUT/STATUS.md" \
    --json "$OUT/status.json"
_status=$?
echo "leg_status=$_status" >> "$OUT/leg.txt"
say "done -- $OUT (status $_status; see STATUS.md)"
exit "$_status"
