#!/bin/sh
# tools/single_device_gaps_nvidia_leg.sh: the on-box body
# (MOJOLEARN_GEMM_LEG_EXTRA for tools/gemm_remote_leg.sh nvidia) of the NVIDIA
# identity column that closes the SEVEN SINGLE-DEVICE GAPS left open on
# 2026-09-19 by tools/gpu_class_gaps_nvidia_leg.sh and
# tools/gpu_class_gaps_amd_leg.sh.
#
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/single_device_gaps_nvidia_leg.sh \
#   sh tools/gemm_remote_leg.sh nvidia --rent --allow-concurrent \
#      --local-card <an apple.card from a previous leg>
#
# THE SEVEN LANES, AND WHY THEY ARE ONE COLUMN AND NOT TWO PHASES.
# docs/VERIFICATION_MATRIX.md on 2026-09-19 (commit 26328d42c):
#
#   gp-normalize-y, gp-sample-y-normalize   amd + apple, missing nvidia
#   ordered-gradient-sum                    amd alone
#   arima-exog, arima-exog-seasonal         NO GPU column at all
#   gbdt-adapter-score-weighted             NO GPU column at all
#   rf-score-weighted                       NO GPU column at all
#
# Seven lanes at nine fixtures is 63 cells. The two legs above split their
# work into phases because each phase paid for its own binding family; this
# one builds EVERY family before it runs anything, so there is nothing to
# split and one column is the whole deliverable.
#
# WHY EVERY FAMILY, WHEN SEVEN LANES NEED SIX OR SEVEN OF THEM.
# BOTH LEGS ON 2026-09-19 SHIPPED AN INCOMPLETE BUILD LIST AND EACH LOST
# LANES TO IT, in the same hour, for the same reason:
#
#   * the NVIDIA leg's list omitted `build_preprocessing`, so
#     gp-normalize-y and gp-sample-y-normalize recorded 18 cells reading
#     "ImportError: mojolearn: numeric_mode='identical' needs
#     python/mojolearn/identical/_mojolearn_preprocessing.so, which is not
#     built" -- GaussianProcessRegressor(normalize_y=True) centres y with
#     StandardScaler's pinned folds, which is the preprocessing family, and
#     nothing in the lane's NAME says so.
#   * the AMD leg's list omitted `build_metrics`, so
#     gbdt-adapter-score-weighted recorded 9 cells with the same shape of
#     refusal -- its `score()` is weighted accuracy and R2, which is the
#     metrics family, and again nothing in the lane's name says so.
#
# A hand-written family list is a SECOND ANSWER to "what does this lane
# bind", kept in a file no lane body imports, and it rots the moment a lane
# reaches one line further. The measured cost of building all of them
# instead is 938 s on an H100 (the full 23-family set, timed on
# 2026-09-15_134552-nvidia-h100-batch2/remote/identity/status.tsv) against
# 605 s for the eleven the NVIDIA leg picked. THREE HUNDRED SECONDS BUYS
# THE WHOLE CLASS OF MISTAKE AWAY, inside a lease with 35 minutes spare.
# So the loop below is DERIVED FROM bindings/build*.sh, not typed out, and
# the PRIORITY list at the top is an ordering hint only: everything gets
# built either way, and a lane that starts binding a new family next week
# still finds it here.
#
# AND THEN THE BACKSTOP ANYWAY. After the column runs, this file GREPS ITS
# OWN JSON for the refusal text's `bindings/build_*.sh`, builds whatever it
# names, and runs the column again over the same path. If the reasoning
# above is wrong in a way nobody has thought of, the leg corrects itself
# inside the lease instead of reporting the gap a third time.
#
# THE HOST MATH LIBRARY, OR NOTHING IMPORTS. See the phase itself, below:
# this is the defect that cost both of 2026-09-19's first pods their entire
# run, and it is not a GPU, a vendor or a lane-list problem.
#
# THE COLUMN MUST BE ADMISSIBLE OR IT IS NOT EVIDENCE.
# python/mojolearn/_verify_reference.admit wants identical mode, a 40-hex
# commit, the default fixture set at the default size, one device,
# heldout_seed 1, no `*_sabotage` flag, and a PATH carrying none of
# "partial", "probe", "unfixed", "post-merge-smoke" and no basename carrying
# "sabotage". Hence: no --fixtures cap, no MOJOLEARN_IDENTITY_*_SABOTAGE
# anywhere in this file, the commit witness taken from the runner's own
# leg.txt rather than guessed, and the deliverable named
# `<label>.single-device-gaps.json` -- which is also why the import check
# below is called `import_check` and not `import_probe`: "probe" is an
# EXCLUDED PATH TOKEN and a stray file of that name next to the column is
# one rename away from refusing it.
#
# THE IMPORT CHECK RUNS AFTER THE BUILDS, NOT BEFORE. Both legs on
# 2026-09-19 placed theirs before the bindings build, where `import
# mojolearn` under MOJOLEARN_NUMERIC_MODE=identical CANNOT succeed -- it
# refuses with "no identical binary exists under .../identical" until at
# least one is built. A check that cannot pass is not a check.
#
# ONE DEVICE, ASSERTED AND PRINTED. `admit` refuses a column with
# par_devices set, and a second visible GPU would make every cell below a
# different claim than the one this column is recorded as.
#
# EVERY PHASE IS BOUNDED, AND THE BOUND SHRINKS AS THE LEASE BURNS. The
# runner polls for a completion sentinel and fetches NOTHING from a body
# that never finishes, so a slow compile must cost a later phase its ceiling
# rather than cost the whole leg its fetch.
#
# POSIX sh: the pod's /bin/sh is dash. `set -u` and deliberately NOT `set -e`:
# a phase that fails is a FINDING and its log must come home.
set -u
cd /root/mojolearn || exit 9
OUT=/root/gemm_leg_out/single-device-gaps
mkdir -p "$OUT/logs"
G="$OUT/gate.txt"
JOBS="${MOJOLEARN_COMPILE_JOBS:-8}"

# The deliverable's lanes. Seven, in one column.
LANES=arima-exog,arima-exog-seasonal,gbdt-adapter-score-weighted,gp-normalize-y,gp-sample-y-normalize,ordered-gradient-sum,rf-score-weighted

# Ordering hint only; the loop below builds every GPU family regardless.
# These are the families the seven lanes are BELIEVED to bind, so that a
# budget that runs out late still leaves a usable column:
#   build                 the shared kernels and fixtures every lane reaches
#   build_gp              GaussianProcessRegressor, both gp lanes
#   build_preprocessing   StandardScaler's folds, normalize_y
#   build_linalg          the Cholesky the gp posterior factors through
#   build_gbdt            gbdt-adapter-score-weighted's fit
#   build_rf/build_trees  rf-score-weighted's fit
#   build_metrics         both score-weighted lanes' accuracy and R2
#   build_arima/build_tsa/build_solver  the two arima-exog lanes
#   build_training        ordered_sum_gradients
PRIORITY="build build_gp build_preprocessing build_linalg build_gbdt build_rf build_trees build_metrics build_arima build_tsa build_solver build_training"

T0=$(date +%s)
BUDGET="${MOJOLEARN_NVIDIA_LEG_BUDGET:-2500}"
# The runner's own poll bound for a 60-minute lease is 3240 s and the device
# check and card ahead of this body cost about 100 s of it. 2500 s leaves the
# runner margin to poll, fetch and terminate after this body writes its
# sentinel.
cap() {
    _want=$1
    _left=$(( T0 + BUDGET - $(date +%s) ))
    [ "$_left" -lt 60 ] && _left=60
    if [ "$_want" -gt "$_left" ]; then echo "$_left"; else echo "$_want"; fi
}
say() { echo "$@" >> "$G"; }
run() {
    _n=$1; shift
    _t0=$(date +%s)
    "$@" > "$OUT/logs/$_n.log" 2>&1
    _e=$?
    echo "$_n	$_e	$(( $(date +%s) - _t0 ))" >> "$OUT/status.tsv"
    return "$_e"
}
say "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
say "budget_seconds=$BUDGET"
say "lanes=$LANES"

# THE COMMIT WITNESS, OR identity_break REFUSES TO WRITE A JSON. The box has
# no .git (the runner ships `git archive` at a pinned sha) and the gemm
# payload writes no commit.txt for the extra body. The runner DOES record the
# commit in /root/gemm_leg_out/leg.txt. Take it from there, and never guess.
if [ ! -s /root/mojolearn/commit.txt ]; then
    _c=$(sed -n 's/^commit=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -1 | awk '{print $1}')
    case "$_c" in
        [0-9a-f][0-9a-f]*) printf '%s\n' "$_c" > /root/mojolearn/commit.txt ;;
        *) say "NO COMMIT WITNESS: leg.txt gave '$_c'; the identity phase will refuse" ;;
    esac
fi
say "commit=$(head -1 /root/mojolearn/commit.txt 2>/dev/null)"

# ------------------------------------------------------------ the box itself
nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader > "$OUT/logs/device.txt" 2>&1
say "device=$(tr '\n' ' ' < "$OUT/logs/device.txt")"
say "visible_gpus=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')"
if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
    _cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
    case "$_cc" in
        9.0)  MOJOLEARN_GPU_ARCHS=sm_90a ;;
        8.9)  MOJOLEARN_GPU_ARCHS=sm_89 ;;
        8.6)  MOJOLEARN_GPU_ARCHS=sm_86 ;;
        8.0)  MOJOLEARN_GPU_ARCHS=sm_80 ;;
        12.0) MOJOLEARN_GPU_ARCHS=sm_120a ;;
        *)    MOJOLEARN_GPU_ARCHS="sm_$(echo "$_cc" | tr -d .)" ;;
    esac
    export MOJOLEARN_GPU_ARCHS
fi
say "gpu_archs=${MOJOLEARN_GPU_ARCHS:-unset}"
# device_class() in python/mojolearn/_verify_reference.py reads the class out
# of this string; it must contain "nvidia" and it does, first.
LABEL="nvidia-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr 'A-Z ' 'a-z-' | tr -cd 'a-z0-9-' | cut -c1-24)-${MOJOLEARN_GPU_ARCHS}"
say "vendor_label=$LABEL"
JSON="$OUT/$LABEL.single-device-gaps.json"

build() {
    run "$1" timeout "$(cap 420)" env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 \
        MOJOLEARN_COMPILE_JOBS="$JOBS" sh "bindings/$1.sh"
}

# ------------------------------------------- THE HOST MATH LIBRARY, OR NOTHING IMPORTS
# MEASURED TWICE ON 2026-09-19, on pod 9laka4vs9h2zli (NVIDIA) and pod
# f7qt7xpx8f6raq (AMD): the identity phase died ONE SECOND in, before any GPU
# work, at
#
#   OSError: python/mojolearn/.libs/libMojolearnMath.so: cannot open shared
#   object file: No such file or directory
#
# with every binding built and NOT ONE CELL recorded.
#
# `python/mojolearn/_portable_math.py` dlopens that library, and
# `_training_impl.py:1844`'s `def kaiming_uniform(self, shape, fan_in,
# a=math.sqrt(5.0))` evaluates it as a DEFAULT ARGUMENT at class definition
# time, so `import mojolearn` needs it unconditionally. It is not lazy and no
# lane can avoid it. Nothing under bindings/ builds it, `python/mojolearn/.libs/`
# is gitignored so `git archive` ships nothing, and the only thing in the tree
# that compiles it is `packaging/macos/build_release_wheel.sh`, which does not
# run on Linux. A developer Mac has it sitting in the checkout from some past
# wheel build and never notices; a freshly rented box cannot import the
# package at all.
#
# This calls the tree's OWN recipe, `packaging/portable_math/stage.py`'s
# build(), rather than retyping its compiler flags here -- those flags
# (-ffp-contract=off, -fno-fast-math, -march=x86-64-v3, -nostdlib) are the
# arithmetic contract, and a second copy of them is a second answer to it.
run portable_math timeout "$(cap 300)" env PYTHONPATH=/root/mojolearn/packaging/portable_math \
    pixi run python -c "import pathlib, stage; stage.build(pathlib.Path('/root/mojolearn/python/mojolearn/.libs/libMojolearnMath.so'))"
say "portable_math_exit=$(awk -F'	' '$1=="portable_math"{print $2}' "$OUT/status.tsv")"
ls -l python/mojolearn/.libs/ >> "$G" 2>&1

# --------------------------------------------------------------- every GPU family
# PRIORITY first, then the rest DERIVED from the tree. `bindings/build*.sh`
# minus `build_host_family.sh` (the parameterized builder the shims exec, not
# a build by itself) and minus every `build_*_host.sh` (a CPU build that
# REFUSES MOJOLEARN_GPU_ARCHS, which this file exports; the seven lanes take
# the device route on a GPU column, so none of them is needed here).
built=0; failed=""
for b in $PRIORITY; do
    [ -f "bindings/$b.sh" ] || { say "PRIORITY names a build that does not exist: $b"; continue; }
    if build "$b"; then built=$(( built + 1 )); else failed="$failed $b"; fi
done
for s in bindings/build*.sh; do
    n=$(basename "$s" .sh)
    [ "$n" = build_host_family ] && continue
    case "$n" in build_*_host) continue ;; esac
    case " $PRIORITY " in *" $n "*) continue ;; esac
    if build "$n"; then built=$(( built + 1 )); else failed="$failed $n"; fi
done
say "gpu_families_built=$built failed=${failed:-none}"
sha256sum python/mojolearn/identical/*.so >> "$G" 2>/dev/null

# THE IMPORT CHECK, AFTER THE BUILDS. Under identical mode this refuses
# outright until at least one binding exists, which is why both of
# 2026-09-19's legs got a guaranteed failure out of the same two lines.
run import_check timeout "$(cap 300)" env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
    pixi run python -c "import mojolearn; print('IMPORT_OK', mojolearn.vendor(), mojolearn.__version__)"
say "import_check=$(tail -1 "$OUT/logs/import_check.log" 2>/dev/null)"

# ------------------------------------------------------------------ the deliverable
# Default fixtures, default size, two repeats in one process, one device. No
# sabotage switch is set anywhere in this run.
column() {
    run "$1" timeout "$(cap 1200)" env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
        pixi run python tools/identity_break.py --lanes "$LANES" --repeats 2 \
        --vendor "$LABEL" --json "$JSON"
    say "$1_exit=$(awk -F'	' -v n="$1" '$1==n{print $2}' "$OUT/status.tsv")"
    grep -E '^cells=|MOVED|DIVERGENT|REFUSED|RELOAD' "$OUT/logs/$1.log" | head -80 >> "$G"
    cp "$OUT/logs/$1.log" "$OUT/$1.log" 2>/dev/null
}
column column
say "elapsed_after_column=$(( $(date +%s) - T0 ))"

# ------------------------------------------------------- THE BACKSTOP, AND WHY
# The refusal these seven lanes exist to escape NAMES ITS OWN CURE, verbatim,
# in the cell: "Build it with MOJOLEARN_NUMERIC_MODE=identical bash
# bindings/build_preprocessing.sh". Read it back out of the JSON rather than
# out of a human's reading of a lane body: if the loop above missed a family
# for a reason nobody has thought of, this fixes it inside the lease instead
# of reporting the same gap a third time. Normally it finds nothing and says
# so. grep -o, not sed: the whole JSON can be one line and sed would keep
# only the last match on it.
MISSING=$(grep -o 'bindings/build[a-z_]*\.sh' "$JSON" 2>/dev/null | sed 's#bindings/##; s#\.sh$##' | sort -u | tr '\n' ' ')
say "backstop_missing_families=${MISSING:-none}"
if [ -n "$MISSING" ]; then
    for b in $MISSING; do
        [ -f "bindings/$b.sh" ] || { say "backstop: no such build script: $b"; continue; }
        run "backstop-$b" timeout "$(cap 420)" env MOJOLEARN_NUMERIC_MODE=identical \
            MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS="$JOBS" sh "bindings/$b.sh"
    done
    sha256sum python/mojolearn/identical/*.so >> "$G" 2>/dev/null
    column column-after-backstop
fi

# THE CROSS-COLUMN DIFF IS NOT RUN HERE. `git archive` ships the source only,
# so bench/results/ does not exist on this box and the Apple and AMD columns
# are not here to diff against. That diff costs nothing at home, against the
# fetched column, and that is where it runs -- and where a DIVERGENT cell gets
# re-run SOLO on this same pod before anyone calls it real.

# --------------------------------------------------------------- bring it home
cat "$OUT/status.tsv" >> "$G" 2>/dev/null
ls -l "$OUT" >> "$G" 2>/dev/null
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
