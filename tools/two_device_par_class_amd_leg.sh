#!/bin/sh
# tools/two_device_par_class_amd_leg.sh: the on-box body (MOJOLEARN_GEMM_LEG_EXTRA
# for tools/gemm_remote_leg.sh amd) of the ONE lease that can move the thirteen
# `par-*` lanes docs/VERIFICATION_MATRIX.md lists under "GPU column on fewer
# than three classes", 2026-09-19.
#
#   MOJOLEARN_GEMM_LEG_GPU_COUNT=2 \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/two_device_par_class_amd_leg.sh \
#   sh tools/gemm_remote_leg.sh amd --rent --allow-concurrent \
#      --local-card <an apple.card from a previous leg>
#
# ---------------------------------------------------------------------------
# READ THIS BEFORE YOU BELIEVE A TWO-DEVICE COLUMN CLOSES A MATRIX GAP.
#
# It does not, and it cannot, and the refusal is three lines of code:
#
#   python/mojolearn/_verify_reference.py:303
#       if str(pkg.get("par_devices") or "0") != "0":
#           return f"par_devices {pkg.get('par_devices')}"
#
# `tools/verification_matrix.py:262` builds its gpu coverage "from ADMITTED
# clean columns" and skips every column `admit` returns a reason for. So a
# column recorded with MOJOLEARN_PAR_DEVICES=0,1 is, BY CONSTRUCTION, invisible
# to the count it would appear to answer. The thirteen lanes are not waiting on
# a second GPU to be COUNTED. They are waiting on a second DEVICE CLASS.
#
# What each of the thirteen actually carries today, read out of the committed
# columns rather than out of the prose:
#
#   eleven   nvidia ALONE, all from ONE column,
#            bench/results/e1g/2026-09-17_203449-nvidia-refregen-par-lanes-b/
#            remote/identity/identity_break.nvidia-...-4090-sm_89.json
#            (par_devices=0, admitted). They need `amd` and `apple`.
#   two      par-forest-pool and par-rbf-sampler: amd AND nvidia already, from
#            bench/results/identity_break/2026-09-15_par-lanes-new/. The only
#            class they lack is `apple`, and an Apple column is a Metal run on
#            the one Mac. NOTHING ON A RENTED BOX CAN GIVE THEM A THIRD CLASS
#            and this leg does not pretend otherwise.
#
# So the deliverable that MOVES THE MATRIX is PHASE ONE below: a ONE-device AMD
# column, par_devices=0, admissible, over all thirteen. It takes the eleven
# from one class to two. It takes NOTHING to three; three needs Apple, which
# this leg cannot rent.
#
# WHY THE BOX IS STILL RENTED WITH TWO GPUs. Because the drivers' own claim --
# `identity_break._par_devices`' docstring, "A two-device column ... must hash
# equal, cell for cell, to the one-device column of the same commit; that
# equality is the drivers' whole claim" -- is only STATEABLE on two devices,
# and the pair is only worth anything when BOTH HALVES COME FROM THE SAME BUILD
# ON THE SAME BOX AT THE SAME COMMIT. Two leases cannot give you that; one can.
# PHASE TWO is that half, and PHASE DIFF holds them to each other ON THE BOX,
# while a disagreement can still be re-run instead of reported.
#
# WHAT IS ALREADY DISCHARGED, so this leg does not buy it twice. On 2026-09-19
# bench/results/identity_break/2026-09-19_hardware-gaps/ landed two-device
# columns for eleven of the thirteen on BOTH vendors (nvidia-par-*.json and
# amd-par-*-two.json). All 99 of their cells were checked here against the
# one-device NVIDIA column above and every one is equal. What that set does NOT
# contain is par-forest-pool or par-rbf-sampler at any recent commit, and it
# contains no AMD ONE-device column for any of the thirteen -- which is exactly
# the admissible thing the matrix is missing. Both gaps are what this leg buys.
#
# ---------------------------------------------------------------------------
# THE DEFECT THAT KILLED TWO PODS TODAY, AND THE PROBE THAT COULD NOT CATCH IT.
#
# `import mojolearn` fails on any freshly source-built Linux box with
#
#   OSError: python/mojolearn/.libs/libMojolearnMath.so: cannot open shared
#   object file: No such file or directory
#
# because `python/mojolearn/_training_impl.py:1844` writes
# `def kaiming_uniform(self, shape, fan_in, a=math.sqrt(5.0))`, and that
# DEFAULT ARGUMENT is evaluated at class-definition time, so `_portable_math`
# dlopens the library unconditionally on import. Nothing under `bindings/`
# builds it and `python/mojolearn/.libs/` is gitignored, so `git archive` ships
# nothing. A developer Mac has the file from some past wheel build and never
# notices. This body calls the tree's OWN recipe,
# `packaging/portable_math/stage.py`'s build(), rather than copying its
# compiler flags (-ffp-contract=off, -fno-fast-math, -nostdlib) -- those flags
# ARE the arithmetic contract and a second copy of them is a second answer.
#
# AND THE PROBE IS PLACED WHERE IT CAN FAIL AND WHERE IT CAN PASS. Both of
# today's legs put `import mojolearn` BEFORE the bindings build, where it
# ALWAYS fails -- `_backend.select()` raises "no identical binary exists under
# .../identical" long before it could reach the math library. Measured:
# ~/mojolearn-evidence/e1g/2026-09-19_amd-gpu-class-gaps-run2/remote/
# gpu-class-gaps/logs/import_probe.log, exit 1, while both columns that ran
# after it came home clean. A check that cannot pass reports nothing. So there
# are TWO probes here and each is at the only point it is informative:
#   libm_probe   BEFORE the builds. ctypes.CDLL on the library alone, which
#                needs no binding and therefore CAN pass -- it is the actual
#                question ("did stage.build() produce a loadable .so?").
#   import_probe AFTER the builds, where `import mojolearn` is a real question,
#                and it prints the vendor it selected.
#
# POSIX sh: the pod's /bin/sh is dash. `set -u` and deliberately NOT `set -e`:
# a phase that fails is a FINDING and its log must come home.
set -u
cd /root/mojolearn || exit 9
OUT=/root/gemm_leg_out/two-device-par
mkdir -p "$OUT/logs"
G="$OUT/gate.txt"
JOBS="${MOJOLEARN_COMPILE_JOBS:-8}"
# rocminfo and rocm-smi live in /opt/rocm/bin, which the ROCm images do not
# always put on a non-login shell's PATH.
PATH="$PATH:/opt/rocm/bin"
export PATH

# THE THIRTEEN, and nothing else. Eleven are the matrix's nvidia-only par
# lanes; par-forest-pool and par-rbf-sampler ride along because they are the
# two whose two-device claim has NOT been restated since 2026-09-15 and they
# cost about two minutes between them.
LANES=par-boosting-clf,par-boosting-reg,par-cd-elasticnet,par-forest-et-clf,par-forest-pool,par-forest-reg,par-gram-ols,par-gram-pca,par-gram-tsvd,par-queries-nn,par-rbf-sampler,par-scaler-minmax,par-svm-svr

# THE SHAPE IS THE DEFAULT ONE AND SMALL BY CONSTRUCTION. Every lane body above
# fits in a few thousand rows (16 to 40 trees, 20 boosting rounds, 2000 SVM
# rows, 64 query rows). This is an IDENTITY leg: no --n, no --fixtures cap, no
# MOJOLEARN_IDENTITY_WIDE, because `admit` refuses a non-default fixture size
# ("non-default fixture size or wide mode") and a capped column is not the
# column the matrix reads. Measured cost of the eleven on a 4090, from the
# column named above: 697 s for all nine fixtures at two repeats.

T0=$(date +%s)
# The runner's own work bound is MINUTES*60 - 600. 2900 s leaves it margin to
# poll, fetch and terminate after this body writes its sentinel.
BUDGET="${MOJOLEARN_PAR_LEG_BUDGET:-2900}"
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

# THE COMMIT WITNESS, OR identity_break REFUSES TO WRITE A JSON. The box has no
# .git (the runner ships `git archive` at a pinned sha) and the gemm payload
# writes no commit.txt for the extra body. The runner DOES record the commit in
# /root/gemm_leg_out/leg.txt. Take it from there, and NEVER TYPE A SHA.
if [ ! -s /root/mojolearn/commit.txt ]; then
    _c=$(sed -n 's/^commit=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -1 | awk '{print $1}')
    case "$_c" in
        [0-9a-f][0-9a-f]*) printf '%s\n' "$_c" > /root/mojolearn/commit.txt ;;
        *) say "NO COMMIT WITNESS: leg.txt gave '$_c'; every identity phase will refuse" ;;
    esac
fi
say "commit=$(head -1 /root/mojolearn/commit.txt 2>/dev/null)"

# ------------------------------------------------------------ the box itself
{ rocm-smi --showproductname; echo "-- driver --"; cat /sys/module/amdgpu/version 2>/dev/null;
  echo "-- agents --"; rocminfo 2>/dev/null | grep -E 'Name:|gfx' | head -40; } > "$OUT/logs/device.txt" 2>&1
say "device=$(rocm-smi --showproductname 2>/dev/null | tr '\n' ' ' | cut -c1-200)"
say "amdgpu_driver=$(cat /sys/module/amdgpu/version 2>/dev/null)"
# TWO DEVICES, COUNTED AND PRINTED, NOT ASSUMED. This is the entire reason the
# pod costs twice what the single-device legs cost today. A box that came back
# with one agent would run PHASE TWO as a second copy of PHASE ONE and the
# "they agree" line would be a tautology -- the classic verification that
# cannot fail. So the count is read here, recorded, and PHASE TWO is refused by
# name if it is not 2.
AGENTS=$(rocminfo 2>/dev/null | grep -c -E '^ *Name: *gfx')
say "visible_agents=$AGENTS"
if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
    MOJOLEARN_GPU_ARCHS=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-z]+')
    export MOJOLEARN_GPU_ARCHS
fi
say "gpu_archs=${MOJOLEARN_GPU_ARCHS:-unset}"
[ -n "${MOJOLEARN_GPU_ARCHS:-}" ] || say "NO ARCHITECTURE: every bindings/build*.sh refuses without one"
_prod=$(rocm-smi --showproductname 2>/dev/null | sed -n 's/.*[Cc]ard [Ss]eries:[[:space:]]*//p' | head -1)
[ -z "$_prod" ] && _prod=$(rocminfo 2>/dev/null | sed -n 's/^ *Marketing Name: *//p' | grep -i -m1 'instinct\|radeon')
_prod=$(printf '%s' "$_prod" | tr 'A-Z ' 'a-z-' | tr -cd 'a-z0-9-' | cut -c1-24)
[ -z "$_prod" ] && _prod=gpu
# device_class() in python/mojolearn/_verify_reference.py reads the class out of
# this string and it must contain "amd", which it does, first. It must ALSO NOT
# END IN "-two": admit() refuses such a column as a "two-device part" whatever
# its par_devices says, and PHASE ONE must survive that check.
LABEL="amd-$_prod-${MOJOLEARN_GPU_ARCHS:-gfx}"
say "vendor_label=$LABEL"

build() {
    run "$1" timeout "$(cap 420)" env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 \
        MOJOLEARN_COMPILE_JOBS="$JOBS" sh "bindings/$1.sh"
}

# ------------------------------------ THE HOST MATH LIBRARY, OR NOTHING IMPORTS
run portable_math timeout "$(cap 300)" env PYTHONPATH=/root/mojolearn/packaging/portable_math \
    pixi run python -c "import pathlib, stage; stage.build(pathlib.Path('/root/mojolearn/python/mojolearn/.libs/libMojolearnMath.so'))"
say "portable_math_exit=$(awk -F'	' '$1=="portable_math"{print $2}' "$OUT/status.tsv")"
ls -l python/mojolearn/.libs/ >> "$G" 2>&1
# The probe that CAN pass here: load the library itself, by the path
# _portable_math.py uses, with nothing else in the way.
run libm_probe timeout "$(cap 120)" pixi run python -c \
    "import ctypes; ctypes.CDLL('/root/mojolearn/python/mojolearn/.libs/libMojolearnMath.so'); print('LIBM_LOADS')"
say "libm_probe=$(tail -1 "$OUT/logs/libm_probe.log" 2>/dev/null)"

# ------------------------------------------------------------------- bindings
# The ten that the NVIDIA column for eleven of these same lanes built and ran
# green on 2026-09-17 (its extra_body.sh, same lane family), plus
# build_kernel_methods for par-rbf-sampler, whose RBFSampler lives in
# python/mojolearn/kernel_methods.py and is in no other family.
for b in build build_linalg build_estimators build_trees build_rf build_gbdt \
         build_solver build_svm build_preprocessing build_metrics build_kernel_methods; do
    build "$b"
done
say "builds_done=$(awk -F'	' '$1 ~ /^build/ && $2==0' "$OUT/status.tsv" | wc -l)"
sha256sum python/mojolearn/identical/*.so >> "$G" 2>/dev/null
# NOW the import is a real question, and the answer is the backend this column
# claims.
run import_probe timeout "$(cap 180)" env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
    pixi run python -c "import mojolearn; print('IMPORT_OK', mojolearn.vendor(), mojolearn.__version__)"
say "import_probe=$(tail -1 "$OUT/logs/import_probe.log" 2>/dev/null)"

# ===================================================== PHASE ONE: THE DELIVERABLE
# MOJOLEARN_PAR_DEVICES=0 -> package.par_devices="0" -> admit() returns None ->
# tools/verification_matrix.py counts it. Default fixtures, default size, two
# repeats in one process. No sabotage switch is set anywhere in this file.
# It runs FIRST so that a lease that runs short still brings home the half that
# the matrix can read.
run column-one timeout "$(cap 1500)" env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_PAR_DEVICES=0 \
    PYTHONPATH=/root/mojolearn/python \
    pixi run python tools/identity_break.py --lanes "$LANES" --repeats 2 \
    --vendor "$LABEL" --json "$OUT/$LABEL.par-one.json"
say "column_one_exit=$(awk -F'	' '$1=="column-one"{print $2}' "$OUT/status.tsv")"
grep -E '^cells=|^# CELL|MOVED|DIVERGENT|REFUSED|RELOAD' "$OUT/logs/column-one.log" | tail -60 >> "$G"
say "elapsed_after_one=$(( $(date +%s) - T0 ))"

# ===================================================== PHASE TWO: THE PAR CLAIM
# MOJOLEARN_PAR_DEVICES=0,1 -> package.par_devices="0,1". This column is
# INADMISSIBLE ON PURPOSE (admit: "par_devices 0,1") and is not evidence of
# coverage; it is evidence of the drivers' equality claim, and it is worth
# recording only because PHASE ONE above came off the same build on the same
# box at the same commit.
if [ "${AGENTS:-0}" -lt 2 ]; then
    say "PHASE TWO REFUSED: rocminfo reports $AGENTS gfx agent(s); a two-device claim needs 2. Nothing was run."
else
    run column-two timeout "$(cap 1500)" env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_PAR_DEVICES=0,1 \
        PYTHONPATH=/root/mojolearn/python \
        pixi run python tools/identity_break.py --lanes "$LANES" --repeats 2 \
        --vendor "$LABEL" --json "$OUT/$LABEL.par-two.json"
    say "column_two_exit=$(awk -F'	' '$1=="column-two"{print $2}' "$OUT/status.tsv")"
    grep -E '^cells=|^# CELL|MOVED|DIVERGENT|REFUSED|RELOAD' "$OUT/logs/column-two.log" | tail -60 >> "$G"
fi
say "elapsed_after_two=$(( $(date +%s) - T0 ))"

# ============================================ PHASE DIFF: ON THE BOX, NOT AT HOME
# The whole point of one lease. `--diff` exits non-zero on any DIVERGENT, MOVED
# or RELOAD-MOVED cell, and it runs HERE, while the box that produced a
# disagreement is still rented and a lane can be re-run SOLO to see whether the
# disagreement survives being alone. Nothing else runs on this box at any point
# in this file -- every phase is one process, in order -- so a cell that
# disagrees here has already met that bar once.
if [ -s "$OUT/$LABEL.par-one.json" ] && [ -s "$OUT/$LABEL.par-two.json" ]; then
    run par_diff timeout "$(cap 300)" env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
        pixi run python tools/identity_break.py --diff "$OUT/$LABEL.par-one.json" "$OUT/$LABEL.par-two.json"
    say "par_diff_exit=$(awk -F'	' '$1=="par_diff"{print $2}' "$OUT/status.tsv")"
    grep -E 'DIVERGENT|MOVED|REFUSED|NOT-COMPARED' "$OUT/logs/par_diff.log" | head -40 >> "$G"
    # THE SOLO RE-RUN, and only for what disagreed. A lane named here is
    # re-run one device at a time into files this leg does NOT commit; they
    # exist so the divergence is reproduced before anyone reports it.
    # Both diff tables put `lane/fixture` in the first pipe-delimited field;
    # the lane is what is before the slash, and it is the unit --lanes takes.
    _bad=$(grep -E '^\| *par-[a-z0-9/_-]+ ' "$OUT/logs/par_diff.log" 2>/dev/null \
           | grep -E 'DIVERGENT|MOVED' | awk -F'|' '{print $2}' | awk '{print $1}' \
           | cut -d/ -f1 | sort -u | tr '\n' ',' | sed 's/,$//')
    if [ -n "$_bad" ]; then
        say "DISAGREEING LANES: $_bad -- re-running each arm SOLO before this is reported"
        run solo_one timeout "$(cap 600)" env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_PAR_DEVICES=0 \
            PYTHONPATH=/root/mojolearn/python pixi run python tools/identity_break.py \
            --lanes "$_bad" --repeats 2 --vendor "$LABEL" --json "$OUT/solo-one-rerun.json"
        run solo_two timeout "$(cap 600)" env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_PAR_DEVICES=0,1 \
            PYTHONPATH=/root/mojolearn/python pixi run python tools/identity_break.py \
            --lanes "$_bad" --repeats 2 --vendor "$LABEL" --json "$OUT/solo-two-rerun.json"
        run solo_diff timeout "$(cap 300)" env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
            pixi run python tools/identity_break.py --diff "$OUT/solo-one-rerun.json" "$OUT/solo-two-rerun.json"
        say "solo_diff_exit=$(awk -F'	' '$1=="solo_diff"{print $2}' "$OUT/status.tsv")"
    else
        say "no DIVERGENT or MOVED cell in the one-vs-two diff"
    fi
else
    say "PHASE DIFF SKIPPED: one or both columns are missing or empty"
fi

# ========================== ADMISSIBILITY, WHILE THE BOX CAN STILL BE ASKED AGAIN
# admit() is PATH SENSITIVE (_EXCLUDED_PATH_TOKENS, _EXCLUDED_BASENAME_TOKENS
# and the quarantine prefix all read the path). Asking it here about the
# ON-BOX path would answer a question nobody will ask again, so it is asked
# about THE PATH THE COLUMN WILL BE COMMITTED AT. It is asked a SECOND time at
# home, at the real path, because only that one is the claim.
# MOJOLEARN_NUMERIC_MODE is set because `_verify_reference` lives INSIDE the
# package, so reaching it runs `mojolearn/__init__.py` and `_backend.select()`,
# which refuses an unset mode on a box that has only identical bindings.
run admit_check timeout "$(cap 120)" env MOJOLEARN_NUMERIC_MODE=identical \
    PYTHONPATH=/root/mojolearn/python pixi run python -c "
import json, sys
from mojolearn import _verify_reference as vr
base = 'bench/results/identity_break/2026-09-19_par-lane-amd-class/'
for name, expect in (('$LABEL.par-one.json', 'ADMISSIBLE'), ('$LABEL.par-two.json', 'par_devices 0,1')):
    try:
        j = json.load(open('$OUT/' + name))
    except Exception as exc:
        print(name, 'UNREADABLE', exc); continue
    why = vr.admit(j, base + name)
    print(name, 'par_devices=' + str((j.get('package') or {}).get('par_devices')),
          'cells=' + str(len(j.get('cells') or {})),
          'admit=' + ('ADMISSIBLE' if why is None else why),
          'EXPECTED ' + expect)
"
cat "$OUT/logs/admit_check.log" >> "$G" 2>/dev/null

# --------------------------------------------------------------- bring it home
cp "$OUT/logs/column-one.log" "$OUT/column-one.log" 2>/dev/null
cp "$OUT/logs/column-two.log" "$OUT/column-two.log" 2>/dev/null
cp "$OUT/logs/par_diff.log" "$OUT/par_diff.log" 2>/dev/null
cat "$OUT/status.tsv" >> "$G" 2>/dev/null
ls -l "$OUT" >> "$G" 2>/dev/null
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
