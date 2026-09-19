#!/bin/sh
# tools/gap_column_leg.sh: the on-box body (MOJOLEARN_GEMM_LEG_EXTRA for
# tools/gemm_remote_leg.sh) of a GPU identity column that closes vendor-class
# gaps in docs/VERIFICATION_MATRIX.md. ONE BODY, EITHER VENDOR, LANES FROM
# THE ENVIRONMENT.
#
#   MOJOLEARN_GAP_LANES=a,b,c MOJOLEARN_GAP_SLUG=vendor-class-gaps \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/gap_column_leg.sh \
#   sh tools/gemm_remote_leg.sh nvidia --rent --allow-concurrent \
#      --local-card <an apple.card from a previous leg>
#
# WHY THIS FILE IS NOT tools/single_device_gaps_nvidia_leg.sh ANY MORE.
# It is that file (ed0d6e48b), renamed and generalized on 2026-09-19 rather
# than copied. By that afternoon the tree held THREE near-identical gap-column
# bodies -- tools/gpu_class_gaps_nvidia_leg.sh, tools/gpu_class_gaps_amd_leg.sh
# and this one -- differing in their lane lists, their `nvidia-smi` versus
# `rocminfo` probe, and in WHICH OF THE DAY'S THREE DEFECTS EACH ONE HAD
# LEARNED ABOUT YET. That last difference is the whole problem: the AMD body
# learned about libMojolearnMath.so and never learned to derive its build set,
# and the NVIDIA bodies learned the opposite halves on different pods. A
# fourth copy would have arrived knowing some other subset. So the vendor
# probe below branches, the lane list comes from the caller, and there is one
# place where a lesson lands.
#
# WHAT THE VENDOR BRANCH ACTUALLY DECIDES, AND WHAT IT DOES NOT.
# Only three things: which SMI to read the device out of, how to derive
# MOJOLEARN_GPU_ARCHS, and what goes in the label. Everything downstream --
# the build set, the host families, the fixtures, the repeats, the
# admissibility rules -- is vendor-independent, and it has to be, because the
# claim this column records is that it is NOT vendor-dependent.
#
# THE FOUR DEFECTS THIS BODY IS BUILT AROUND, ALL FOUND ON 2026-09-19:
#
# 1. THE HOST MATH LIBRARY, OR NOTHING IMPORTS. See the phase itself, below.
#    Two pods came home with every binding built and not one cell. It is not
#    a GPU, a vendor or a lane-list problem and it fails identically on both.
#
# 2. A HAND-WRITTEN GPU FAMILY LIST ROTS. Both of the morning's legs shipped
#    one and each lost lanes to it in the same hour: the NVIDIA list omitted
#    `build_preprocessing`, so gp-normalize-y recorded 18 cells reading
#    "needs .../identical/_mojolearn_preprocessing.so, which is not built"
#    (GaussianProcessRegressor(normalize_y=True) centres y with StandardScaler's
#    pinned folds, and nothing in the lane's NAME says so); the AMD list
#    omitted `build_metrics` and gbdt-adapter-score-weighted lost nine cells
#    the same way. So the GPU set below is DERIVED from bindings/build*.sh and
#    every family is built. Measured: 1204 s for all 23 on an RTX 4090 against
#    605 s for a picked eleven. Three hundred seconds buys the class of
#    mistake away inside a lease that still finished with 35 minutes spare.
#
# 3. AND THE HOST FAMILIES ARE A SECOND, SEPARATE LIST THAT NOBODY BUILT.
#    Every lane here is anchored `cross-route` by tools/lane_applicability.py:
#    the GPU cell is compared against the SAME BOX'S HOST ROUTE, so a lane
#    whose host binding is missing refuses on a GPU column even with all 23
#    GPU families present. `bindings/build_*_host.sh` is a shim that execs
#    bindings/build_host_family.sh, which REFUSES when MOJOLEARN_GPU_ARCHS is
#    set -- which this file exports -- so they cannot simply be swept into the
#    loop above. They are derived instead from the ONE table that already
#    answers "which host family serves this lane", python/mojolearn/
#    host_surface.py's FAMILIES, read at run time with the lane list in hand.
#    Not typed out here: a list of families in a leg script is the same second
#    answer that defect 2 is about, wearing host clothes.
#
# 4. AND THEN THE BACKSTOP ANYWAY. After the column runs, this file GREPS ITS
#    OWN JSON for the refusal text's `bindings/build_*.sh`, builds whatever it
#    names, and runs the column again over the same path. The refusal NAMES
#    ITS OWN CURE verbatim in the cell. If the two derivations above are wrong
#    in a way nobody has thought of, the leg corrects itself inside the lease
#    instead of reporting the same gap a third time. Normally it finds nothing
#    and says so, which is what it reported on pod o20ynm82l55geo.
#
# THE DELIVERABLE COMES BEFORE THE INSURANCE. The ordering below is: the
# families this lane list is DECLARED to need, then the column, then every
# remaining GPU family, then the backstop and a second column if it found
# anything. ed0d6e48b built all 23 first and had 35 minutes to spare, so this
# is not a budget fix; it is so that a lease that dies early still comes home
# with a column rather than with a build log.
#
# A DEGENERATE LANE IS NOT IN THE LANE LIST AND MUST NOT BE.
# tools/lane_applicability.py holds the rule: a lane whose arithmetic is the
# CPU host route measures THAT BOX'S CPU on a GPU column and says nothing
# about cuda or hip. Running one here would record a passing cell that means
# nothing. MOJOLEARN_GAP_DEGENERATE names such lanes instead, and the body
# runs `lane_applicability --check` on them ON THIS BOX so the refusal comes
# home in the operator's own words rather than in mine.
#
# THE COLUMN MUST BE ADMISSIBLE OR IT IS NOT EVIDENCE.
# python/mojolearn/_verify_reference.admit wants identical mode, a 40-hex
# commit, the default fixture set at the default size, one device,
# heldout_seed 1, no `*_sabotage` flag, and a PATH carrying none of
# "partial", "probe", "unfixed", "post-merge-smoke" and no basename carrying
# "sabotage". Hence: no --fixtures cap, no MOJOLEARN_IDENTITY_*_SABOTAGE
# anywhere in this file, the commit witness taken from the runner's own
# leg.txt rather than guessed, and the import check called `import_check` and
# not `import_probe` -- "probe" is an EXCLUDED PATH TOKEN and a stray file of
# that name next to the column is one rename away from refusing it.
#
# THE IMPORT CHECK RUNS AFTER THE BUILDS, NOT BEFORE. Both of the morning's
# legs placed theirs before the bindings build, where `import mojolearn` under
# MOJOLEARN_NUMERIC_MODE=identical CANNOT succeed -- it refuses with "no
# identical binary exists under .../identical" until at least one is built. A
# check that cannot pass is not a check.
#
# ONE DEVICE, ASSERTED AND PRINTED. `admit` refuses a column recording
# par_devices, and a second visible GPU would make every cell below a
# different claim than the one this column is recorded as. On AMD there is a
# second reason: the MI300X SR-IOV peer-copy stale read (fixed by routing
# cross-device bytes through transfer_bytes host staging, d36cd8cb4). This leg
# does not test that fix and does not depend on it -- GPU_COUNT stays 1 and no
# phase here copies a byte between devices.
#
# EVERY PHASE IS BOUNDED, AND THE BOUND SHRINKS AS THE LEASE BURNS. The
# runner polls for a completion sentinel and fetches NOTHING from a body that
# never finishes, so a slow compile must cost a later phase its ceiling rather
# than cost the whole leg its fetch.
#
# POSIX sh: the pod's /bin/sh is dash. `set -u` and deliberately NOT `set -e`:
# a phase that fails is a FINDING and its log must come home.
set -u
cd /root/mojolearn || exit 9
SLUG="${MOJOLEARN_GAP_SLUG:-gap-column}"
OUT="/root/gemm_leg_out/$SLUG"
mkdir -p "$OUT/logs"
G="$OUT/gate.txt"
JOBS="${MOJOLEARN_COMPILE_JOBS:-8}"
# rocminfo and rocm-smi live in /opt/rocm/bin, which the ROCm images do not
# always put on a non-login shell's PATH.
PATH="$PATH:/opt/rocm/bin"
export PATH

LANES="${MOJOLEARN_GAP_LANES:-}"
DEGENERATE="${MOJOLEARN_GAP_DEGENERATE:-}"

T0=$(date +%s)
BUDGET="${MOJOLEARN_GAP_BUDGET:-2500}"
# The runner's own work bound is MINUTES*60 - 600; for a 60-minute lease that
# is 3000 s, and the device check and card ahead of this body cost about 100 s
# of it. 2500 s leaves the runner margin to poll, fetch and terminate after
# this body writes its sentinel.
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
say "slug=$SLUG"
say "lanes=$LANES"
say "degenerate_lanes=${DEGENERATE:-none}"
if [ -z "$LANES" ]; then
    say "NO LANES: MOJOLEARN_GAP_LANES is empty; this body has nothing to record"
    exit 8
fi

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
# The ONLY vendor branch in this file. device_class() in
# python/mojolearn/_verify_reference.py reads the class out of the label, so
# the label must start with "nvidia" or "amd" and below it does.
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    VENDOR=nvidia
    nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader \
        > "$OUT/logs/device.txt" 2>&1
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
    _prod=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
else
    VENDOR=amd
    { rocm-smi --showproductname; echo "-- driver --"; cat /sys/module/amdgpu/version 2>/dev/null;
      echo "-- agents --"; rocminfo 2>/dev/null | grep -E 'Name:|gfx' | head -20; } \
        > "$OUT/logs/device.txt" 2>&1
    say "device=$(rocm-smi --showproductname 2>/dev/null | tr '\n' ' ' | cut -c1-200)"
    say "amdgpu_driver=$(cat /sys/module/amdgpu/version 2>/dev/null)"
    say "visible_gpus=$(rocminfo 2>/dev/null | grep -c -E '^ *Name: *gfx')"
    if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
        MOJOLEARN_GPU_ARCHS=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-z]+')
        export MOJOLEARN_GPU_ARCHS
    fi
    # The marketing name, from rocm-smi's "Card series" field and, when that
    # layout moves between ROCm releases, from rocminfo's Marketing Name.
    _prod=$(rocm-smi --showproductname 2>/dev/null | sed -n 's/.*[Cc]ard [Ss]eries:[[:space:]]*//p' | head -1)
    [ -z "$_prod" ] && _prod=$(rocminfo 2>/dev/null | sed -n 's/^ *Marketing Name: *//p' | grep -i -m1 'instinct\|radeon')
fi
say "vendor_probe=$VENDOR"
say "gpu_archs=${MOJOLEARN_GPU_ARCHS:-unset}"
[ -z "${MOJOLEARN_GPU_ARCHS:-}" ] && say "NO ARCHITECTURE: bindings/build_byte_lm.sh and its siblings refuse without one"
_prod=$(printf '%s' "$_prod" | tr 'A-Z ' 'a-z-' | tr -cd 'a-z0-9-' | cut -c1-24)
[ -z "$_prod" ] && _prod=gpu
LABEL="$VENDOR-$_prod-${MOJOLEARN_GPU_ARCHS:-unknown}"
say "vendor_label=$LABEL"
JSON="$OUT/$LABEL.$SLUG.json"

build() {
    run "$1" timeout "$(cap 420)" env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 \
        MOJOLEARN_COMPILE_JOBS="$JOBS" sh "bindings/$1.sh"
}
# `env -u MOJOLEARN_GPU_ARCHS` comes FIRST and not as decoration: this file
# exports that variable and bindings/build_host_family.sh REFUSES a CPU build
# that carries one. `env FOO=1 -u BAR` does NOT unset BAR, because env stops
# parsing options at the first assignment.
build_host() {
    run "$1" timeout "$(cap 420)" env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_TARGET_COLUMN=cpu \
        MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_BUILD_JOBS="$JOBS" sh "bindings/$1.sh"
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

# -------------------------------------------------- WHAT THIS LANE LIST DECLARES
# Two derivations, both read out of the tree with the lane list in hand, both
# printed so a wrong one is visible in the gate file rather than in a refusal
# nine minutes later. `routes` is the GPU family a host family routes to;
# `training_lanes`/`inference_lanes` is which lanes it serves. This is the
# same table python/mojolearn/host_surface.py's own gate reads.
DECL=$(timeout "$(cap 120)" env PYTHONPATH=/root/mojolearn/python pixi run python - "$LANES" <<'PY' 2>> "$OUT/logs/declared.log"
import sys
import mojolearn.host_surface as hs
want = {s for s in sys.argv[1].split(",") if s}
host, gpu = [], []
for f in hs.FAMILIES:
    served = set(f.get("training_lanes", ())) | set(f.get("inference_lanes", ()))
    if not (served & want):
        continue
    host.append("build_%s_host" % f["family"])
    r = f.get("routes")
    if r:
        gpu.append("build_" + r.replace("_mojolearn_", ""))
print("HOST " + " ".join(sorted(set(host))))
print("GPU " + " ".join(sorted(set(gpu))))
PY
)
DECL_HOST=$(printf '%s\n' "$DECL" | sed -n 's/^HOST //p')
DECL_GPU=$(printf '%s\n' "$DECL" | sed -n 's/^GPU //p')
say "declared_host_families=${DECL_HOST:-NONE-DERIVED}"
say "declared_gpu_families=${DECL_GPU:-NONE-DERIVED}"
[ -z "$DECL_HOST" ] && say "THE HOST DERIVATION RETURNED NOTHING. See logs/declared.log; the loop below builds every GPU family regardless, and the backstop still names what refuses."

# ---------------------------------------- the declared families, then the column
# `build` is the shared kernels and fixtures every lane reaches, so it leads
# whatever the derivation said.
built=0; failed=""; DONE=""
for b in build $DECL_GPU; do
    [ -f "bindings/$b.sh" ] || { say "derived a GPU build that does not exist: $b"; continue; }
    case " $DONE " in *" $b "*) continue ;; esac
    DONE="$DONE $b"
    if build "$b"; then built=$(( built + 1 )); else failed="$failed $b"; fi
done
for b in $DECL_HOST; do
    [ -f "bindings/$b.sh" ] || { say "derived a host build that does not exist: $b"; continue; }
    case " $DONE " in *" $b "*) continue ;; esac
    DONE="$DONE $b"
    if build_host "$b"; then built=$(( built + 1 )); else failed="$failed $b"; fi
done
say "declared_families_built=$built failed=${failed:-none}"

# THE IMPORT CHECK, AFTER THE BUILDS. Under identical mode this refuses
# outright until at least one binding exists, which is why both of
# 2026-09-19's morning legs got a guaranteed failure out of the same two lines.
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
    grep -E '^cells=|MOVED|DIVERGENT|REFUSED|RELOAD' "$OUT/logs/$1.log" | head -120 >> "$G"
    cp "$OUT/logs/$1.log" "$OUT/$1.log" 2>/dev/null
}
column column
say "elapsed_after_column=$(( $(date +%s) - T0 ))"

# --------------------------------------------- THE INSURANCE: every other family
# The deliverable is already on disk. Whatever is left of the budget goes on
# the families the derivation did NOT name, so that the backstop below has
# them in hand: `bindings/build*.sh` minus `build_host_family.sh` (the
# parameterized builder the shims exec, not a build by itself) and minus every
# `build_*_host.sh` already built above.
for s in bindings/build*.sh; do
    n=$(basename "$s" .sh)
    [ "$n" = build_host_family ] && continue
    case " $DONE " in *" $n "*) continue ;; esac
    DONE="$DONE $n"
    case "$n" in
        build_*_host) build_host "$n" || failed="$failed $n" ;;
        *)            build "$n"      || failed="$failed $n" ;;
    esac
done
say "all_families_failed=${failed:-none}"
sha256sum python/mojolearn/identical/*.so python/mojolearn/host/*.so >> "$G" 2>/dev/null
say "elapsed_after_all_families=$(( $(date +%s) - T0 ))"

# ------------------------------------------------------- THE BACKSTOP, AND WHY
# The refusal these lanes exist to escape NAMES ITS OWN CURE, verbatim, in the
# cell: "Build it with MOJOLEARN_NUMERIC_MODE=identical bash
# bindings/build_preprocessing.sh". Read it back out of the JSON rather than
# out of a human's reading of a lane body: if the derivations above missed a
# family for a reason nobody has thought of, this fixes it inside the lease
# instead of reporting the same gap a third time. Normally it finds nothing
# and says so. grep -o, not sed: the whole JSON can be one line and sed would
# keep only the last match on it.
MISSING=$(grep -o 'bindings/build[a-z_]*\.sh' "$JSON" 2>/dev/null | sed 's#bindings/##; s#\.sh$##' | sort -u | tr '\n' ' ')
say "backstop_missing_families=${MISSING:-none}"
if [ -n "$MISSING" ]; then
    for b in $MISSING; do
        [ -f "bindings/$b.sh" ] || { say "backstop: no such build script: $b"; continue; }
        case "$b" in
            *_host) build_host "$b" ;;
            *)      build "$b" ;;
        esac
    done
    sha256sum python/mojolearn/identical/*.so python/mojolearn/host/*.so >> "$G" 2>/dev/null
    column column-after-backstop
fi

# ------------------------------------- THE LANES THAT ARE NOT HERE, IN ITS WORDS
# Not a lane list this file argues with: the rule lives in
# tools/lane_applicability.py and this prints what it says ON THIS BOX. A lane
# whose arithmetic is the CPU host route would record a passing cell here that
# measures this box's CPU and says nothing about cuda or hip.
if [ -n "$DEGENERATE" ]; then
    _col="$VENDOR-1gpu"
    # shellcheck disable=SC2046
    run degenerate_check timeout "$(cap 120)" env PYTHONPATH=/root/mojolearn/python \
        pixi run python tools/lane_applicability.py --check --column "$_col" \
        --lanes $(printf '%s' "$DEGENERATE" | tr ',' ' ')
    say "--- the refusal, verbatim, from $_col ---"
    cat "$OUT/logs/degenerate_check.log" >> "$G" 2>/dev/null
    cp "$OUT/logs/degenerate_check.log" "$OUT/degenerate_check.log" 2>/dev/null
fi

# THE CROSS-COLUMN DIFF IS NOT RUN HERE. `git archive` ships the source only,
# so bench/results/ does not exist on this box and the other vendors' columns
# are not here to diff against. That diff costs nothing at home, against the
# fetched column, and that is where it runs -- and where a DIVERGENT cell gets
# re-run SOLO on this same pod before anyone calls it real.

# --------------------------------------------------------------- bring it home
cat "$OUT/status.tsv" >> "$G" 2>/dev/null
ls -l "$OUT" >> "$G" 2>/dev/null
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
