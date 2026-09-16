#!/bin/sh
# tools/decode_columns_leg.sh: the on-box body (MOJOLEARN_GEMM_LEG_EXTRA for
# tools/gemm_remote_leg.sh, either vendor) of lane/decode-columns, 2026-09-16.
#
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/decode_columns_leg.sh \
#   sh tools/gemm_remote_leg.sh nvidia --rent
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/decode_columns_leg.sh \
#   sh tools/gemm_remote_leg.sh amd --rent
#
# NO GPU IS PINNED and none needs to be. Any NVIDIA card and any AMD card
# answers this leg, and the caller walks a cheapest-first list of specs and
# stops at the first one that exists. A pinned spec starved a Hot Aisle leg
# for thirty minutes this morning and created nothing.
#
# WHAT THIS BOX IS FOR. lane/stateful-cpu-decoding (merged 3708d7596) claims
# that a sequence decoded ONE TOKEN AT A TIME with a carried state is
# BITWISE the same sequence run as ONE fresh-state forward pass, at every
# position. It proved that on TWO columns, the CPU host column cpu-apple-m4
# and the Apple/Metal column, and wrote in its own report: "NVIDIA and AMD
# confirmation of the stepfull part is OWED. No box was rented (asking
# first)." This box is the answer to that sentence.
#
# WHAT THE stepfull PART ASKS ON THIS BOX, AND WHY IT IS NOT THE CPU RUN
# AGAIN. tools/identity_break.py's `_public_est` returns the *Inference
# wrapper ONLY on a CPU column ("GPU columns are unchanged"). Here
# ml.vendor() is not "cpu", so the part asks the GPU block classes
# themselves: TransformerBlock, Mamba1/2/3Block and SambaStack, their own
# forward, allocate_state and step, on this silicon. The hash is the full
# pass's bytes, so a column that holds the equality lands on the SAME
# sixteen hex digits the CPU and Apple columns recorded.
#
# THE SABOTAGE ARMS ARE NOT OPTIONAL and there are TWO, because the first
# one this part ever had could not fail.
#
#   (a) MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1 perturbs the whole-sequence pass
#       so every hashed stepfull cell MUST read BATCH_MOVED with the
#       position and both bit patterns. Rehearsed on the Mac before this box
#       was rented: 8 of 8 fired.
#
#   (b) the ONE ULP arm. lane/stateful-cpu-decoding's FIRST attempt at it
#       perturbed k_cache[0] and h[0] and reported BITWISE EQUAL, which is
#       indistinguishable from a pass. The causes were structural: the
#       transformer's k_cache absorbs one ULP at all of the first thirty-two
#       cells, mamba1's conv_window absorbs it at cells 0 and 1, and
#       mamba2's h is all zeros inside the first chunk so +1 ULP there is a
#       denormal that ftz flushes back to zero. So the arm SCANS for a
#       sensitive cell, prints the bits before and after, and FAILS when no
#       cell fires. tools/step_vs_full_check.py carries it for the CPU host
#       route; the heredoc below carries the SAME arm over the GPU classes,
#       which is the arm this box is for.
#
# POSIX sh: the pod's /bin/sh is dash. `set -u` and deliberately NOT `set -e`:
# a phase that fails is a FINDING and its log must come home.
set -u
cd /root/mojolearn || exit 9
OUT=/root/gemm_leg_out/decode-columns
mkdir -p "$OUT/logs"
G="$OUT/gate.txt"
LANES=transformer,transformer-window,mamba1,mamba2,mamba2-dtlimit,mamba3,samba,samba-untied-dropout-accum
JOBS="${MOJOLEARN_COMPILE_JOBS:-8}"

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
say "lanes=$LANES"

# THE COMMIT WITNESS, OR identity_break REFUSES TO WRITE A JSON. The box has
# no .git and the gemm payload does not write commit.txt for the extra body;
# a 2026-09-16 leg reached its identity phase with an empty witness and had
# to be fixed by hand over ssh mid-build. The runner DOES record the commit
# in /root/gemm_leg_out/leg.txt. Take it from there, and never guess.
if [ ! -s /root/mojolearn/commit.txt ]; then
    _c=$(sed -n 's/^commit=//p' /root/gemm_leg_out/leg.txt 2>/dev/null | head -1)
    case "$_c" in
        [0-9a-f][0-9a-f]*) printf '%s\n' "$_c" > /root/mojolearn/commit.txt ;;
        *) say "NO COMMIT WITNESS: leg.txt gave '$_c'; every identity phase will refuse" ;;
    esac
fi
say "commit=$(head -1 /root/mojolearn/commit.txt 2>/dev/null)"

# ------------------------------------------------------------ the box itself
IS_AMD=0
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader > "$OUT/logs/device.txt" 2>&1
else
    IS_AMD=1
    { rocminfo 2>/dev/null | grep -m4 -E 'gfx[0-9a-z]+|Marketing Name'; rocm-smi --showproductname 2>/dev/null | head -20; } > "$OUT/logs/device.txt" 2>&1
fi
say "device=$(tr '\n' ' ' < "$OUT/logs/device.txt")"
if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
    if [ "$IS_AMD" = 0 ]; then
        _cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
        case "$_cc" in
            9.0)  MOJOLEARN_GPU_ARCHS=sm_90a ;;
            8.9)  MOJOLEARN_GPU_ARCHS=sm_89 ;;
            8.6)  MOJOLEARN_GPU_ARCHS=sm_86 ;;
            8.0)  MOJOLEARN_GPU_ARCHS=sm_80 ;;
            12.0) MOJOLEARN_GPU_ARCHS=sm_120a ;;
            *)    MOJOLEARN_GPU_ARCHS="sm_$(echo "$_cc" | tr -d .)" ;;
        esac
    else
        MOJOLEARN_GPU_ARCHS=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-z]+')
    fi
    export MOJOLEARN_GPU_ARCHS
fi
say "gpu_archs=${MOJOLEARN_GPU_ARCHS:-unset}"
if [ "$IS_AMD" = 0 ]; then
    LABEL="nvidia-$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr 'A-Z ' 'a-z-' | tr -cd 'a-z0-9-' | cut -c1-24)-${MOJOLEARN_GPU_ARCHS}"
else
    LABEL="amd-${MOJOLEARN_GPU_ARCHS}"
fi
say "vendor_label=$LABEL"

# ------------------------------------------------------------------ GPU builds
# The eight decode lanes reach three families and the base: the Mamba blocks
# (_mojolearn_mamba), the Transformer block (_mojolearn_transformer), the
# neural RNG and optimizer the fits run through (_mojolearn_training), and
# the shared kernels and fixtures under bindings/build.sh. Nothing else is
# built, because every other family is a minute of a sixty-minute lease
# spent on a lane this box is not for.
for b in build build_training build_mamba build_transformer; do
    run "$b" env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 \
        MOJOLEARN_COMPILE_JOBS="$JOBS" sh "bindings/$b.sh"
done
sha256sum python/mojolearn/identical/*.so >> "$G" 2>/dev/null
say "vendor_readback=$(env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python pixi run python -c 'import mojolearn; print(mojolearn.vendor())' 2>&1 | tail -1)"

# ================================================================ THE DELIVERABLE
# The GPU stepfull column, all eight lanes, base fixture, twice in one
# process. --no-batch and --no-rlpair keep this to the part that is owed:
# every other part of these lanes already has its GPU columns.
run stepfull env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
    pixi run python tools/identity_break.py --lanes "$LANES" --fixtures base --repeats 2 \
    --step-full --no-batch --no-rlpair --vendor "$LABEL" --json "$OUT/$LABEL.stepfull.json"
say "stepfull_exit=$(awk -F'	' '$1=="stepfull"{print $2}' "$OUT/status.tsv")"
grep -E '^cells=|^stepfull:|stepfull *\||MOVED|DIVERGENT|REFUSED' "$OUT/logs/stepfull.log" | head -40 >> "$G"
cp "$OUT/logs/stepfull.log" "$OUT/stepfull.log" 2>/dev/null

# ============================================== SABOTAGE (a): the part's own arm
# Every hashed stepfull cell MUST read BATCH_MOVED. A column of cells that
# could not have failed is not a column. Rehearsed on the Mac first: 8 of 8.
run stepfull-sabotage env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_IDENTITY_BATCH_SABOTAGE=1 \
    PYTHONPATH=/root/mojolearn/python \
    pixi run python tools/identity_break.py --lanes "$LANES" --fixtures base --repeats 1 \
    --step-full --no-batch --no-rlpair --vendor "$LABEL" --json "$OUT/$LABEL.stepfull-sabotage.json"
say "stepfull_sabotage_exit=$(awk -F'	' '$1=="stepfull-sabotage"{print $2}' "$OUT/status.tsv")"
grep -E 'BATCH_MOVED|^stepfull:' "$OUT/logs/stepfull-sabotage.log" | head -30 >> "$G"
cp "$OUT/logs/stepfull-sabotage.log" "$OUT/stepfull-sabotage.log" 2>/dev/null
# The diff of the two runs on THIS box, so the control is attached to the
# column it controls and not only to a second file.
run diff-sabotage env PYTHONPATH=/root/mojolearn/python pixi run python tools/identity_break.py \
    --diff "$OUT/$LABEL.stepfull.json" "$OUT/$LABEL.stepfull-sabotage.json" --lanes "$LANES"
cp "$OUT/logs/diff-sabotage.log" "$OUT/diff.column-vs-sabotage.txt" 2>/dev/null

# =================================================== SABOTAGE (b): ONE ULP, GPU
# The arm lane/stateful-cpu-decoding got wrong the first time. It perturbs
# ONE carried cell of ONE carried buffer by ONE ULP before step `at`, and
# requires the per-position comparison to fire. It SCANS, because a single
# ULP in a single cached component is often absorbed, and it FAILS the run
# when no cell fires, so a silent BITWISE EQUAL cannot pass for a result.
#
# tools/step_vs_full_check.py carries this arm over the *Inference wrappers,
# which bind _mojolearn_neural_host BY NAME and are therefore the CPU host
# route even here. This block reuses that file's machinery with the GPU
# classes swapped in, so the arm runs on the silicon this box was rented
# for. The swap is by name because build_cases() reads the class off the
# `mojolearn` module at call time.
cat > /root/step_full_gpu.py <<'PYEOF'
"""The one-ULP arm of tools/step_vs_full_check.py over the GPU block
classes (lane/decode-columns, 2026-09-16).

step_vs_full_check.build_cases() names ml.TransformerBlockInference,
ml.Mamba{1,2,3}BlockInference and ml.SambaInference, each of which binds
_mojolearn_neural_host by name and so runs the CPU host route on any box.
The GPU classes take the same constructor arguments (weights dict; n_heads
and window keywords for the transformer; config and weights for the stack),
so swapping them onto the module runs the identical comparison and the
identical fail-first scan on the GPU.
"""
import sys
import mojolearn as ml

sys.path.insert(0, "/root/mojolearn/tools")
import step_vs_full_check as S

print("vendor =", ml.vendor())
if ml.vendor() == "cpu":
    raise SystemExit("REFUSING: this is the GPU arm and ml.vendor() says cpu")
ml.TransformerBlockInference = ml.TransformerBlock
ml.Mamba1BlockInference = ml.Mamba1Block
ml.Mamba2BlockInference = ml.Mamba2Block
ml.Mamba3BlockInference = ml.Mamba3Block
ml.SambaInference = ml.SambaStack
print("classes:", ml.TransformerBlockInference.__name__, ml.Mamba1BlockInference.__name__,
      ml.Mamba2BlockInference.__name__, ml.Mamba3BlockInference.__name__,
      ml.SambaInference.__name__)
sys.exit(S.main(sys.argv[1:]))
PYEOF
run stepfull-gpu-equal env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
    pixi run python /root/step_full_gpu.py
say "gpu_one_ulp_equal_exit=$(awk -F'	' '$1=="stepfull-gpu-equal"{print $2}' "$OUT/status.tsv")"
cat "$OUT/logs/stepfull-gpu-equal.log" >> "$G" 2>/dev/null
run stepfull-gpu-oneulp env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
    pixi run python /root/step_full_gpu.py --sabotage
say "gpu_one_ulp_sabotage_exit=$(awk -F'	' '$1=="stepfull-gpu-oneulp"{print $2}' "$OUT/status.tsv")"
cat "$OUT/logs/stepfull-gpu-oneulp.log" >> "$G" 2>/dev/null
cp "$OUT/logs/stepfull-gpu-equal.log" "$OUT/one-ulp.gpu.equal.log" 2>/dev/null
cp "$OUT/logs/stepfull-gpu-oneulp.log" "$OUT/one-ulp.gpu.sabotage.log" 2>/dev/null

# ============================== the x86/aarch64 CPU host route on this same box
# A third column, free, from a machine that is neither this Mac nor Apple
# silicon: the *Inference wrappers over the neural host binding, which is
# what a user who pip installs on Linux and never touches a GPU actually
# runs. `env -u MOJOLEARN_GPU_ARCHS` FIRST and not as decoration: this file
# exports that variable for the GPU builds above and
# bindings/build_host_family.sh refuses a CPU build that carries one ("a CPU
# build takes no MOJOLEARN_GPU_ARCHS"). On 2026-09-16 four host builds exited
# 2 in zero seconds for exactly that and took a check and both sabotage arms
# with them. `env FOO=1 -u BAR` does NOT unset BAR, because env stops parsing
# options at the first assignment; the -u comes first here for that reason.
run build-neural-host env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_NUMERIC_MODE=identical \
    MOJOLEARN_BUILD_JOBS="$JOBS" sh bindings/build_neural_host.sh
say "build_neural_host_exit=$(awk -F'	' '$1=="build-neural-host"{print $2}' "$OUT/status.tsv")"
run hostroute-equal env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
    pixi run python tools/step_vs_full_check.py
say "hostroute_equal_exit=$(awk -F'	' '$1=="hostroute-equal"{print $2}' "$OUT/status.tsv")"
cat "$OUT/logs/hostroute-equal.log" >> "$G" 2>/dev/null
run hostroute-oneulp env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python \
    pixi run python tools/step_vs_full_check.py --sabotage
say "hostroute_one_ulp_exit=$(awk -F'	' '$1=="hostroute-oneulp"{print $2}' "$OUT/status.tsv")"
cat "$OUT/logs/hostroute-oneulp.log" >> "$G" 2>/dev/null
cp "$OUT/logs/hostroute-equal.log" "$OUT/one-ulp.hostroute.equal.log" 2>/dev/null
cp "$OUT/logs/hostroute-oneulp.log" "$OUT/one-ulp.hostroute.sabotage.log" 2>/dev/null

# THE CROSS-COLUMN DIFF IS NOT RUN HERE. `git archive` ships the source
# only, so bench/results/ does not exist on this box and the CPU and Apple
# columns are not here to diff against. A 2026-09-16 leg lost two diff
# phases to FileNotFoundError for exactly that. The diff costs nothing at
# home, against the fetched column, and that is where it runs.

# --------------------------------------------------------------- bring it home
cat "$OUT/status.tsv" >> "$G" 2>/dev/null
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
