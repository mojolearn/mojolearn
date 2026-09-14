# Workstream D fixes on a GPU box, HALF A of two (lane/expose-d-fixes).
# Fits one 60-minute lease: the HDBSCAN crash control and bisect, the seven
# touched bindings (build_training added, the base binding last), the seven
# surface test modules, check-hdbscan and check-resample. Half B
# (expose_d_fixes_body_b.sh) runs check-mixture, check-kernel-methods and
# check-cholesky, which need no binding.
#
# Sized from the first MI300X leg at 2b2f568b0 (Hot Aisle 8core,
# bench/results/e1g/2026-09-14_164307-amd-mi300x-hotaisle-expose-d): the body
# started about 4 minutes into the lease, six family builds took about 4
# minutes, the seven test modules about 1, check-cholesky under 1, and
# check-kernel-methods had not printed by the 60-minute cap. Every step here
# has its own bound, and every step prints its elapsed seconds so the next
# leg can be sized from this one.
#
# Launch from a clean detached worktree at the commit under test, e.g.
#   MOJOLEARN_HOTAISLE_EXTRA_ENV="MOJOLEARN_COMMIT=<sha>" tools/hotaisle_leg.sh
# with this file as the extra body. RunPod passes no environment: bake
# commit.txt in a wrapper.
set -u
cd /root/mojolearn 2>/dev/null || cd "$(pwd)"
OUT=/root/gemm_leg_out; mkdir -p "$OUT"
[ -n "${MOJOLEARN_COMMIT:-}" ] && echo "$MOJOLEARN_COMMIT" > commit.txt
[ -f commit.txt ] && echo "commit $(cat commit.txt)"

if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
    if command -v rocminfo > /dev/null 2>&1; then
        MOJOLEARN_GPU_ARCHS=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-z]+')
    elif command -v nvidia-smi > /dev/null 2>&1; then
        _cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
        case "$_cc" in 9.0) MOJOLEARN_GPU_ARCHS=sm_90a ;; *) MOJOLEARN_GPU_ARCHS="sm_$(echo "$_cc" | tr -d .)" ;; esac
    fi
    export MOJOLEARN_GPU_ARCHS
fi
echo "archs=${MOJOLEARN_GPU_ARCHS:-<unset>} column=${MOJOLEARN_TARGET_COLUMN:-<unset>}"

# bounded SECONDS NAME CMD...: runs CMD under timeout(1) when present, logs
# to $OUT/NAME.log, prints the exit code and the elapsed seconds.
bounded() {
    _secs=$1; _name=$2; shift 2
    _t0=$(date +%s)
    if command -v timeout > /dev/null 2>&1; then
        timeout -k 20 "$_secs" "$@" > "$OUT/$_name.log" 2>&1
    else
        "$@" > "$OUT/$_name.log" 2>&1
    fi
    _rc=$?
    echo "== $_name exit $_rc seconds $(( $(date +%s) - _t0 )) (bound $_secs)"
    [ "$_rc" = 124 ] && echo "   $_name HIT ITS BOUND"
    return $_rc
}
quiet() { grep -v "warning:\|^ *\^\|^    var\|^Imported\|^Included\|note:\|^ *~" "$1" | tail -"${2:-3}" | cut -c1-240; }
B="env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1"

# 1. THE HDBSCAN CONTROL. The 2b2f568b0 stability kernel must still crash the
# instruction selector on gfx942; if it builds, the crash was not that
# kernel's and a green default build below proves nothing about the
# respelling. Built into the real output path, which the default build
# overwrites next. Skipped off AMD, where there was never a crash.
case "${MOJOLEARN_GPU_ARCHS:-}" in
gfx*)
    bounded 900 build_hdbscan_control_pre0914 $B MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_HDB_ISEL_PRE_0914" sh bindings/build_hdbscan.sh
    echo "   control (expect exit 139, 'AMDGPU DAG->DAG'):"; grep -m2 -E "Running pass 'AMDGPU|built " "$OUT/build_hdbscan_control_pre0914.log" | cut -c1-200
    rm -f python/mojolearn/identical/_mojolearn_hdbscan.so
    ;;
esac

# 2. THE BINDINGS, hdbscan first so a crash is known early; the base binding
# last (bindings/build.sh; test_kmeans_metric_surface reaches it).
for b in build_hdbscan build_gp build_kernel_methods build_mixture build_resample build_training build; do
    bounded 900 "$b" $B sh "bindings/$b.sh"
    quiet "$OUT/$b.log" 2
done

# 3. THE HDBSCAN BISECT, only when the default build still crashes: each stub
# cuts one part of the stability kernels (hdbscan/impl/detail/stabilities.mojo,
# the gfx942 banner). The part whose stub builds is the construct. Stubbed
# binaries raise by name and are removed before the tests.
if [ ! -f python/mojolearn/identical/_mojolearn_hdbscan.so ]; then
    echo "== default HDBSCAN build did not produce a binary; bisecting"
    for stub in STUB_SUM STUB_MIN STUB_BIRTH STUB_BIRTHS_INIT "STUB_MIN -D MOJOLEARN_HDB_ISEL_STUB_BIRTH -D MOJOLEARN_HDB_ISEL_STUB_SUM"; do
        tag=$(echo "$stub" | tr -c 'A-Za-z0-9\n' '_' | cut -c1-60)
        bounded 600 "build_hdbscan_bisect_$tag" $B MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_HDB_ISEL_$stub" sh bindings/build_hdbscan.sh
        grep -m2 -E "Running pass|built " "$OUT/build_hdbscan_bisect_$tag.log" | cut -c1-200
        rm -f python/mojolearn/identical/_mojolearn_hdbscan.so
    done
fi
ls python/mojolearn/identical/

# 4. THE SEVEN SURFACE TEST MODULES, as modules from python/.
cd python
for t in hdbscan mixture kernel_methods cholesky resample training_primitives kmeans_metric; do
    bounded 600 "test_${t}_surface" env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python pixi run python -m "mojolearn.tests.test_${t}_surface"
    grep -E "^  FAIL|REPORT, not asserted|ARM DID NOT RUN|GREEN|RED|not built" "$OUT/test_${t}_surface.log" | cut -c1-240
done
cd /root/mojolearn 2>/dev/null || cd ..

# 5. TWO CHECKS, bounded. check-hdbscan is the pixi task (FAST, as the task
# is defined) and then the same check under IDENTICAL, which is the tier the
# stability kernel ships in.
bounded 900 check-hdbscan pixi run check-hdbscan
quiet "$OUT/check-hdbscan.log" 4
bounded 900 check-hdbscan-identical pixi run mojo run -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 hdbscan/checks/hdbscan_check.mojo
quiet "$OUT/check-hdbscan-identical.log" 4
bounded 900 check-resample pixi run check-resample
quiet "$OUT/check-resample.log" 4
exit 0
