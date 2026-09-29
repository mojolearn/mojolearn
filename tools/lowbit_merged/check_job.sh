#!/bin/bash
# tools/lowbit_merged/check_job.sh -- THE ONE LIGHT CHECK of lane/lowbit-merged
# before it goes to main (2026-09-29). It answers one question: does main
# still BUILD and pass what it passed, with the low-bit branches merged in.
# The profile's identity was closed on four boxes by its own lanes and is not
# re-argued here.
#
#   1. the two bindings the merge changed build (linalg, device and host),
#      and the base binding the loader tests import;
#   2. the gates of what the merge adds or touches pass on this box;
#   3. fp32.v1 did not move: no file of the fp32 profile is in the merge's
#      diff, and the fp32 identity check passes;
#   4. the Python tests of the changed modules pass.
#
# One GPU, one job. Prints one verdict line per phase and MERGE CHECK GREEN
# or MERGE CHECK RED at the end.
set -u
cd "$(dirname "$0")/../.." || exit 9
export PATH="$HOME/.pixi/bin:$PATH"
OUT="$PWD/bench/results/lowbit_merged/$(hostname -s)"
rm -rf "$OUT"; mkdir -p "$OUT"
red=0
phase() {  # name, expected (pass|fail), command...
    name=$1; want=$2; shift 2
    "$@" > "$OUT/$name.log" 2>&1; rc=$?
    if [ "$want" = pass ]; then [ $rc -eq 0 ] && v=held || { v=BROKEN; red=1; }
    else [ $rc -ne 0 ] && v=held || { v=BROKEN; red=1; }; fi
    printf 'PHASE %-44s exit=%-3s expected=%-4s %s\n' "$name" "$rc" "$want" "$v"
    [ "$v" = held ] || tail -25 "$OUT/$name.log" | sed 's/^/    | /'
}
{
    echo "tree_head=$(git rev-parse HEAD 2>/dev/null) dirty_files=$(git status --porcelain 2>/dev/null | grep -c .)"
    echo "machine=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null | head -1)"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} | tee "$OUT/run.txt"

phase install-default pass pixi install -e default
phase install-test pass pixi install -e test
export MOJOLEARN_NUMERIC_MODE=identical
phase build-linalg pass sh bindings/build_linalg.sh
# The host family build refuses to replace an existing .so (build_host_family.sh);
# a rerun on this tree met the previous run's output (nvc2-0013). The check
# rebuilds it from the synced source, so the old one is removed first.
rm -f python/mojolearn/host/_mojolearn_linalg_host.so
phase build-linalg-host pass sh bindings/build_linalg_host.sh
# The base binding (_mojolearn.so) is not in the merge's diff, but the loader
# and trainer tests below import it; without it they fail on ImportError
# (nvc2-0012) instead of running. Built here so they really run.
phase build-base pass sh bindings/build.sh
# The loader's block tests (test_models_loader: bfloat16 reads, mamba) need the
# training, transformer and mamba bindings (nvc2-0013, nvc2-0014); the merge changes models/causal_lm.py, so they must run
# rather than fail on ImportError (nvc2-0013).
phase build-training pass sh bindings/build_training.sh
phase build-transformer pass sh bindings/build_transformer.sh
phase build-mamba pass sh bindings/build_mamba.sh

phase gate-gemm-identity pass pixi run check-gemm-identity
phase gate-lowbit pass pixi run check-gemm-lowbit
phase gate-lowbit-sabotage fail pixi run check-gemm-lowbit-sabotage
phase gate-int15 pass pixi run check-gemm-int15
phase gate-int15-sabotage fail pixi run check-gemm-int15-sabotage
phase gate-int15-tuned pass pixi run check-gemm-int15-tuned
phase gate-int8-mma-tuned pass pixi run check-gemm-int8-mma-tuned
phase gate-int8-mma-tuned-sabotage fail pixi run check-gemm-int8-mma-tuned-sabotage
phase gate-int8-pieces-tuned pass pixi run check-gemm-int8-pieces-tuned
phase gate-int8-pieces-tuned-sabotage fail pixi run check-gemm-int8-pieces-tuned-sabotage

tests=""
for t in test_numeric_profile test_linalg_lowbit test_linalg_int15 test_linalg_identity test_host_surface test_models_loader test_training_primitives_surface; do
    [ -f python/mojolearn/tests/$t.py ] && tests="$tests mojolearn/tests/$t.py"
done
phase python-tests pass sh -c "cd python && pixi run -e test python -m pytest -q -rs -p no:cacheprovider $tests"
grep -E 'passed|failed|error|skipped' "$OUT/python-tests.log" | tail -3 | sed 's/^/    | /'
grep -E '^SKIPPED' "$OUT/python-tests.log" | cut -c1-200 | head -12 | sed 's/^/    | /'

echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT/run.txt"
if [ $red -eq 0 ]; then echo "MERGE CHECK GREEN"; else echo "MERGE CHECK RED"; fi
exit $red
