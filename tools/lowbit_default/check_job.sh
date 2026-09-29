#!/bin/bash
# tools/lowbit_default/check_job.sh -- THE ONE LIGHT CHECK of lane/lowbit-default
# (the inference default becomes fixed15_v1 for the transformer models) before
# it goes to main (2026-09-29). It is tools/lowbit_merged/check_job.sh plus
# what the flip needs: the byte LM and host bindings the trainer and CPU-route
# tests import, the transformer block gates, every trainer's and every model
# family's tests, the verifier's causal-LM lanes against the reference table
# (pinned to fp32_v1), and SmolLM2-360M's hashes and generate times
# (tools/lowbit_default/default_gate.py). The original header follows.
#
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
OUT="$PWD/bench/results/lowbit_default/$(hostname -s)"
MODEL=""
for d in "${LB_MODEL:-}" "$HOME/models/SmolLM2-360M" /root/models/SmolLM2-360M; do
    [ -n "$d" ] && [ -f "$d/model.safetensors" ] && { MODEL=$d; break; }
done
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
# lane/lowbit-default: the byte LM trainer's binding (one explicit target off
# Apple) and the host bindings the CPU route (device="cpu") and the host
# trainer import. The host builds refuse an existing output.
ARCH=""
command -v nvidia-smi > /dev/null 2>&1 && ARCH=sm_$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')
rm -f python/mojolearn/identical/_mojolearn_byte_lm.so
phase build-byte-lm pass env MOJOLEARN_GPU_ARCHS=$ARCH sh bindings/build_byte_lm.sh
# the portable math library SambaStack's optimizer dlopens (the tree's own
# recipe, as tools/gap_column_leg.sh builds it); without it the Samba tests skip
phase build-portable-math pass env PYTHONPATH=$PWD/packaging/portable_math pixi run python -c "import pathlib, stage; stage.build(pathlib.Path('$PWD/python/mojolearn/.libs/libMojolearnMath.so'))"
# THE WHOLE SUITE NEEDS EVERY BINDING (orchestrator, 2026-09-29): every
# other IDENTICAL GPU binding and every host binding, four at a time, each
# with its own log; one phase line per binding that fails.
build_one() {  # script
    b=$(basename "$1" .sh); log="$OUT/all_$b.log"
    case "$b" in
        *_host) f=${b#build_}; rm -f "python/mojolearn/host/_mojolearn_${f}.so"
                env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_TARGET_COLUMN=cpu MOJOLEARN_SKIP_BUILD_GATE=1 sh "$1" > "$log" 2>&1 ;;
        *)      env MOJOLEARN_GPU_ARCHS=$ARCH MOJOLEARN_SKIP_BUILD_GATE=1 sh "$1" > "$log" 2>&1 ;;
    esac
    echo "$b $?"
}
export -f build_one; export OUT ARCH
ls bindings/build_*.sh | grep -v build_host_family.sh \
    | grep -vE '/build(_linalg|_linalg_host|_training|_transformer|_mamba|_byte_lm|_core_host|_neural_host|_transformer_host|_mamba_host|_training_host|_tokenizer_host|_byte_lm_host)?\.sh$' \
    | xargs -P 4 -I{} bash -c 'build_one {}' > "$OUT/build_all.tsv"
while read -r b rc; do
    if [ "$rc" = 0 ]; then :; else printf 'PHASE %-44s exit=%-3s expected=pass BROKEN\n' "all-$b" "$rc"; red=1; tail -15 "$OUT/all_$b.log" | sed 's/^/    | /'; fi
done < "$OUT/build_all.tsv"
echo "built $(awk '$2==0' "$OUT/build_all.tsv" | wc -l) of $(wc -l < "$OUT/build_all.tsv") further bindings"
for f in core linalg neural transformer mamba training tokenizer byte_lm; do
    [ "$f" = linalg ] && continue  # built above
    rm -f "python/mojolearn/host/_mojolearn_${f}_host.so"
    phase build-${f}-host pass env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_TARGET_COLUMN=cpu sh bindings/build_${f}_host.sh
done

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
# the block under both profiles (the default's block path) and fp32.v1's block check
phase gate-transformer pass pixi run check-transformer
phase gate-transformer-int15 pass pixi run check-transformer-int15
phase gate-transformer-int15-sabotage fail pixi run check-transformer-int15-sabotage

# THE WHOLE SUITE, not a list (orchestrator, 2026-09-29)
phase python-tests pass sh -c "cd python && pixi run -e test python -m pytest -q -rsfE -p no:cacheprovider mojolearn/tests"
grep -E 'passed|failed|error|skipped' "$OUT/python-tests.log" | tail -3 | sed 's/^/    | /'
grep -E '^SKIPPED' "$OUT/python-tests.log" | cut -c1-300 | sed 's/^/    | /'
grep -E '^(FAILED|ERROR)' "$OUT/python-tests.log" | cut -c1-300 | sed 's/^/    | /'
# the default lane's own tests must RUN here, not skip
phase profile-tests-ran fail grep -E '^SKIPPED.*test_numeric_profile' "$OUT/python-tests.log"

# the verifier's causal-LM lanes against the reference table (fp32_v1 by name
# in the harness), under the new default and under the escape hatch
# (one device: the reference is the one-device record)
unset MOJOLEARN_PAR_DEVICES
# (the `verify` driver's comparator self-test needs bindings this check does
# not build, nvc2-0029; the harness itself runs the lane and the table is read here)
TABLE=python/mojolearn/verify_reference/table.json
phase verify-causal-lm-default pass sh -c "PYTHONPATH=$PWD/python pixi run -e test python tools/identity_break.py --lanes hf-causal-lm --repeats 2 --json $OUT/ib_default.json > $OUT/ib_default.run 2>&1; pixi run -e test python tools/lowbit_default/causal_lm_refs.py $OUT/ib_default.json $TABLE"
phase verify-causal-lm-fp32env pass sh -c "MOJOLEARN_NUMERIC_PROFILE=fp32_v1 PYTHONPATH=$PWD/python pixi run -e test python tools/identity_break.py --lanes hf-causal-lm --repeats 2 --json $OUT/ib_fp32.json > $OUT/ib_fp32.run 2>&1; pixi run -e test python tools/lowbit_default/causal_lm_refs.py $OUT/ib_fp32.json $TABLE"
grep -E 'compared|DIFFERENT|NOT HASHED' "$OUT/verify-causal-lm-default.log" | tail -4 | sed 's/^/    | /'

# SmolLM2-360M: the hashes under the default and the hatches, and generate
if [ -n "$MODEL" ]; then
    phase smollm2-default-gate pass sh -c "pixi run -e default python tools/lowbit_default/default_gate.py --model $MODEL --phases hash,generate --out $OUT"
    grep -E 'RESULT|GATE' "$OUT/smollm2-default-gate.log" | sed 's/^/    | /'
    phase smollm2-resident-gate pass sh -c "pixi run -e default python tools/lowbit_default/resident_gate.py --model $MODEL"
    grep -E 'RESULT|GATE' "$OUT/smollm2-resident-gate.log" | sed 's/^/    | /'
else
    echo "PHASE smollm2-default-gate BROKEN: no staged SmolLM2-360M"; red=1
fi

echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$OUT/run.txt"
if [ $red -eq 0 ]; then echo "MERGE CHECK GREEN"; else echo "MERGE CHECK RED"; fi
exit $red
