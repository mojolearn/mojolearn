#!/bin/sh
# lane/neural-apple (2026-09-28): one Apple speed request for the neural
# family. Run by `tools/apple_steward.py submit --kind speed` in the steward's
# worktree at the commit, after the builds (bindings/build.sh,
# build_training.sh, build_mamba.sh, build_transformer.sh,
# build_embedding.sh; the byte LM binding is built here when absent). Prints one NEURAL line per lane
# (bench_board_neural's `ours` arm: public API, host inputs in, result back,
# synchronized; median of NEURAL_ROUNDS timed rounds after a warm-up; output
# digests for the before/after bit check), then the byte LM step at the T3
# shard shape with component timing (consecutive steps, no exports between).
#   NEURAL_LANES   lanes to race (default: every GPU lane)
#   NEURAL_ROUNDS  timed rounds per lane (default 5)
#   NEURAL_STEP    0 skips the byte LM component step
#   NEURAL_STEP_SHAPE  B L DM H KV HD FF LAYERS VOCAB (default the T3 shard)
#   NEURAL_CENSUS  1 also builds a byte LM copy under
#                  -D MOJOLEARN_STEP_PHASE_TIMERS=1 (a copy of the package in
#                  $OUT/census) and prints its per-step launch/sync counts
#   NEURAL_CENSUS_DEFINES  the census copy's defines (default
#                  "-D MOJOLEARN_STEP_PHASE_TIMERS=1"; add
#                  "-D MOJOLEARN_ATTN_PHASE_TIMERS=1" for per-kernel attention
#                  timers, printed as COMP attn.<kernel>)
set -u
OUT=${NEURAL_OUT:-$HOME/mojolearn-evidence/neural-apple-speed/$(git rev-parse --short HEAD)-$(date -u +%H%M%S)}
LANES=${NEURAL_LANES:-"lm-train-step lm-forward gemm transformer-forward mamba1-forward mamba2-forward mamba3-forward samba-train-step samba-forward mlp-train-step"}
ROUNDS=${NEURAL_ROUNDS:-5}
mkdir -p "$OUT"
# The byte LM binding is ALWAYS rebuilt at this commit: a steward worktree
# keeps untracked files between requests, so a binding left by an earlier
# request would otherwise be timed as this commit's (it happened: 30497d57e's
# step line timed b11745d8e's binding). build_byte_lm.sh refuses an existing
# output, hence the rm.
rm -f python/mojolearn/identical/_mojolearn_byte_lm.so
pixi run -e default sh bindings/build_byte_lm.sh > "$OUT/byte_lm.build.log" 2>&1 || echo "BYTE-LM-BUILD FAILED"
echo "BYTE-LM $(shasum -a 256 python/mojolearn/identical/_mojolearn_byte_lm.so 2>/dev/null | cut -c1-16) built at $(git rev-parse --short HEAD)"
echo "OUT $OUT commit $(git rev-parse --short HEAD) host $(hostname) $(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
export MOJOLEARN_NUMERIC_MODE=${MOJOLEARN_NUMERIC_MODE:-identical}
# The source tree is the package (the conductor's byte stream and the step
# probe import it; the ours worker gets it from bench_board_neural itself).
PYTHONPATH="$(pwd)/python${PYTHONPATH:+:$PYTHONPATH}"; export PYTHONPATH
# The conductor runs in the skgpu env: the Mamba lanes' weights come from
# mamba/corpus/gen_corpus.py, which imports torch. The ours worker runs in
# the default env (the bindings' env); no torch arm is raced here.
OURS_PY=$(pixi run -e default python -c 'import sys; print(sys.executable)')
pixi install -e skgpu > "$OUT/skgpu.install.log" 2>&1 || echo "SKGPU-INSTALL FAILED"
rc=0
for l in $LANES; do
    t0=$(date +%s)
    pixi run -e skgpu python tools/bench_board_neural.py race --lane "$l" --shape full --arms ours \
        --ours-python "$OURS_PY" --rounds "$ROUNDS" --out "$OUT/race" --work "$OUT/work" > "$OUT/race-$l.log" 2>&1 || rc=1
    grep -E '^NEURAL(-REFUSED)? ' "$OUT/race-$l.log"
    grep -E '^NEURAL-ROUND ' "$OUT/race-$l.log" | awk '{print $4, $5, $6}' | tr '\n' ' '; echo
    echo "LANE-WALL $l $(( $(date +%s) - t0 )) s"
done
summarize() {
    pixi run -e default python - "$1" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print("STEP estash_gate", r.get("attention_estash_gate"))
print("STEP first", r.get("first_call_seconds"), "steady", r.get("steady_step_seconds"), "median", r.get("steady_median_seconds"), "tok/s", r.get("steady_median_tokens_per_second"), "attn", r.get("attention_arm"), "gemm", r.get("gemm_plan"), "glue", r.get("step_glue_arm"))
ct = r.get("component_timing_ms") or {}
print("STEP component total ms", r.get("component_timing_total_ms"), "step_seconds", r.get("step_seconds"), "covered", r.get("component_timing_covered_fraction"), "timed walls", r.get("component_timing_step_seconds_all"))
for k, v in sorted(ct.items(), key=lambda kv: -(kv[1] if isinstance(kv[1], (int, float)) else 0))[:110]:
    print("COMP %-48s %s" % (k, v))
cc = r.get("component_counts") or {}
for k, v in sorted(cc.items(), key=lambda kv: -(kv[1] if isinstance(kv[1], (int, float)) else 0))[:60]:
    print("COUNT %-48s %s" % (k, v))
for k in ("final_witness", "step_witnesses", "component_timing_witnesses"):
    if k in r:
        print("WITNESS", k, json.dumps(r[k])[:800])
PY
}
if [ "${NEURAL_CENSUS:-0}" = 1 ]; then
    rm -rf "$OUT/census"; mkdir -p "$OUT/census"
    cp -R python/mojolearn "$OUT/census/mojolearn"
    rm -f "$OUT/census/mojolearn/identical/_mojolearn_byte_lm.so"
    MOJOLEARN_BYTE_LM_OUTDIR="$OUT/census/mojolearn/identical" \
        MOJOLEARN_BUILD_EXTRA_DEFINES="${NEURAL_CENSUS_DEFINES:--D MOJOLEARN_STEP_PHASE_TIMERS=1}" \
        pixi run -e default sh bindings/build_byte_lm.sh > "$OUT/census.build.log" 2>&1 || echo "CENSUS-BUILD FAILED"
    # shellcheck disable=SC2086
    set -- ${NEURAL_STEP_SHAPE:-4 2048 768 12 12 64 2048 12 50257}
    PYTHONPATH="$OUT/census" pixi run -e default python tools/lm_step_memory_probe.py --out "$OUT/census_step" \
        --shape "$@" --steps 2 --resident-lean --component-timing --component-timing-steps 2 \
        --budget-seconds 3000 > "$OUT/census_step.log" 2>&1 || rc=1
    echo "CENSUS (timers inflate every time below; the counts are exact)"
    summarize "$OUT/census_step/result.json" | sed 's/^/CENSUS /'
fi
if [ "${NEURAL_STEP:-1}" != 0 ]; then
    # shellcheck disable=SC2086
    set -- ${NEURAL_STEP_SHAPE:-4 2048 768 12 12 64 2048 12 50257}
    pixi run -e default python tools/lm_step_memory_probe.py --out "$OUT/step" --shape "$@" \
        --steps 4 --resident-lean --component-timing --component-timing-steps 3 \
        --budget-seconds 3000 > "$OUT/step.log" 2>&1 || rc=1
    grep -v '^ ' "$OUT/step.log" | tail -15
    summarize "$OUT/step/result.json"
fi
exit $rc
