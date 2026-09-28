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
set -u
OUT=${NEURAL_OUT:-$HOME/mojolearn-evidence/neural-apple-speed/$(git rev-parse --short HEAD)-$(date -u +%H%M%S)}
LANES=${NEURAL_LANES:-"lm-train-step lm-forward gemm transformer-forward mamba1-forward mamba2-forward mamba3-forward samba-train-step samba-forward mlp-train-step"}
ROUNDS=${NEURAL_ROUNDS:-5}
mkdir -p "$OUT"
# build_byte_lm.sh refuses an existing output; the steward seeds it from its
# build store when the source closure is unchanged, so build only when absent.
[ -f python/mojolearn/identical/_mojolearn_byte_lm.so ] || pixi run -e default sh bindings/build_byte_lm.sh > "$OUT/byte_lm.build.log" 2>&1 || echo "BYTE-LM-BUILD FAILED"
echo "OUT $OUT commit $(git rev-parse --short HEAD) host $(hostname) $(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
export MOJOLEARN_NUMERIC_MODE=${MOJOLEARN_NUMERIC_MODE:-identical}
rc=0
for l in $LANES; do
    t0=$(date +%s)
    pixi run -e default python tools/bench_board_neural.py race --lane "$l" --shape full --arms ours \
        --rounds "$ROUNDS" --out "$OUT/race" --work "$OUT/work" > "$OUT/race-$l.log" 2>&1 || rc=1
    grep -E '^NEURAL(-REFUSED)? ' "$OUT/race-$l.log"
    grep -E '^NEURAL-ROUND ' "$OUT/race-$l.log" | awk '{print $4, $5, $6}' | tr '\n' ' '; echo
    echo "LANE-WALL $l $(( $(date +%s) - t0 )) s"
done
if [ "${NEURAL_STEP:-1}" != 0 ]; then
    # shellcheck disable=SC2086
    set -- ${NEURAL_STEP_SHAPE:-4 2048 768 12 12 64 2048 12 50257}
    pixi run -e default python tools/lm_step_memory_probe.py --out "$OUT/step" --shape "$@" \
        --steps 4 --resident-lean --component-timing --component-timing-steps 3 \
        --budget-seconds 3000 > "$OUT/step.log" 2>&1 || rc=1
    tail -3 "$OUT/step.log"
    pixi run -e default python - "$OUT/step/result.json" <<'PY'
import json, sys
r = json.load(open(sys.argv[1]))
print("STEP first", r.get("first_call_seconds"), "steady", r.get("steady_step_seconds"), "median", r.get("steady_median_seconds"), "tok/s", r.get("steady_median_tokens_per_second"), "attn", r.get("attention_arm"), "gemm", r.get("gemm_plan"), "glue", r.get("step_glue_arm"))
ct = r.get("component_timing_ms") or {}
tot = r.get("component_timing_total_ms")
print("STEP component total ms", tot, "step_seconds", r.get("step_seconds"), "covered", r.get("component_timing_covered_fraction"), "timed walls", r.get("component_timing_step_seconds_all"))
for k, v in sorted(ct.items(), key=lambda kv: -(kv[1] if isinstance(kv[1], (int, float)) else 0))[:45]:
    print("COMP %-48s %s" % (k, v))
for k in ("final_witness", "step_witnesses"):
    if k in r:
        print("WITNESS", k, json.dumps(r[k])[:600])
PY
fi
exit $rc
