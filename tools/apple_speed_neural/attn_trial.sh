#!/bin/sh
# lane/neural-apple (2026-09-28): ONE attention trial build of the byte LM
# (-D MOJOLEARN_ATTN_ARM_TRIAL=1, into a private package copy) and the lean
# step under each named arm (MOJOLEARN_ATTN_ARM), one after another on one
# Mac, with each arm's step and final witnesses. Every arm is a schedule of
# the same arithmetic (fused_attention.mojo's arm contract), so the witnesses
# must be equal; the times say which schedule suits this GPU.
#   TRIAL_ARMS   space separated arm names; "-" = the build default
#   TRIAL_SHAPE  B L DM H KV HD FF LAYERS VOCAB (default the T3 shard)
#   TRIAL_STEPS  steps per arm (default 4: one setup + three steady)
set -u
OUT=${TRIAL_OUT:-$HOME/mojolearn-evidence/neural-apple-speed/trial-$(git rev-parse --short HEAD)-$(date -u +%H%M%S)}
mkdir -p "$OUT"
export MOJOLEARN_NUMERIC_MODE=identical
echo "TRIAL OUT $OUT commit $(git rev-parse --short HEAD) host $(sysctl -n machdep.cpu.brand_string 2>/dev/null) mem $(sysctl -n hw.memsize 2>/dev/null)"
# shellcheck disable=SC2086
set -- ${TRIAL_SHAPE:-4 2048 768 12 12 64 2048 12 50257}
SHAPE="$*"
rm -rf "$OUT/pkg"; mkdir -p "$OUT/pkg"
cp -R python/mojolearn "$OUT/pkg/mojolearn"
rm -f "$OUT/pkg/mojolearn/identical/_mojolearn_byte_lm.so"
t0=$(date +%s)
MOJOLEARN_BYTE_LM_OUTDIR="$OUT/pkg/mojolearn/identical" MOJOLEARN_BUILD_EXTRA_DEFINES="-D MOJOLEARN_ATTN_ARM_TRIAL=1" \
    pixi run -e default sh bindings/build_byte_lm.sh > "$OUT/build.log" 2>&1 || { echo "TRIAL BUILD FAILED"; tail -20 "$OUT/build.log"; exit 1; }
echo "TRIAL BUILD $(( $(date +%s) - t0 )) s"
rc=0
for arm in ${TRIAL_ARMS:-- stash_tiled_ztiled_r32_fgrid_r32_qres_pf stash_tiled_ztiled_r64_fgrid_r32_qres_pf stash_tiled_fgrid_r32_qres_pf_zdefer stash_tiled_fgrid_r32_qres_pf_zlag stash_tiled_fgrid_r32_qres_pf_kvgrid_r32 stash_tiled_fgrid_r32_qres_pf_kvrecompute stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32 stash_tiled_fgrid_r32_qres_pf_estash_dres_kvgrid_r32_bswz}; do
    d="$OUT/arm-$arm"; rm -rf "$d"; mkdir -p "$d"
    if [ "$arm" = "-" ]; then unset MOJOLEARN_ATTN_ARM; else MOJOLEARN_ATTN_ARM=$arm; export MOJOLEARN_ATTN_ARM; fi
    # shellcheck disable=SC2086
    PYTHONPATH="$OUT/pkg" pixi run -e default python tools/lm_step_memory_probe.py --out "$d/step" --shape $SHAPE \
        --steps "${TRIAL_STEPS:-4}" --resident-lean --budget-seconds 3000 > "$d/step.log" 2>&1 || rc=1
    pixi run -e default python - "$d/step/result.json" "$arm" <<'PY'
import hashlib, json, sys
try:
    r = json.load(open(sys.argv[1]))
except Exception as e:
    print("TRIAL-ARM", sys.argv[2], "NO RESULT", e); sys.exit(0)
fw = json.dumps((r.get("final_witness") or {}).get("sha256"), sort_keys=True)
sw = json.dumps([s.get("sha256") for s in r.get("step_witnesses") or []], sort_keys=True)
print("TRIAL-ARM %s ran=%s steady=%s median=%s first=%s maxrss=%s final=%s steps=%s" % (
    sys.argv[2], r.get("attention_arm"), [round(x, 4) for x in r.get("steady_step_seconds") or []],
    r.get("steady_median_seconds"), r.get("first_call_seconds"), r.get("process_ru_maxrss_bytes"),
    hashlib.sha256(fw.encode()).hexdigest()[:16], hashlib.sha256(sw.encode()).hexdigest()[:16]))
PY
    grep -iE "refus|error|infinity|Traceback" "$d/step.log" | head -3
done
unset MOJOLEARN_ATTN_ARM
exit $rc
