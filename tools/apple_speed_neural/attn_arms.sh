#!/bin/sh
# lane/neural-apple (2026-09-28): the byte LM step under each attention
# default word a no-trial build can select by define (checks/kernel_matrix_attn.mojo
# attn_default_arm_for's check knobs), on one Mac, one after another, with the
# lean step's final witness per arm (the bits must be equal: every word is a
# schedule, never a result). Each arm builds its own byte LM copy into a
# private package copy, so the worktree's binding is untouched.
#   ARMS_DEFINES  space separated knob names (default: the Apple default, then
#                 R3/KVGRID/ESTASH/BSWZ every-column words); "-" = no define;
#                 "A+B" = both knobs in one arm
#   ARMS_SHAPE    B L DM H KV HD FF LAYERS VOCAB (default 1 x 2048, 12 layers)
#   ARMS_STEPS    steps per arm (default 5: one setup + four steady)
set -u
OUT=${ARMS_OUT:-$HOME/mojolearn-evidence/neural-apple-speed/arms-$(git rev-parse --short HEAD)-$(date -u +%H%M%S)}
mkdir -p "$OUT"
PYTHONPATH="$(pwd)/python${PYTHONPATH:+:$PYTHONPATH}"; export PYTHONPATH
export MOJOLEARN_NUMERIC_MODE=identical
echo "ARMS OUT $OUT commit $(git rev-parse --short HEAD) host $(sysctl -n machdep.cpu.brand_string 2>/dev/null) mem $(sysctl -n hw.memsize 2>/dev/null)"
# shellcheck disable=SC2086
set -- ${ARMS_SHAPE:-1 2048 768 12 12 64 2048 12 50257}
SHAPE="$*"
rc=0
for arm in ${ARMS_DEFINES:-- MOJOLEARN_ATTN_DEFAULT_KVGRID_EVERY_COLUMN MOJOLEARN_ATTN_DEFAULT_ESTASH_EVERY_COLUMN MOJOLEARN_ATTN_DEFAULT_BSWZ_EVERY_COLUMN}; do
    d="$OUT/$arm"; rm -rf "$d"; mkdir -p "$d/pkg"
    cp -R python/mojolearn "$d/pkg/mojolearn"
    rm -f "$d/pkg/mojolearn/identical/_mojolearn_byte_lm.so"
    # "A+B" builds one arm under both knobs (lane/neural-apple2)
    defs=""; [ "$arm" != "-" ] && for kn in $(echo "$arm" | tr '+' ' '); do defs="$defs -D $kn=1"; done
    MOJOLEARN_BYTE_LM_OUTDIR="$d/pkg/mojolearn/identical" MOJOLEARN_BUILD_EXTRA_DEFINES="$defs" \
        pixi run -e default sh bindings/build_byte_lm.sh > "$d/build.log" 2>&1 || { echo "ARM $arm BUILD FAILED"; rc=1; continue; }
    # shellcheck disable=SC2086
    PYTHONPATH="$d/pkg" pixi run -e default python tools/lm_step_memory_probe.py --out "$d/step" --shape $SHAPE \
        --steps "${ARMS_STEPS:-5}" --resident-lean --budget-seconds 3000 > "$d/step.log" 2>&1 || rc=1
    pixi run -e default python - "$d/step/result.json" "$arm" <<'PY'
import hashlib, json, sys
try:
    r = json.load(open(sys.argv[1]))
except Exception as e:
    print("ARM", sys.argv[2], "NO RESULT", e); sys.exit(0)
fw = json.dumps(r.get("final_witness"), sort_keys=True)
sw = json.dumps([s.get("sha256") for s in r.get("step_witnesses") or []], sort_keys=True)
print("ARM %s attn=%s steady=%s median=%s first=%s peak_rss=%s final_witness=%s step_witnesses=%s gate=%s" % (
    sys.argv[2], r.get("attention_arm"), [round(x, 4) for x in r.get("steady_step_seconds") or []],
    r.get("steady_median_seconds"), r.get("first_call_seconds"), r.get("process_ru_maxrss_bytes"),
    hashlib.sha256(fw.encode()).hexdigest()[:16], hashlib.sha256(sw.encode()).hexdigest()[:16],
    r.get("attention_estash_gate")))
PY
    grep -iE "refus|error|infinity" "$d/step.log" | head -3
done
exit $rc
