#!/bin/sh
# lane/neural-apple (2026-09-28): an alternating A/B of public neural lanes
# between commits on ONE Mac. Each variant is a `git worktree` at its commit
# with only AB_BUILDS built there; the races alternate variant by variant,
# AB_REPS times, so drift hits both arms alike. The conductor is this tree's
# tools/bench_board_neural.py; the ours worker imports the variant's package
# (MOJOLEARN_BENCH_INSTALLED=1, PYTHONPATH = the variant's python/).
#   AB_VARIANTS  space separated label=commit
#   AB_BUILDS    space separated bindings/build_*.sh (default build.sh + mamba)
#   AB_LANES     space separated bench_board_neural lanes
#   AB_REPS      alternations (default 3); AB_ROUNDS rounds per race (default 5)
set -u
OUT=${AB_OUT:-$HOME/mojolearn-evidence/neural-apple-speed/ab-$(git rev-parse --short HEAD)-$(date -u +%H%M%S)}
mkdir -p "$OUT"
echo "AB OUT $OUT host $(sysctl -n machdep.cpu.brand_string 2>/dev/null)"
export MOJOLEARN_NUMERIC_MODE=identical
OURS_PY=$(pixi run -e default python -c 'import sys; print(sys.executable)')
pixi install -e skgpu > "$OUT/skgpu.install.log" 2>&1 || echo "SKGPU-INSTALL FAILED"
labels=""
for v in $AB_VARIANTS; do
    label=${v%%=*}; ref=${v#*=}
    wt="$OUT/wt-$label"
    git worktree add --detach "$wt" "$ref" > "$OUT/wt-$label.log" 2>&1 || { echo "AB $label WORKTREE FAILED"; continue; }
    # the same pixi.toml/lock at every variant (checked): share this tree's env
    if cmp -s pixi.lock "$wt/pixi.lock"; then ln -s "$(pwd)/.pixi" "$wt/.pixi"; else echo "AB $label pixi.lock differs; it installs its own env"; fi
    ok=1
    for s in ${AB_BUILDS:-bindings/build.sh bindings/build_mamba.sh}; do
        (cd "$wt" && MOJOLEARN_NUMERIC_MODE=identical pixi run -e default sh "$s") >> "$OUT/build-$label.log" 2>&1 || { echo "AB $label BUILD $s FAILED"; ok=0; }
    done
    [ $ok = 1 ] && labels="$labels $label"
done
for rep in $(seq 1 "${AB_REPS:-3}"); do
    for label in $labels; do
        for l in $AB_LANES; do
            MOJOLEARN_BENCH_INSTALLED=1 PYTHONPATH="$OUT/wt-$label/python" pixi run -e skgpu python tools/bench_board_neural.py race \
                --lane "$l" --shape full --arms ours --ours-python "$OURS_PY" --rounds "${AB_ROUNDS:-5}" \
                --out "$OUT/race-$label-$rep" --work "$OUT/work" > "$OUT/race-$label-$rep-$l.log" 2>&1
            m=$(grep -E '^NEURAL lane=' "$OUT/race-$label-$rep-$l.log" | sed 's/.*median_ms=\([0-9.]*\).*/\1/')
            dg=$(grep -E '^NEURAL-ROUND ' "$OUT/race-$label-$rep-$l.log" | tail -1 | sed 's/.*digest=//')
            q=$(grep -E '^NEURAL lane=' "$OUT/race-$label-$rep-$l.log" | sed 's/.*quality=//')
            echo "AB rep=$rep variant=$label lane=$l median_ms=$m digest=$dg quality=$q"
        done
    done
done
# AB_STEP_SHAPE (lane/neural-apple2): the byte LM step at this shape
# (B L DM H KV HD FF LAYERS VOCAB) in every variant, alternating AB_STEP_REPS
# times, each its own binding built in the variant's tree; steady median and
# the final witness / loss digests.
if [ -n "${AB_STEP_SHAPE:-}" ]; then
    for label in $labels; do
        rm -f "$OUT/wt-$label/python/mojolearn/identical/_mojolearn_byte_lm.so"
        (cd "$OUT/wt-$label" && MOJOLEARN_NUMERIC_MODE=identical pixi run -e default sh bindings/build_byte_lm.sh) > "$OUT/bytelm-$label.log" 2>&1 || echo "AB $label BYTE-LM BUILD FAILED"
    done
    # AB_STEP_SHAPE may name several shapes separated by ';'.
    echo "$AB_STEP_SHAPE" | tr ';' '\n' > "$OUT/step_shapes.txt"
    si=0
    while read -r shp; do
    [ -n "$shp" ] || continue
    si=$((si + 1))
    echo "AB-STEP shape $si = $shp"
    for rep in $(seq 1 "${AB_STEP_REPS:-2}"); do
        for label in $labels; do
            d="$OUT/step$si-$label-$rep"
            # shellcheck disable=SC2086
            (cd "$OUT/wt-$label" && PYTHONPATH="$OUT/wt-$label/python" pixi run -e default python tools/lm_step_memory_probe.py --out "$d" \
                --shape $shp --steps "${AB_STEP_STEPS:-4}" --resident-lean --budget-seconds 3000) > "$d.log" 2>&1 < /dev/null
            pixi run -e default python - "$d/result.json" "$label" "$rep.s$si" <<'PY'
import json, sys
try:
    r = json.load(open(sys.argv[1]))
except Exception as e:
    print("AB-STEP", sys.argv[2], sys.argv[3], "NO RESULT", e); sys.exit(0)
fw = (r.get("final_witness") or {}).get("sha256") or {}
lw = [(s.get("sha256") or {}).get("loss", "")[:8] for s in r.get("step_witnesses") or []]
print("AB-STEP rep=%s variant=%s median=%s steady=%s gate=%s grad=%s param=%s m=%s v=%s losses=%s" % (
    sys.argv[3], sys.argv[2], r.get("steady_median_seconds"),
    [round(x, 4) for x in r.get("steady_step_seconds") or []], r.get("attention_estash_gate"),
    fw.get("gradients", "")[:16], fw.get("parameters", "")[:16], fw.get("m", "")[:16], fw.get("v", "")[:16], lw))
PY
        done
    done
    done < "$OUT/step_shapes.txt"
fi
for v in $AB_VARIANTS; do git worktree remove --force "$OUT/wt-${v%%=*}" > /dev/null 2>&1; done
exit 0
