# DEVIATION 2712/2713 gate on a GPU box: build the production Mamba binding
# for this box, then run the poison gate (mode 0) and its band control
# (modes 2 and 3). MOJOLEARN_COMMIT may be absent (RunPod passes no env):
# the wrapper bakes commit.txt.
set -u
cd "${MOJOLEARN_LEG_TREE:-/root/mojolearn}" 2>/dev/null || cd "$(dirname "$0")"
OUT="${MOJOLEARN_LEG_OUT:-/root/gemm_leg_out}"; mkdir -p "$OUT"
[ -n "${MOJOLEARN_COMMIT:-}" ] && echo "$MOJOLEARN_COMMIT" > commit.txt
[ -f commit.txt ] && cat commit.txt
LABEL="${MOJOLEARN_BOX_LABEL:?box label}"
# the shipped archive carries no bench/results: fetch the record's three
# vendor columns from the public repository (curl on the raw endpoint; the
# Hot Aisle ROCm container has no git)
REC=bench/results/identity_break/2026-09-14_120-lanes-2711flip
RAW=https://raw.githubusercontent.com/mojolearn/mojolearn/main/$REC
mkdir -p "$REC"
for c in apple-m4 nvidia-h100-sm_90a amd-mi300x-gfx942; do
  [ -s "$REC/$c.json" ] && continue
  curl -fsSL --retry 3 -o "$REC/$c.json" "$RAW/$c.json" || echo "record fetch FAILED for $c (curl)"
done
ls -la "$REC" | tail -3
export MOJOLEARN_POISON_RECORD="$REC"
echo "== build the base binding (the Mamba lanes import all_finite_f32 from it)"; env -u MOJOLEARN_TARGET_COLUMN MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build.sh 2>&1 | grep -v "warning:\|^ *\^\|^    var\|^Imported" | tail -3
echo "== build the production mamba binding"; env -u MOJOLEARN_TARGET_COLUMN MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/build_mamba.sh 2>&1 | grep -v "warning:\|^ *\^\|^    var\|^Imported" | tail -3
ls python/mojolearn/identical/
for m in 0 1 2 3; do
  echo "== poison gate mode $m"
  env MOJOLEARN_MAMBA_POISON_SABOTAGE=$m MOJOLEARN_BOX_LABEL="$LABEL-poison$m" MOJOLEARN_POISON_KEEP=1 sh tools/mamba_poison_gate.sh > "$OUT/poison_gate_$m.log" 2>&1
  echo "exit $?" >> "$OUT/poison_gate_$m.log"
  grep -E "poison gate: defines|cells=|poison column|SABOTAGE|PASSED|FAILED|^require|exit " "$OUT/poison_gate_$m.log" | grep -v "^| mamba" | cut -c1-200
  W=$(grep -o "/[^ ]*mojolearn-poison\.[A-Za-z0-9]*" "$OUT/poison_gate_$m.log" | head -1); [ -n "$W" ] && cp "$W/poison.json" "$OUT/poison_mode$m.json" 2>/dev/null && cp "$W/diff.txt" "$OUT/diff_mode$m.txt" 2>/dev/null
  rm -rf "$W"
done
exit 0
