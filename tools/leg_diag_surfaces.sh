#!/usr/bin/env bash
# THE PYTHON SURFACE GATES, ON THE RENTED BOX (2026-09-02).
#
# Runs as tools/e1_bootstrap.sh's phase-9 diag hook (MOJOLEARN_P9_DIAG), after
# that phase has built the bindings for each tier it was given. It gets $OUT
# and $REPO in the environment and writes under $OUT/diag/.
#
# WHY IT EXISTS. Four Python surfaces -- the Mamba blocks, the transformer
# block, ARIMA and the Gaussian process -- have had their gates run on ONE
# APPLE M4 and nowhere else. The lane cards underneath them are cross-vendor;
# the SURFACES are not, and until this script runs on an NVIDIA box and an AMD
# box, "exposed on three vendors" is a sentence nobody has measured. This is
# what turns that sentence into a result.
#
# IT NEVER FAILS THE LEG. Every gate's exit code is RECORDED, not raised: a
# leg that dies on the first red brings home nothing about the other five,
# and the whole point is the matrix. The summary file is the deliverable;
# `PHASE9-FINDING` in the console log is the bootstrap's own alarm.
#
# THE BINDING MUST BE BUILT FOR THE TIER OR THE GATE IS A NO-OP THAT LOOKS
# LIKE A PASS. `_backend.py` raises ImportError BY NAME for a missing tier
# binary, so a REFUSED row here means "P9_BINDINGS did not include this
# binding", never "the surface is broken". Pass the five newer bindings
# explicitly: the phase-9 default (E1_IDENT_BINDINGS) is the TEN older ones
# and does not build arima, training, gp, mamba or transformer.
set -u

OUT="${OUT:?the bootstrap sets this}"
REPO="${REPO:-$(pwd)}"
D="$OUT/diag"
mkdir -p "$D"
SUM="$D/SURFACES.txt"

: > "$SUM"
{
  echo "# Python surface gates on this box"
  echo "# host:   $(uname -n)  $(uname -m)"
  echo "# kernel: $(uname -r)"
  echo "# date:   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "# commit: ${MOJOLEARN_COMMIT:-unknown}"
  echo
} >> "$SUM"

# The GPU, named from the box rather than assumed from the leg's label.
if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi --showproductname 2>/dev/null | head -20 > "$D/gpu.txt" || true
elif command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$D/gpu.txt" 2>/dev/null || true
fi
[ -s "$D/gpu.txt" ] && { echo "# gpu: $(head -1 "$D/gpu.txt")"; echo; } >> "$SUM"

run_gate() {
  tier="$1"; gate="$2"
  log="$D/${tier}_${gate}.log"
  ( cd "$REPO" && PYTHONPATH=python MOJOLEARN_NUMERIC_MODE="$tier" \
      pixi run -e gbmbench python3 -m "mojolearn.tests.$gate" ) > "$log" 2>&1
  rc=$?
  # A missing tier binary is REFUSED, not RED: it means this leg did not
  # build that binding, which is a leg-configuration fact, not a surface fact.
  if grep -q 'is not built; build it with' "$log" 2>/dev/null; then
    verdict="REFUSED (binding not built for this tier)"
  elif [ "$rc" -eq 0 ]; then
    verdict="GREEN"
  else
    verdict="RED (rc=$rc)"
  fi
  printf '%-14s %-28s %s\n' "$tier" "$gate" "$verdict" >> "$SUM"
  printf '  %s %s -> %s\n' "$tier" "$gate" "$verdict"
  # The verdict line the gate printed about itself, which carries the check
  # counts and, in the identical tier, what it asserted rather than reported.
  tail -3 "$log" | sed 's/^/      /'
}

TIERS="${MOJOLEARN_DIAG_TIERS:-fast deterministic identical}"
for tier in $TIERS; do
  echo "== $tier =="
  run_gate "$tier" test_mamba_surface
  run_gate "$tier" test_transformer_surface
  # ARIMA's gate is two tiers by construction (its own header: the mode is
  # frozen at import, and `deterministic` makes it no promise it does not
  # already make under `fast`), so it is not run there.
  case "$tier" in
    deterministic) : ;;
    *) run_gate "$tier" test_arima_surface ;;
  esac
done

echo
echo "== summary =="
cat "$SUM"
