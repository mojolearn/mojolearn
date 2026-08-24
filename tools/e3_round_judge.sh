#!/usr/bin/env bash
# E3: judge one whole-library vendor round in one command, so the verdict
# is reproducible from the artifacts and not from a session's memory.
#
#   bash tools/e3_round_judge.sh <mac_ref_dir> <nv_dir> [<amd_dir>] [--write]
#
# Each directory is one machine's `tools/e1_bootstrap.sh` output
# (bench/results/e1/<stamp>-<host>/). The first is the REFERENCE column
# (Apple). This prints, in order:
#
#   1. commits -- every directory must record the SAME commit
#   2. the tree matrix (e2_cells.json + e1_fits.json + e2_mojo_cards.json)
#      through tools/e2_matrix_diff.py
#   3. the unsupervised matrix (e2u/e2u_cells.json) through the same differ
#   4. the E1U cards (e1u/{kmeans,knn,dbscan}.card) through
#      tools/identity_trace_diff.py, stage by stage
#   5. the phase-1/6 gate lines from each bootstrap.log: every OK counted,
#      every FAIL/Error/FINDING printed verbatim
#   6. train-here-infer-there: the box's cross_infer_mac_models_on_box.json
#      (Mac models predicted on the box), and the other direction run HERE
#      under IDENTICAL mode (box models predicted on this Mac)
#   7. the classical lanes' cards (bootstrap phase 8: gemm, cd, kde,
#      linkage, svm, metrics): IDENTICAL cards Apple vs each box through
#      identity_trace_diff (judged); FAST cards recorded; phase-8 findings
#
# Exit code is 0 only when 2, 3 and 4 are all IDENTICAL/REFUSED= and 5
# shows no FAIL lines. With --write the tables land as
# <mac_ref_dir>/e3_verdicts_<label>.md for E3_RESULTS.md to include.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

WRITE=0
DIRS=()
for a in "$@"; do
  case "$a" in --write) WRITE=1 ;; *) DIRS+=("$a") ;; esac
done
[ ${#DIRS[@]} -ge 2 ] || { sed -n 2,30p "$0"; exit 2; }

label_of() {
  case "$(basename "$1")" in
    *-nv*) echo NVIDIA ;; *-amd*) echo AMD ;; *MacBook*|*Mac*) echo APPLE ;;
    *) basename "$1" ;;
  esac
}
REF="${DIRS[0]}"
RC=0
ARGS=()
for d in "${DIRS[@]}"; do ARGS+=("$d:$(label_of "$d")"); done

echo "########## 1. commits"
ref_commit="$(cat "$REF/commit.txt" 2>/dev/null | head -1)"
for d in "${DIRS[@]}"; do
  c="$(cat "$d/commit.txt" 2>/dev/null | head -1)"
  echo "  $(label_of "$d"): ${c:-MISSING}  $(grep -m1 -i 'gpu\|device' "$d/environment.txt" 2>/dev/null | cut -c1-80)"
  [ "$c" = "$ref_commit" ] || { echo "  !! commit differs from the reference"; RC=1; }
done

echo "########## 2. the tree matrix (E2 + E1 four + Mojo-only cards)"
python3 tools/e2_matrix_diff.py "${ARGS[@]}" > /tmp/e3_trees.md; r=$?
tail -n "$((${#DIRS[@]}))" /tmp/e3_trees.md
grep -E '\*\*' /tmp/e3_trees.md | head -40
[ $r = 0 ] || RC=1

echo "########## 3. the unsupervised matrix (E2U)"
UARGS=()
for d in "${DIRS[@]}"; do
  if [ -d "$d/e2u" ]; then UARGS+=("$d/e2u:$(label_of "$d")"); else echo "  $(label_of "$d"): NO e2u directory"; RC=1; fi
done
if [ ${#UARGS[@]} -ge 2 ]; then
  python3 tools/e2_matrix_diff.py "${UARGS[@]}" > /tmp/e3_e2u.md; r=$?
  tail -n "$((${#DIRS[@]}))" /tmp/e3_e2u.md
  grep -E '\*\*' /tmp/e3_e2u.md | head -40
  [ $r = 0 ] || RC=1
fi

echo "########## 4. the E1U cards, stage by stage"
for arm in kmeans knn dbscan; do
  for d in "${DIRS[@]:1}"; do
    a="$REF/e1u/$arm.card"; b="$d/e1u/$arm.card"
    if [ ! -f "$a" ] || [ ! -f "$b" ]; then echo "  $arm APPLE vs $(label_of "$d"): card MISSING"; RC=1; continue; fi
    out="$(python3 tools/identity_trace_diff.py "$a" "$b" 2>&1)"; r=$?
    n="$(grep -vc '^#\|^$' "$a")"
    if [ $r = 0 ]; then echo "  $arm APPLE vs $(label_of "$d"): IDENTICAL ($n stages)"
    else echo "  $arm APPLE vs $(label_of "$d"): DIVERGENT"; echo "$out" | head -6 | sed 's/^/      /'; RC=1; fi
  done
done

echo "########## 5. gate lines per bootstrap.log"
for d in "${DIRS[@]}"; do
  log="$d/bootstrap.log"
  [ -f "$log" ] || { echo "  $(label_of "$d"): no bootstrap.log"; continue; }
  oks=$(grep -cE ' OK\b|OK:' "$log")
  phases=$(grep -c '^=== phase' "$log")
  echo "  $(label_of "$d"): $phases phase markers, $oks OK lines"
  grep -nE 'FINDING|FAIL|Unhandled exception|Traceback|error:' "$log" | grep -v 'warning' | head -12 | sed 's/^/      /'
  grep -nE 'signed-zero arm hash|signed-zero arm:|contraction: a\*b\+c is|ieee arith check OK' "$log" | sed 's/^/      /'
  if grep -qE 'FAIL|Unhandled exception|Traceback' "$log"; then RC=1; fi
done

echo "########## 6. train-here-infer-there"
for d in "${DIRS[@]:1}"; do
  j="$d/cross_infer_mac_models_on_box.json"
  if [ -f "$j" ]; then
    python3 - "$j" "$(label_of "$d")" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); lab = sys.argv[2]
e2 = d.get("e2", {})
ok = sum(1 for v in e2.values() if v.get("match"))
bad = [k for k, v in e2.items() if not v.get("match")]
print(f"  Mac models on {lab}: {ok}/{len(e2)} E2 cells match" + (f"; MISMATCH {bad[:8]}" if bad else ""))
PY
  else
    echo "  Mac models on $(label_of "$d"): no cross_infer file"
  fi
  # the other direction, run HERE: box models predicted on this Mac, IDENTICAL
  out="$REF/cross_infer_$(label_of "$d" | tr 'A-Z' 'a-z')_models_on_mac.json"
  if [ ! -f "$out" ]; then
    MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python pixi run -e gbmbench python3 tools/e1_cross_infer.py "$d" "$out" > /tmp/e3_xi.log 2>&1 \
      || { echo "  $(label_of "$d") models on Mac: cross-infer FAILED (/tmp/e3_xi.log)"; RC=1; }
  fi
  [ -f "$out" ] && python3 - "$out" "$(label_of "$d")" <<'PY'
import json, sys
d = json.load(open(sys.argv[1])); lab = sys.argv[2]
e2 = d.get("e2", {})
ok = sum(1 for v in e2.values() if v.get("match"))
bad = [k for k, v in e2.items() if not v.get("match")]
print(f"  {lab} models on Mac: {ok}/{len(e2)} E2 cells match" + (f"; MISMATCH {bad[:8]}" if bad else ""))
PY
done

echo "########## 7. the lanes' cards (phase 8): IDENTICAL judged, FAST recorded"
# `mamba` joined this list 2026-08-23. It is NOT a classical lane: it is the
# Mamba-1 block under profile mojolearn.identical.mamba1.fp32.v1, and its FAST
# arm has never been built, so a missing fast card for it is expected.
for lane in gemm cd kde linkage svm metrics mamba; do
  for d in "${DIRS[@]:1}"; do
    a="$REF/lanes/$lane.identical.card"; b="$d/lanes/$lane.identical.card"
    if [ ! -f "$a" ] || [ ! -f "$b" ]; then echo "  $lane IDENTICAL APPLE vs $(label_of "$d"): card MISSING ($([ -f "$a" ] || echo ref)$([ -f "$b" ] || echo ' other'))"; RC=1; continue; fi
    out="$(python3 tools/identity_trace_diff.py "$a" "$b" 2>&1)"; r=$?
    n="$(grep -vc '^#\|^$' "$a")"
    if [ $r = 0 ]; then echo "  $lane IDENTICAL APPLE vs $(label_of "$d"): IDENTICAL ($n stages)"
    else echo "  $lane IDENTICAL APPLE vs $(label_of "$d"): DIVERGENT"; echo "$out" | grep -E 'FIRST DIVERGENCE|A: seq|B: seq|record counts' | head -4 | sed 's/^/      /'; RC=1; fi
    fa="$REF/lanes/$lane.fast.card"; fb="$d/lanes/$lane.fast.card"
    if [ -f "$fa" ] && [ -f "$fb" ]; then
      if python3 tools/identity_trace_diff.py "$fa" "$fb" >/dev/null 2>&1; then echo "      (FAST cards happen to agree too)"; else echo "      (FAST cards differ -- recorded, the shipped arm makes no cross-vendor claim)"; fi
    fi
  done
done
for d in "${DIRS[@]}"; do
  [ -d "$d/lanes" ] || continue
  nf=$(grep -c 'PHASE8-FINDING' "$d/bootstrap.log" 2>/dev/null)
  echo "  $(label_of "$d"): $nf phase-8 findings$( [ "$nf" != 0 ] && grep 'PHASE8-FINDING' "$d/bootstrap.log" | head -6 | sed 's/^/\n      /' )"
  [ "$nf" = 0 ] || RC=1
done

if [ $WRITE = 1 ]; then
  cp /tmp/e3_trees.md "$REF/e3_verdicts_trees.md"
  [ -f /tmp/e3_e2u.md ] && cp /tmp/e3_e2u.md "$REF/e3_verdicts_e2u.md"
  echo "wrote $REF/e3_verdicts_trees.md and e3_verdicts_e2u.md"
fi
echo "########## E3 round verdict: $([ $RC = 0 ] && echo IDENTICAL || echo NOT-CLOSED) (rc=$RC)"
exit $RC
