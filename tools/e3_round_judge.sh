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
#   7. the lanes' cards (bootstrap phase 8: gemm, cd, kde, linkage, svm,
#      metrics, mamba): IDENTICAL cards Apple vs each box through
#      identity_trace_diff (judged); FAST cards recorded, never judged;
#      phase-8 findings, each one either UNEXPECTED -- which fails the round
#      -- or named by the allowlist below and FORGIVEN IN THE OPEN
#
# Exit code is 0 only when 2, 3 and 4 are all IDENTICAL/REFUSED=, 5 shows no
# FAIL line the allowlist does not name, and 7 has no divergence, no missing
# IDENTICAL card and no unexpected phase-8 finding. With --write the tables
# land as <mac_ref_dir>/e3_verdicts_<label>.md for E3_RESULTS.md to include.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

# ---------------------------------------------------------------------------
# DEVIATION 860 -- THE PHASE-8 FINDING ALLOWLIST, BY NAME.
#
# Section 7 used to end with
#
#     nf=$(grep -c 'PHASE8-FINDING' "$d/bootstrap.log")
#     [ "$nf" = 0 ] || RC=1
#
# and that became wrong the day the mamba lane was wired into phase 8
# (b56c158, 2026-08-23). tools/e1_bootstrap.sh says at the lane itself that
# FAST MODE HAS NEVER BEEN BUILT FOR THIS LANE, so a `PHASE8-FINDING: mamba
# [fast]` line is EXPECTED and is information rather than an alarm. Counting
# it turned the whole round red, on EVERY column, for a documented and
# deliberate absence. A gate that is red for a known reason is a gate an
# operator learns to walk past, and that is how the next real finding gets
# missed.
#
# THE FIX IS A NAME, NOT A NUMBER. A count threshold ("allow one finding") or
# a `|| true` would forgive the NEXT finding too, whatever it turned out to
# be, and the point of this gate is that nobody knows what the next one is.
# Each entry below is a PREFIX of the finding line it forgives, so it names
# the lane AND the arm: `mamba [fast]` does not forgive `mamba [identical]`
# and it does not forgive `metrics-regression [fast]`.
#
# IT IS FORGIVEN IN THE OPEN. Section 7 prints this list, prints every line
# it matched, and says so when an entry matched nothing. A silently forgiven
# failure is worse than no gate at all -- this repository's standing position
# and the reason the forgiveness is printed even when it fires every time.
#
# WHEN AN ENTRY HERE IS LEGITIMATE
#   * the thing it forgives is a DOCUMENTED, DELIBERATE ABSENCE: a mode or an
#     arm that was never built, in a file that says so AT THE SITE;
#   * the document is in the tree, not in a session's memory, and the entry
#     cites it;
#   * nothing is claimed from the arm that is missing, so the absence cannot
#     hide a cross-vendor result.
#
# WHAT MAKES AN ENTRY A LIE
#   * A DEFECT THAT HAS NOT BEEN FIXED YET. `metrics-regression [fast]` went
#     red on leg 11 (2026-08-23: "a subnormal p must contribute 0, not -inf
#     or NaN"). It does NOT go here. It is a bug, it has an owner, and
#     forgiving it would convert a red gate into a comment nobody reads.
#   * Anything on an IDENTICAL arm. The IDENTICAL arm is the entire claim of
#     the round; there is no such thing as an expected failure there.
#   * A lane that is merely flaky, or red on one vendor only. That is a
#     finding ABOUT THE VENDOR and it is what this whole file exists for.
#
# An entry that stops matching is a STALE entry and section 7 says so. Delete
# it then: keeping it is how the next real finding gets forgiven by a line
# nobody re-read.
# ---------------------------------------------------------------------------
PHASE8_EXPECTED_FINDINGS=(
  # tools/e1_bootstrap.sh, phase 8, at the mamba lane: "FAST MODE HAS NEVER
  # BEEN BUILT FOR THIS LANE, so a PHASE8-FINDING on the fast arm is EXPECTED
  # here and is information, not an alarm." Both spellings that arm can emit
  # start with this prefix -- `... failed (see <log>)` from run_lane_arm and
  # `... check FAILED: ...` from run_lane_check.
  "PHASE8-FINDING: mamba [fast]"
  # mamba2 joined phase 8 on 2026-09-01 with the same FAST posture as mamba:
  # that arm has never been recorded anywhere, so its finding is expected.
  "PHASE8-FINDING: mamba2 [fast]"
  # mamba3 joined phase 8 on 2026-09-01 (the day its lane landed) with the
  # same FAST posture: FAST has never been recorded for it anywhere
  # (IDENTICAL_MAMBA3_CONTRACT.md "STILL OWED"), so its finding is expected.
  "PHASE8-FINDING: mamba3 [fast]"
)

# A lane whose FAST arm was never built has a FAST card that is ABSENT rather
# than different, and section 7 has to say which of those it is looking at.
# This list is separate from the one above on purpose: a lane can be missing
# a fast card without emitting any finding (nothing ran, so nothing failed),
# and a lane can emit an expected finding and still leave a card behind.
#
# A MISSING **IDENTICAL** CARD IS ON NO LIST ANYWHERE. It is a hard failure
# for every lane including mamba, because that card is what the round is made
# of. Nothing below can excuse one.
PHASE8_EXPECTED_MISSING_FAST=(
  mamba
  mamba2
  mamba3
)

# An empty entry is a prefix of every line, so it would forgive the whole
# gate while looking like an allowlist. Refuse it here rather than discover
# it in a green round.
for _e in "${PHASE8_EXPECTED_FINDINGS[@]}"; do
  [ -n "$_e" ] || { echo "e3_round_judge: an EMPTY allowlist entry forgives EVERY finding. Refusing to judge." >&2; exit 2; }
done

phase8_finding_expected() {   # <finding line> -> 0 when an entry NAMES it
  local _line="$1" _e
  [ ${#PHASE8_EXPECTED_FINDINGS[@]} -gt 0 ] || return 1
  for _e in "${PHASE8_EXPECTED_FINDINGS[@]}"; do
    case "$_line" in "$_e"*) return 0 ;; esac
  done
  return 1
}

phase8_fast_card_expected_missing() {   # <lane> -> 0 when its FAST arm is known absent
  local _lane="$1" _e
  [ ${#PHASE8_EXPECTED_MISSING_FAST[@]} -gt 0 ] || return 1
  for _e in "${PHASE8_EXPECTED_MISSING_FAST[@]}"; do
    [ "$_lane" = "$_e" ] && return 0
  done
  return 1
}

# DEVIATION 861 -- THE ALLOWLIST IS APPLIED IN TWO PLACES, AND BOTH ARE HERE.
# Section 7's finding gate is its home. Section 5 reads the SAME bootstrap.log
# and reds the round on the bare word FAIL, and the bootstrap's spelling for a
# failed lane check is `PHASE8-FINDING: <lane> [<mode>] check FAILED: ...` --
# so without this the allowlist would be decorative: section 7 would forgive
# the mamba fast line and section 5 would re-red the round on the identical
# characters. Section 5 drops ONLY lines an entry above names, counts what it
# dropped, and prints the count. Its PRINTING greps still read the whole log,
# so nothing is hidden from the screen; only the verdict is affected.
P8_ALLOW_FILE="/tmp/e3_p8_allow.$$"
printf '%s\n' "${PHASE8_EXPECTED_FINDINGS[@]}" > "$P8_ALLOW_FILE"
p8_strip_expected() {   # <log> <out> -- drop only lines BEGINNING with an entry
  awk 'NR==FNR { pre[++n]=$0; next }
       { for (i = 1; i <= n; i++) if (index($0, pre[i]) == 1) next
         print }' "$P8_ALLOW_FILE" "$1" > "$2"
}

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
gate_i=0
for d in "${DIRS[@]}"; do
  log="$d/bootstrap.log"
  [ -f "$log" ] || { echo "  $(label_of "$d"): no bootstrap.log"; continue; }
  oks=$(grep -cE ' OK\b|OK:' "$log")
  phases=$(grep -c '^=== phase' "$log")
  echo "  $(label_of "$d"): $phases phase markers, $oks OK lines"
  # THE PRINTING GREPS READ THE WHOLE LOG. Nothing is hidden from the screen;
  # only the verdict below is affected by the allowlist (DEVIATION 861).
  grep -nE 'FINDING|FAIL|Unhandled exception|Traceback|error:' "$log" | grep -v 'warning' | head -12 | sed 's/^/      /'
  grep -nE 'signed-zero arm hash|signed-zero arm:|contraction: a\*b\+c is|ieee arith check OK' "$log" | sed 's/^/      /'
  # DEVIATION 861: judge a copy with the ALLOWLISTED phase-8 findings removed
  # and nothing else removed. `PHASE8-FINDING: mamba [fast] check FAILED: ...`
  # contains the bare word FAIL, so this scan would otherwise re-red the round
  # on the exact line section 7 forgives by name.
  gate_i=$((gate_i + 1))
  scan="/tmp/e3_gate_$$_$gate_i"
  p8_strip_expected "$log" "$scan"
  ndrop=$(( $(wc -l < "$log") - $(wc -l < "$scan") ))
  # awk terminates its last record with a newline even when the input did
  # not, so a log with no final newline can make this difference negative.
  # A negative count is nonsense on the screen; the filter dropped nothing.
  [ "$ndrop" -gt 0 ] || ndrop=0
  if [ ! -s "$scan" ] && [ -s "$log" ]; then
    # THE FILTER ATE THE LOG. A gate that reads an empty file passes
    # everything, so fall back to the unfiltered one and say so rather than
    # returning a green nobody can explain.
    echo "      !! the allowlist filter produced an EMPTY log; judging the unfiltered one instead"
    scan="$log"; ndrop=0
  fi
  [ "$ndrop" = 0 ] || echo "      ($ndrop allowlisted phase-8 finding line(s) excluded from the FAIL scan -- section 7 names them)"
  if grep -qE 'FAIL|Unhandled exception|Traceback' "$scan"; then RC=1; fi
  [ "$scan" = "$log" ] || rm -f "$scan"
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
# `mamba` joined this list 2026-08-23 (b56c158). It is NOT a classical lane:
# it is the Mamba-1 block under profile mojolearn.identical.mamba1.fp32.v1.
# ITS FAST ARM HAS NEVER BEEN BUILT ANYWHERE, so on that arm its card is
# expected ABSENT and its PHASE8-FINDING is expected PRESENT. Both of those
# are named, by name, at the top of this file, and both are printed here when
# they happen -- see DEVIATION 860.
#
# NOTHING EXCUSES A MISSING **IDENTICAL** CARD, mamba included. That card is
# what the round is made of, so its absence is RC=1 for every lane in the
# list below, exactly as it was before the allowlist existed.
for lane in gemm cd kde linkage svm metrics mamba mamba2 mamba3 iforest transformer \
           arima cholesky embedding gp hdbscan holtwinters ivf \
           kernelmethods gmm resample spectral \
           training-loss training-optimizer training-step tsa; do
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
    else
      # A MISSING FAST CARD IS NEVER RC=1 -- the shipped FAST arm makes no
      # cross-vendor claim, so the round has nothing to lose here. It is still
      # SAID, and it says WHICH KIND of absence it is: a lane whose FAST arm
      # has never been built anywhere (named at the top of this file) reads
      # differently from a lane that simply wrote no card on this run, and the
      # second one is a fact about the run that deserves a reader's eye.
      # Before this the two were the same silence.
      fmiss=""
      [ -f "$fa" ] || fmiss="APPLE"
      [ -f "$fb" ] || fmiss="${fmiss:+$fmiss and }$(label_of "$d")"
      if phase8_fast_card_expected_missing "$lane"; then
        echo "      (FAST card absent on $fmiss -- EXPECTED: this lane's FAST arm has never been built)"
      else
        echo "      (FAST card absent on $fmiss -- recorded, not judged; the FAST arm makes no cross-vendor claim)"
      fi
    fi
  done
done
# THE PHASE-8 FINDINGS (DEVIATION 860). Every finding line is either NAMED by
# the allowlist at the top of this file, in which case it is forgiven AND
# PRINTED as forgiven, or it is UNEXPECTED and it fails the round. The old
# spelling was `nf=$(grep -c PHASE8-FINDING ...); [ "$nf" = 0 ] || RC=1`,
# which counted the expected mamba fast line and turned every column red.
echo "  the phase-8 finding allowlist in force (defined at the top of this file):"
for e in "${PHASE8_EXPECTED_FINDINGS[@]}"; do
  echo "      forgives any finding beginning: $e"
done
for d in "${DIRS[@]}"; do
  [ -d "$d/lanes" ] || continue
  flog="$d/bootstrap.log"
  fexp="/tmp/e3_p8_exp_$$"; funx="/tmp/e3_p8_unexp_$$"
  : > "$fexp"; : > "$funx"
  while IFS= read -r fline; do
    [ -n "$fline" ] || continue
    if phase8_finding_expected "$fline"; then printf '%s\n' "$fline" >> "$fexp"
    else printf '%s\n' "$fline" >> "$funx"; fi
  done < <(grep -h 'PHASE8-FINDING' "$flog" 2>/dev/null)
  nexp=$(grep -c . "$fexp"); nunx=$(grep -c . "$funx")
  echo "  $(label_of "$d"): $((nexp + nunx)) phase-8 finding(s) -- $nunx unexpected, $nexp expected and FORGIVEN"
  if [ "$nexp" != 0 ]; then
    echo "      FORGIVEN by name (this is the forgiveness, in the open):"
    head -12 "$fexp" | sed 's/^/        /'
  fi
  if [ "$nunx" != 0 ]; then
    echo "      UNEXPECTED -- these fail the round:"
    head -12 "$funx" | sed 's/^/        /'
    RC=1
  fi
  rm -f "$fexp" "$funx"
done
# AN ENTRY THAT NEVER FIRES IS A STALE ENTRY. Said out loud, and deliberately
# NOT a failure: on a round where no column ran that lane there is nothing for
# it to match, and turning that into red would be the same mistake in the
# other direction. Delete the entry when the absence it names has been filled
# in -- an allowlist nobody re-reads is how the next real finding is forgiven.
for e in "${PHASE8_EXPECTED_FINDINGS[@]}"; do
  ehit=0
  for d in "${DIRS[@]}"; do
    [ -f "$d/bootstrap.log" ] || continue
    grep -qF "$e" "$d/bootstrap.log" && ehit=1
  done
  [ "$ehit" = 1 ] || echo "      NOTE: the allowlist entry '$e' matched nothing in this round; if the arm it names has been built since, DELETE THE ENTRY."
done

if [ $WRITE = 1 ]; then
  cp /tmp/e3_trees.md "$REF/e3_verdicts_trees.md"
  [ -f /tmp/e3_e2u.md ] && cp /tmp/e3_e2u.md "$REF/e3_verdicts_e2u.md"
  echo "wrote $REF/e3_verdicts_trees.md and e3_verdicts_e2u.md"
fi
rm -f "$P8_ALLOW_FILE"
echo "########## E3 round verdict: $([ $RC = 0 ] && echo IDENTICAL || echo NOT-CLOSED) (rc=$RC)"
exit $RC
