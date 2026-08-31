#!/usr/bin/env bash
# DEVIATION 551's CROSS-VENDOR LEG. Runs ON THE BOX as e1_bootstrap.sh's
# phase 9 diag (MOJOLEARN_P9_DIAG=tools/diag/rbc_551_leg.sh,
# MOJOLEARN_P9_ONLY_DIAG=1), with $OUT and $REPO set.
#
# WHAT THIS LEG IS FOR, AND WHY NOTHING ON THE MAC CAN REPLACE IT.
#
# DEVIATION 551 canonicalizes the ball cover's CSR rows so that the same query
# returns the same BYTES on every vendor. The mechanism it repairs is a lane
# width: RBC_LANES is 32 on Apple and NVIDIA and 64 on CDNA, so the chunked
# backward walk in `rbc_eps_nn_query_fill` assigns slots in a different
# sequence there. A 32-lane box cannot run a 64-lane ballot, so no run on the
# M4 -- in either mode, at any fixture size -- carries information about the
# thing 551 exists to fix. `one-box-verdict-is-not-three` forbids reading the
# claim off the M4's own green.
#
# The artifact is `RBC-DIGEST`, an FNV-1a over the whole CSR (shape, row
# starts, columns), printed by the shared query helper in BOTH modes. The
# comparison is made OFF the box, against
# bench/results/identity/RBC_551_APPLE_M4.txt:
#
#   IDENTICAL  the three eps= digests MUST equal Apple's. That is the claim.
#   FAST       they are EXPECTED to DIFFER from Apple's, because the raw
#              emission order is what the lane width moves. This is the
#              non-vacuity control: if FAST matched too, the IDENTICAL match
#              would be evidence about the fixture, not about 551.
#
# A FAST arm that matched Apple would not be a pass. It would mean this box's
# emission order happens to agree with a 32-lane one, and the leg would have
# to be re-run on a fixture that separates them before the IDENTICAL match
# could be believed. That verdict is computed here, on the box, so a reader of
# the summary does not have to reconstruct it.
set -uo pipefail
OUT="${OUT:?}"; REPO="${REPO:?}"
cd "$REPO"
D="$OUT/diag"; mkdir -p "$D"
export PATH="$HOME/.pixi/bin:$PATH"
say() { echo "[$(date +%T) rbc551] $*"; }
T0=$(date +%s)

# `timeout` is GNU coreutils and is NOT on macOS. It is always present on the
# rented Linux box, which is where this runs for real; the fallback exists so
# the script can be DRY-RUN on the Mac before a box is rented. The dry run is
# the only thing that has ever caught a bug in this file (2026-08-31: a missing
# digest was scoring as a passing control, see `verdict`).
TMO=""
command -v timeout >/dev/null 2>&1 && TMO="timeout -k 30 ${RBC551_TIMEOUT:-900}"

MAIN="neighbors/ball_cover_main.mojo"

say "1/2 FAST arm"
$TMO pixi run mojo run -I . "$MAIN" > "$D/rbc551_fast.log" 2>&1
FAST_RC=$?
grep -E "^RBC-DIGEST|^ball_cover|raw emission order" "$D/rbc551_fast.log" | sed 's/^/    /'

say "2/2 IDENTICAL arm"
$TMO bash tools/with_identical_mode.sh pixi run mojo run -I . "$MAIN" \
  > "$D/rbc551_identical.log" 2>&1
IDENT_RC=$?
grep -E "^RBC-DIGEST|^ball_cover|raw emission order" "$D/rbc551_identical.log" | sed 's/^/    /'

# The three portable rows only. The sabotage labels are printed too, but two
# of the three sabotages mutate device state and their digests are meaningful
# only against the same box's own before/after pair, never across boxes.
digests() {   # $1 = log, $2 = mode
  grep -E "^RBC-DIGEST mode=$2 label=eps=" "$1" 2>/dev/null | sort
}
{
  echo "# DEVIATION 551 cross-vendor CSR digests -- THIS BOX"
  # NO `.git` ON AN ARCHIVE-DELIVERED BOX (2026-08-31, when the AMD leg
  # stopped shipping a 236 MB bundle). `commit.txt` is written by the leg
  # from the Mac side and is the attribution of record; `git rev-parse` is
  # the fallback for a box that was cloned.
  echo "commit=$(cat "$REPO/commit.txt" 2>/dev/null || git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "uname=$(uname -srm)"
  echo "fast_rc=$FAST_RC  identical_rc=$IDENT_RC"
  echo "seconds=$(( $(date +%s) - T0 ))"
  echo "-- FAST --"
  digests "$D/rbc551_fast.log" FAST
  echo "-- IDENTICAL --"
  digests "$D/rbc551_identical.log" IDENTICAL
} > "$D/rbc551_digests.txt"

# The Apple column, carried in the commit, so the comparison happens HERE and
# a fetched summary already says match or mismatch.
APPLE="bench/results/identity/RBC_551_APPLE_M4.txt"
verdict() {   # $1 = mode, $2 = log, $3 = expect-equal (1) or expect-differ (0)
  local mode="$1" log="$2" want="$3" n_same=0 n_diff=0 n_missing=0 n_seen=0
  while read -r label _mode _n _nnz digest; do
    case "$label" in \#*|""|label) continue ;; esac
    [ "$_mode" = "$mode" ] || continue
    case "$label" in eps=*) ;; *) continue ;; esac
    n_seen=$((n_seen + 1))
    local here
    here=$(grep -E "^RBC-DIGEST mode=$mode label=$label " "$log" 2>/dev/null \
           | sed 's/.* digest=//' | head -1)
    # A MISSING DIGEST IS NOT A DIFFERENCE. The first version of this function
    # counted it as one, which made the FAST arm -- whose PASS condition is
    # "differs from Apple" -- report "the control is not vacuous" for a run
    # that had crashed before printing anything at all. Three of three arms
    # printed nothing and the control still went green. That is `reached but
    # inert` in its purest form and the dry run on the Mac is what caught it.
    if [ -z "$here" ]; then
      echo "  $mode $label: MISSING -- this box printed no digest for this fixture"
      n_missing=$((n_missing + 1)); continue
    fi
    if [ "$here" = "$digest" ]; then
      echo "  $mode $label: SAME   apple=$digest"
      n_same=$((n_same + 1))
    else
      echo "  $mode $label: DIFFER apple=$digest here=$here"
      n_diff=$((n_diff + 1))
    fi
  done < "$APPLE"
  if [ "$n_seen" = 0 ]; then
    echo "  $mode: NO APPLE ROWS READ -- the baseline file is missing or malformed; VERDICT UNAVAILABLE"
    return 2
  fi
  if [ "$n_missing" -gt 0 ]; then
    echo "  $mode VERDICT: $n_missing of $n_seen fixtures printed NO digest. Neither arm can pass on a run that did not produce the artifact; read $log."
    return 3
  fi
  if [ "$want" = 1 ]; then
    [ "$n_diff" = 0 ] && echo "  $mode VERDICT: all $n_same match Apple" && return 0
    echo "  $mode VERDICT: $n_diff of $n_seen DIFFER from Apple -- 551 does NOT hold across these two vendors"
    return 1
  else
    [ "$n_diff" -gt 0 ] && echo "  $mode VERDICT: $n_diff of $n_seen differ from Apple, as expected; the control is not vacuous" && return 0
    echo "  $mode VERDICT: all $n_seen MATCH Apple under FAST. The fixture does not separate the two lane widths here, so the IDENTICAL match below proves nothing about 551 and this leg must be re-run on a fixture that does."
    return 1
  fi
}

{
  echo "== FAST (expect DIFFER from Apple: the non-vacuity control) =="
  verdict FAST "$D/rbc551_fast.log" 0; FAST_VERDICT=$?
  echo "== IDENTICAL (expect SAME as Apple: the claim) =="
  verdict IDENTICAL "$D/rbc551_identical.log" 1; IDENT_VERDICT=$?
  echo
  echo "fast_rc=$FAST_RC identical_rc=$IDENT_RC fast_verdict=$FAST_VERDICT identical_verdict=$IDENT_VERDICT"
  if [ "$FAST_RC" = 0 ] && [ "$IDENT_RC" = 0 ] && [ "$FAST_VERDICT" = 0 ] && [ "$IDENT_VERDICT" = 0 ]; then
    echo "DEVIATION 551: CLOSED ACROSS THESE TWO VENDORS"
  else
    echo "DEVIATION 551: NOT CLOSED -- read the two arms above before believing either"
  fi
} 2>&1 | tee "$D/rbc551_VERDICT.txt"

cat "$D/rbc551_VERDICT.txt" > "$D/SUMMARY.txt"
grep -q "^DEVIATION 551: CLOSED" "$D/SUMMARY.txt"
