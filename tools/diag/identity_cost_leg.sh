#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
#
# TWO OWED MEASUREMENTS IN ONE LEASE. Runs ON THE BOX as e1_bootstrap.sh's
# phase-9 diag, with $OUT and $REPO set.
#
#   1. THE IDENTITY TAX. `tools/lanes_price.sh` builds the FAST and the
#      IDENTICAL binary and then ALTERNATES two ready binaries, F I F I,
#      with no compile inside the measurement window. Until today that had
#      only ever run on an Apple M4 (bench/results/IDENTITY_COST_OWED.md),
#      and the Apple number is plausibly the CHEAPEST of the three: Apple's
#      fmin/fmax and its sqrt already agree with the pinned forms, which is
#      why IDENTITY_PATHS row 10 exists, while NVIDIA's approximate PTX sqrt
#      does not, which is what DEVIATION 550 is about. Quoting one box as
#      "the" cost errs in the direction that flatters us.
#
#   2. THE LEAST-SQUARES RE-VERIFY. `check_ols_is_launch_invariant` failed on
#      the H100 and the MI325X in BOTH modes: two identical OLS fits in one
#      process disagreed at coefficient 0. Commit 4528f993 found it was never
#      a kernel bug -- one host staging buffer, two async copies, and a host
#      rewrite racing the DMA in between, so the design matrix differed run to
#      run. Apple passed because unified memory leaves no DMA to race, which
#      is exactly why the fix CANNOT be confirmed on Apple. This is the box
#      that can confirm it.
#
# The two are together because they want the same thing: one box, one lease,
# nothing else competing for the GPU.
set -uo pipefail
OUT="${OUT:?}"; REPO="${REPO:?}"
cd "$REPO"
D="$OUT/diag"; mkdir -p "$D"
export PATH="$HOME/.pixi/bin:$PATH"
say() { echo "[$(date +%T) idcost] $*"; }
T0=$(date +%s)

say "1/2 the least-squares re-verify (the cheap one, and it can FAIL loudly)"
MOJOLEARN_LINALG_IDENTITY_OUT="$D/linalg" \
  bash tools/with_identical_mode.sh pixi run check-linalg-identity \
  > "$D/linalg_identity.log" 2>&1
OLS_RC=$?
grep -E "check_ols_is_launch_invariant|^check_|OK|FAIL" "$D/linalg_identity.log" 2>/dev/null | tail -8 | sed 's/^/    /'

say "2/2 the identity tax, six lanes, alternated"
MOJOLEARN_LANES_PRICE_OUT="$D/lanes_price" \
  bash tools/lanes_price.sh > "$D/lanes_price.log" 2>&1
PRICE_RC=$?
tail -12 "$D/lanes_price.log" 2>/dev/null | sed 's/^/    /'

{
  echo "commit=$(cat "$REPO/commit.txt" 2>/dev/null || git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "uname=$(uname -srm)"
  echo "ols_rc=$OLS_RC  price_rc=$PRICE_RC"
  echo "seconds=$(( $(date +%s) - T0 ))"
  echo "-- ols --"
  grep -E "check_ols_is_launch_invariant" "$D/linalg_identity.log" 2>/dev/null | tail -3
  echo "-- identity tax (ratio.tsv) --"
  cat "$D/lanes_price/ratio.tsv" 2>/dev/null || echo "(no ratio.tsv; read lanes_price.log)"
} | tee "$D/SUMMARY.txt"

[ "$OLS_RC" = 0 ] && [ "$PRICE_RC" = 0 ]
