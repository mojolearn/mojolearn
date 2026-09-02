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
#      WHICH LANES AND AT WHAT SIZE IS NOW THIS LEG'S ENVIRONMENT'S CHOICE;
#      see the block above the call. The default is unchanged.
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

# THE LANE LIST AND THE FIXTURE SIZES COME FROM THIS LEG'S ENVIRONMENT.
# This used to call the script bare, which is the six frozen lanes at the
# sizes an Apple M4 was measured at, and on a datacenter GPU those are too
# small to separate the two arms. Measured 2026-09-02 at one commit on an M4
# and an H100, 5 rounds: the cd lane took 1.3 ms on the H100 against 89 ms on
# the M4, and the gemm lane took 100 MICROSECONDS against 9.2 ms, with a
# per-round ratio band of 0.205 .. 0.773. A band that straddles 1.0 says the
# fixture cannot tell the two arms apart. That is a fixture that is too
# small; it is not a price, and it is not a finding about identity.
#
# DEFAULT BEHAVIOUR IS UNCHANGED. With nothing set this runs the same six
# lanes at the same sizes the bare call ran. Set any of these in the leg's
# environment to ask a different question:
#
#   MOJOLEARN_LANES_PRICE_LANES   which lanes, space separated. Default is
#                                 the six frozen ones; the paper's Section 7
#                                 arms are: kmeans knn dbscan gram nt gemv.
#   MOJOLEARN_LANES_PRICE_ROUNDS  alternated rounds per lane (default 5).
#   MOJOLEARN_LANES_PRICE_KMEANS_ROWS, _KNN_INDEX, _KNN_METHOD,
#   _DBSCAN_ROWS, _GRAM_ROWS, _NT_ROWS, _GEMV_DIM
#                                 the six new lanes' fixture sizes. Unset
#                                 means the Apple default, which keeps the
#                                 published Apple numbers reproducible;
#                                 bench/lanes_price_main.mojo's header names
#                                 a datacenter step for each of them.
#
# AN EMPTY VALUE IS THE SAME AS UNSET all the way down, which is why these
# are forwarded unconditionally: tools/lanes_price.sh reads its own two with
# `${VAR:-default}`, and the driver's `_env_int` returns its default for the
# empty string.
PRICE_LANES="${MOJOLEARN_LANES_PRICE_LANES:-cd kde linkage svm metrics gemm}"
say "2/2 the identity tax, alternated, lanes [$PRICE_LANES]"
MOJOLEARN_LANES_PRICE_OUT="$D/lanes_price" \
  MOJOLEARN_LANES_PRICE_LANES="$PRICE_LANES" \
  MOJOLEARN_LANES_PRICE_ROUNDS="${MOJOLEARN_LANES_PRICE_ROUNDS:-}" \
  MOJOLEARN_LANES_PRICE_KMEANS_ROWS="${MOJOLEARN_LANES_PRICE_KMEANS_ROWS:-}" \
  MOJOLEARN_LANES_PRICE_KNN_INDEX="${MOJOLEARN_LANES_PRICE_KNN_INDEX:-}" \
  MOJOLEARN_LANES_PRICE_KNN_METHOD="${MOJOLEARN_LANES_PRICE_KNN_METHOD:-}" \
  MOJOLEARN_LANES_PRICE_DBSCAN_ROWS="${MOJOLEARN_LANES_PRICE_DBSCAN_ROWS:-}" \
  MOJOLEARN_LANES_PRICE_GRAM_ROWS="${MOJOLEARN_LANES_PRICE_GRAM_ROWS:-}" \
  MOJOLEARN_LANES_PRICE_NT_ROWS="${MOJOLEARN_LANES_PRICE_NT_ROWS:-}" \
  MOJOLEARN_LANES_PRICE_GEMV_DIM="${MOJOLEARN_LANES_PRICE_GEMV_DIM:-}" \
  bash tools/lanes_price.sh > "$D/lanes_price.log" 2>&1
PRICE_RC=$?
tail -12 "$D/lanes_price.log" 2>/dev/null | sed 's/^/    /'

{
  echo "commit=$(cat "$REPO/commit.txt" 2>/dev/null || git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "uname=$(uname -srm)"
  echo "ols_rc=$OLS_RC  price_rc=$PRICE_RC"
  echo "seconds=$(( $(date +%s) - T0 ))"
  # WHAT WAS ASKED FOR, beside what came back. A ratio table is unreadable
  # without the fixture sizes that produced it, and an unset size knob is
  # the Apple default rather than nothing.
  echo "price_lanes=$PRICE_LANES"
  echo "price_rounds=${MOJOLEARN_LANES_PRICE_ROUNDS:-(default 5)}"
  echo "price_sizes=kmeans_rows=${MOJOLEARN_LANES_PRICE_KMEANS_ROWS:-(apple default)}" \
       "knn_index=${MOJOLEARN_LANES_PRICE_KNN_INDEX:-(apple default)}" \
       "knn_method=${MOJOLEARN_LANES_PRICE_KNN_METHOD:-(auto)}" \
       "dbscan_rows=${MOJOLEARN_LANES_PRICE_DBSCAN_ROWS:-(apple default)}" \
       "gram_rows=${MOJOLEARN_LANES_PRICE_GRAM_ROWS:-(apple default)}" \
       "nt_rows=${MOJOLEARN_LANES_PRICE_NT_ROWS:-(apple default)}" \
       "gemv_dim=${MOJOLEARN_LANES_PRICE_GEMV_DIM:-(apple default)}"
  echo "-- ols --"
  grep -E "check_ols_is_launch_invariant" "$D/linalg_identity.log" 2>/dev/null | tail -3
  echo "-- identity tax (ratio.tsv) --"
  cat "$D/lanes_price/ratio.tsv" 2>/dev/null || echo "(no ratio.tsv; read lanes_price.log)"
} | tee "$D/SUMMARY.txt"

[ "$OLS_RC" = 0 ] && [ "$PRICE_RC" = 0 ]
