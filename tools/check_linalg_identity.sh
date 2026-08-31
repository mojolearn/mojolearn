#!/bin/sh
# The linear-algebra identity gate: the matrix products, PCA, tSVD, OLS,
# ridge and logistic regression, in BOTH modes.
#
#   pixi run check-linalg-identity
#
# IDENTITY_PATHS rows 27-31 (and the ridge / logistic rows). DEVIATIONS 520-527, 545-549.
#
# WHY BOTH MODES, AND WHY THAT IS THE WHOLE DESIGN OF THIS SCRIPT.
#
# `GLOBAL_NUMERIC_MODE` is comptime, so one process is one mode and no check
# can compare the two from inside. Every check in the identity files is
# therefore written to assert something DIFFERENT in each mode:
#
#   FAST       the shipped arm still behaves as measured, and where the
#              upstream is not launch-invariant the check REPORTS it rather
#              than asserting -- that report is the evidence the pins are
#              worth something. `check_pinned_gemm_is_batch_invariant` is
#              the one to read: it prints the vendor matmul's bits for one
#              cell at three launch sizes and says whether they agree.
#   IDENTICAL  the pins are REACHED, the answer does not move with the
#              launch geometry, and every shape whose only other arm is a
#              closed vendor library RAISES instead of quietly returning a
#              model this mode promises is vendor-independent.
#
# A run of only one mode proves half of that and reads like the whole of it,
# which is why this is one command. Modelled on
# `tools/check_unsupervised_identity.sh`, which does the same for k-means,
# k-NN and DBSCAN; the two are deliberately the same shape.
#
# EVERY CHECK PRINTS THE MODE IT COMPILED IN. That is not decoration: the
# flip is an edit to a shared file, and a run that compiled inside another
# session's flip window would otherwise report the other arm's numbers under
# this arm's label. It happened during this lane's own development, which is
# why `with_identical_mode.sh` now takes the build lock (DEVIATION 514) and
# why the printed witness is here as well. Two independent guards, because
# a silently mislabelled measurement is the failure that teaches nothing.
#
# THE ALGORITHM SUITES RUN IN BOTH MODES TOO: the identity files gate the
# new property, and those gate that the property did not cost the answer. A
# fit that is reproducible and wrong is not progress.
#
# THE FLIP IS SESSION-LOCAL AND REVERTED (`tools/with_identical_mode.sh`,
# trap on EXIT/INT/TERM, under the shared build lock). It must never be
# committed -- E1_RUNBOOK's preconditions.
set -e

cd "$(dirname "$0")/.."

# Identity files first, then the algorithm suites they must not have broken.
FILES="core/gemm_identity_check.mojo
core/column_stats_identity_check.mojo
original/gram_splitk_check.mojo
decomposition/original/jacobi_check.mojo
decomposition/original/pca_check.mojo
glm/original/ols_check.mojo
glm/original/ridge_check.mojo
glm/original/logistic_check.mojo
decomposition/pca_main.mojo
decomposition/jacobi_main.mojo
glm/ols_main.mojo"

run_all() {
    fail=0
    for f in $FILES; do
        if [ ! -f "$f" ]; then
            echo "  -- $f  (ABSENT, skipped)"
            continue
        fi
        echo "  -- $f"
        out=$(pixi run mojo run ${MOJOLEARN_MOJO_DEFINES:-} -I . "$f" 2>&1) || fail=1
        echo "$out" | grep -E "^check_|^== |^Unhandled|error:" || true
        if [ "$fail" -ne 0 ]; then
            echo "FAILED: $f"
            return 1
        fi
    done
    return 0
}

if [ "$MOJOLEARN_LINALG_INNER" = "1" ]; then
    # Re-entered by `with_identical_mode.sh` with the mode already flipped.
    run_all
    exit $?
fi

# The FAST arm takes the build lock too. Without it this arm can compile
# inside ANOTHER session's flip window and print "FAST" over IDENTICAL
# numbers -- the race DEVIATION 514 was opened for, from the other side.
echo "== NUMERIC_FAST (the shipped build) =="
# BOTH PASSES ALWAYS RUN (2026-08-23). Until then a FAST-pass failure ended
# the script before the IDENTICAL pass, and on every H100/MI325X leg through
# leg 9 some Apple-shaped FAST assertion did exactly that -- so the
# IDENTICAL pass, which is the pass the cross-vendor claim rests on, never
# ran on a box. The two passes answer different questions (FAST: the shipped
# arm on this vendor; IDENTICAL: the pinned arm is reached and right); each
# is recorded on its own and the exit status is the OR of the two.
set +e
MOJOLEARN_LINALG_INNER=1 MOJOLEARN_MOJO_DEFINES= MOJOLEARN_NUMERIC_MODE=fast tools/with_build_lock.sh "$0"
fast_rc=$?
[ $fast_rc = 0 ] || echo "linear-algebra identity: FAST pass FAILED (rc=$fast_rc) -- recorded; the IDENTICAL pass runs regardless"

echo
echo "== NUMERIC_IDENTICAL (build define, session-local) =="
MOJOLEARN_LINALG_INNER=1 tools/with_identical_mode.sh "$0"
ident_rc=$?
[ $ident_rc = 0 ] || echo "linear-algebra identity: IDENTICAL pass FAILED (rc=$ident_rc)"

echo
if [ $fast_rc = 0 ] && [ $ident_rc = 0 ]; then
  echo "linear-algebra identity: both modes green."
else
  echo "linear-algebra identity: FAST rc=$fast_rc, IDENTICAL rc=$ident_rc -- NOT both green (read each pass's lines above)"
fi
echo "Reminder: this is ONE DEVICE. Cross-vendor identity is E1's to"
echo "measure; see IDENTITY_PATHS.md and E1_RUNBOOK.md."
[ $fast_rc = 0 ] && [ $ident_rc = 0 ]
