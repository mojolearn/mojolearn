#!/bin/sh
# The linear-algebra identity gate: the matrix products, PCA, tSVD and OLS,
# in BOTH modes.
#
#   pixi run check-linalg-identity
#
# IDENTITY_PATHS rows 27-31. DEVIATIONS 520, 521, 522, 523, 524, 525, 526.
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
mojo_only/gram_splitk_check.mojo
decomposition/mojo_only/jacobi_check.mojo
decomposition/mojo_only/pca_check.mojo
glm/mojo_only/ols_check.mojo
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
        out=$(pixi run mojo run -I . "$f" 2>&1) || fail=1
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
MOJOLEARN_LINALG_INNER=1 tools/with_build_lock.sh "$0"

echo
echo "== NUMERIC_IDENTICAL (session-local flip, reverted on exit) =="
MOJOLEARN_LINALG_INNER=1 tools/with_identical_mode.sh "$0"

echo
echo "linear-algebra identity: both modes green."
echo "Reminder: this is ONE DEVICE. Cross-vendor identity is E1's to"
echo "measure; see IDENTITY_PATHS.md and E1_RUNBOOK.md."
