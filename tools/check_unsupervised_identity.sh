#!/bin/sh
# The unsupervised identity gate: k-means, k-NN and DBSCAN, in BOTH modes.
#
#   pixi run check-unsupervised-identity
#
# WHY BOTH MODES, AND WHY THAT IS THE WHOLE DESIGN OF THIS SCRIPT.
#
# `GLOBAL_NUMERIC_MODE` is comptime, so one process is one mode and no check
# can compare the two from inside. Every check in the three identity files
# is therefore written to assert something DIFFERENT in each mode:
#
#   FAST       the shipped arm still behaves as measured, and where the
#              upstream is non-reproducible the check REPORTS it rather than
#              asserting -- that report is the evidence the pins are worth
#              something. (`check_knn_fused_tie_set_is_geometry_invariant`
#              has printed a tie moving between two identical queries, on
#              one device, in one process.)
#   IDENTICAL  the pins are REACHED (each fixture separates the pinned
#              spelling from the unpinned one), the answer does not move
#              with the launch geometry, and every truncation that would
#              return a race-dependent answer raises instead.
#
# A run of only one mode proves half of that and reads like the whole of it,
# which is why this is one command.
#
# The three ALGORITHM suites run in both modes too: the identity files gate
# the new property, and those gate that the property did not cost the
# answer. A fit that is reproducible and wrong is not progress.
#
# THE FLIP IS SESSION-LOCAL AND REVERTED (`tools/with_identical_mode.sh`,
# trap on EXIT/INT/TERM). It must never be committed -- E1_RUNBOOK's
# preconditions -- and a stray IDENTICAL build silently changes every number
# the next session measures.
set -e

cd "$(dirname "$0")/.."

FILES="cluster/mojo_only/kmeans_identity_check.mojo
neighbors/mojo_only/knn_identity_check.mojo
dbscan/mojo_only/dbscan_identity_check.mojo
cluster/kmeans_main.mojo
neighbors/knn_main.mojo
dbscan/dbscan_main.mojo"

run_all() {
    fail=0
    for f in $FILES; do
        echo "  -- $f"
        out=$(pixi run mojo run -I . "$f" 2>&1) || fail=1
        echo "$out" | grep -E "^check_|^ball_cover|^Unhandled|error:" || true
        if [ "$fail" -ne 0 ]; then
            echo "FAILED: $f"
            return 1
        fi
    done
    return 0
}

if [ "$MOJOLEARN_UNSUP_INNER" = "1" ]; then
    # Re-entered by `with_identical_mode.sh` with the mode already flipped.
    run_all
    exit $?
fi

echo "== NUMERIC_FAST (the shipped build) =="
run_all

echo
echo "== NUMERIC_IDENTICAL (session-local flip, reverted on exit) =="
MOJOLEARN_UNSUP_INNER=1 tools/with_identical_mode.sh "$0"

echo
echo "unsupervised identity: both modes green."
echo "Reminder: this is ONE DEVICE. Cross-vendor identity is E1's to"
echo "measure; see IDENTITY_PATHS.md and E1_RUNBOOK.md."
