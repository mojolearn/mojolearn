#!/bin/sh
# The OLS identity card (IDENTITY_PATHS row 32, DEVIATION 527).
#
#   pixi run ols-card                 # -> /tmp/ols.card and /tmp/ols.control.card
#   MOJOLEARN_OLS_CARD_OUT=x.card pixi run ols-card
#
# Under IDENTICAL, because that is the mode the card is a claim about. The
# driver runs the fit TWICE into two files and raises if they differ, so a
# card that is not even reproducible on ONE device never reaches a diff.
set -e
cd "$(dirname "$0")/.."
OUT="${MOJOLEARN_OLS_CARD_OUT:-/tmp/ols.card}"
MOJOLEARN_IDENTITY_TRACE="$OUT" \
MOJOLEARN_OLS_CARD_CONTROL="$OUT.control" \
    tools/with_identical_mode.sh pixi run mojo run -I . glm/ols_trace_main.mojo
echo
echo "card:    $OUT"
echo "control: $OUT.control"
echo "Ship it to the other machine and run:"
echo "  python3 tools/identity_trace_diff.py $OUT <other>.card"
