#!/usr/bin/env bash
# THE SMOKE LEG'S PAYLOAD: install the published wheel from TestPyPI (or
# PyPI) into a CLEAN venv on the box, outside the checkout, and run the
# release smoke in every tier. Runs ON THE BOX as phase 9's diag
# (MOJOLEARN_P9_DIAG=packaging/linux/leg_diag_install.sh).
#
#   MOJOLEARN_WHEEL_VERSION=X.Y.Z         required
#   MOJOLEARN_WHEEL_INDEX=testpypi|pypi   default testpypi
#
# Leaves under $OUT/diag/: install.log, pip_show.txt, gates/smoke_<tier>.json,
# SUMMARY.txt. The vendor asserted is the one READ BACK from the built
# tree's own probe of this box, so the same script serves both legs.
set -uo pipefail
OUT="${OUT:?}"; REPO="${REPO:?}"
cd "$REPO"
D="$OUT/diag"; mkdir -p "$D/gates"
VER="${MOJOLEARN_WHEEL_VERSION:?set MOJOLEARN_WHEEL_VERSION}"
IDX="${MOJOLEARN_WHEEL_INDEX:-testpypi}"
say() { echo "[$(date +%T) leg_install] $*"; }

VENV=/root/mojolearn-smoke-venv
rm -rf "$VENV"
python3 -m venv "$VENV" || { say "venv failed"; exit 2; }
PIP="$VENV/bin/pip"
if [ "$IDX" = testpypi ]; then
  "$PIP" install --no-cache-dir --index-url https://test.pypi.org/simple/ \
    --extra-index-url https://pypi.org/simple/ "mojolearn==$VER" > "$D/install.log" 2>&1
else
  "$PIP" install --no-cache-dir "mojolearn==$VER" > "$D/install.log" 2>&1
fi
RC=$?
tail -5 "$D/install.log" | sed 's/^/    /'
[ "$RC" = 0 ] || { say "pip install refused the wheel (see diag/install.log)"; echo "install FAILED" > "$D/SUMMARY.txt"; exit 1; }
"$PIP" show -f mojolearn > "$D/pip_show.txt" 2>&1

# The vendor to assert: what the box is, by the same probe the package uses.
VENDOR=$(cd / && "$VENV/bin/python" -c "import mojolearn; print(mojolearn.vendor())" 2>/dev/null | tail -1)
say "installed mojolearn==$VER from $IDX; vendor read back: ${VENDOR:-<import failed>}"
[ -n "$VENDOR" ] || { (cd / && "$VENV/bin/python" -c "import mojolearn") > "$D/import_error.log" 2>&1; tail -20 "$D/import_error.log"; echo "import FAILED" > "$D/SUMMARY.txt"; exit 1; }

SMOKE_RC=0
for tier in fast deterministic identical; do
  # cwd is / so the checkout cannot shadow the installed package.
  ( cd / && MOJOLEARN_NUMERIC_MODE=$tier MOJOLEARN_REPO="$REPO" \
      timeout -k 30 "${MOJOLEARN_SMOKE_TIMEOUT:-900}" \
      "$VENV/bin/python" -u "$REPO/packaging/linux/smoke.py" --vendor "$VENDOR" \
      --json "$D/gates/smoke_$tier.json" ) > "$D/gates/smoke_$tier.log" 2>&1 || SMOKE_RC=1
  tail -2 "$D/gates/smoke_$tier.log" | sed 's/^/    /'
done
{
  echo "index=$IDX version=$VER vendor=$VENDOR smoke_rc=$SMOKE_RC"
  for tier in fast deterministic identical; do
    echo "smoke[$tier]: $(tail -1 "$D/gates/smoke_$tier.log" 2>/dev/null)"
  done
} | tee "$D/SUMMARY.txt"
exit $SMOKE_RC
