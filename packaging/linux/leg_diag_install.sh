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

# A CLEAN VENV, FOUR WAYS, because `python3 -m venv` is not always available.
#
# 2026-08-30: this leg died here on the DigitalOcean image with "The virtual
# environment was not created successfully because ensurepip is not
# available", which is the SAME missing-package family that killed the AMD
# BUILD leg on patchelf earlier the same day. That one was fixed by giving
# the build four routes. This script was left with one, so the class of
# defect survived in a second place and cost a second rented box. Every
# route is tried in order and every outcome is printed, so a future failure
# names which routes were attempted rather than just the first.
VENV=/root/mojolearn-smoke-venv
rm -rf "$VENV"
mkvenv() {
  rm -rf "$VENV"
  say "venv route: $1"
  shift
  "$@" >/dev/null 2>&1 && [ -x "$VENV/bin/python3" ]
}
VENV_OK=0
# 1. the normal way.
mkvenv "python3 -m venv" python3 -m venv "$VENV" && VENV_OK=1
# 2. install the distro package that provides ensurepip, then retry. This is
#    the actual fix on Debian and Ubuntu images, and it is what the error
#    message itself asks for.
if [ "$VENV_OK" = 0 ]; then
  PYMM=$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)
  (apt-get update -qq && apt-get install -y -qq "python3-venv" "python${PYMM}-venv") >/dev/null 2>&1 || true
  mkvenv "python3 -m venv after apt-get install python${PYMM}-venv" python3 -m venv "$VENV" && VENV_OK=1
fi
# 3. no ensurepip, so build the venv without pip and bootstrap pip into it.
if [ "$VENV_OK" = 0 ]; then
  if python3 -m venv --without-pip "$VENV" >/dev/null 2>&1 && [ -x "$VENV/bin/python3" ]; then
    if (curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py \
        && "$VENV/bin/python3" /tmp/get-pip.py -q) >/dev/null 2>&1; then
      say "venv route: --without-pip + get-pip.py"
      VENV_OK=1
    fi
  fi
fi
# 4. virtualenv, which carries its own pip and needs no ensurepip at all.
if [ "$VENV_OK" = 0 ]; then
  (python3 -m pip install -q --user virtualenv) >/dev/null 2>&1 || true
  mkvenv "virtualenv" python3 -m virtualenv "$VENV" && VENV_OK=1
fi
if [ "$VENV_OK" = 0 ] || [ ! -x "$VENV/bin/pip" ]; then
  say "no clean venv could be made on this image; tried python3 -m venv, apt-get python3-venv, --without-pip + get-pip.py, and virtualenv"
  echo "venv FAILED" > "$D/SUMMARY.txt"
  exit 2
fi
PIP="$VENV/bin/pip"
say "clean venv at $VENV ($("$VENV/bin/python3" -V 2>&1))"
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
