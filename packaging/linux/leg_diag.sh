#!/usr/bin/env bash
# THE BUILD LEG'S PAYLOAD. Runs ON THE BOX as tools/e1_bootstrap.sh's phase 9
# diag (MOJOLEARN_P9_DIAG=packaging/linux/leg_diag.sh), with $OUT (the
# bootstrap's artifact directory, fetched whole by the leg) and $REPO set.
#
# What it leaves under $OUT/diag/ (everything the Mac needs, nothing else):
#   wheels/sets/<vendor>.tar.gz      the three staged sets + .libs + manifest
#   wheels/sets/<vendor>/manifest.json, readback.txt   (also inside the tar)
#   wheels/SIZES.txt, wheels/build_logs/
#   gates/smoke_<tier>.json          gate (a), one per tier
#   gates/sabotage.json              gate (b)
#   gates/nogpu.json                 gate (d)
#   SUMMARY.txt                      one screen: sizes and verdicts
#
# The untarred set and the assembled package copy are DELETED at the end so
# the fetch carries one copy of the binaries, not three.
set -uo pipefail
OUT="${OUT:?}"; REPO="${REPO:?}"
cd "$REPO"
D="$OUT/diag"; mkdir -p "$D/gates" "$D/wheels"
export PATH="$HOME/.pixi/bin:$PATH"
PIXI_ENV="${MOJOLEARN_BUILD_PIXI_ENV:-gbmbench}"
PY="pixi run -e $PIXI_ENV python3"
say() { echo "[$(date +%T) leg_diag] $*"; }
T0=$(date +%s)

say "1/5 build_sets"
bash packaging/linux/build_sets.sh "$D/wheels"; BUILD_RC=$?
VENDOR=$(sed -n 's/^vendor=//p' "$D/wheels/SIZES.txt" 2>/dev/null | head -1)
[ -n "$VENDOR" ] || { say "no vendor in SIZES.txt; stopping"; echo "build FAILED, no vendor" > "$D/SUMMARY.txt"; exit 1; }
ARCH=$(sed -n 's/^arch=//p' "$D/wheels/SIZES.txt" 2>/dev/null | head -1)
[ -n "$ARCH" ] || { say "no arch in SIZES.txt; build_sets.sh predates the architecture axis"; echo "build FAILED, no arch" > "$D/SUMMARY.txt"; exit 1; }
SET="$D/wheels/sets/$VENDOR/$ARCH"

say "2/5 assemble the package as the wheel will lay it out"
PKGROOT="$D/pkgroot"; rm -rf "$PKGROOT"; mkdir -p "$PKGROOT/mojolearn"
cp python/mojolearn/*.py "$PKGROOT/mojolearn/"
cp python/mojolearn_diagnostics.py "$PKGROOT/"
mkdir -p "$PKGROOT/mojolearn/$VENDOR"
cp -r "$SET" "$PKGROOT/mojolearn/$VENDOR/$ARCH"
rm -f "$PKGROOT/mojolearn/$VENDOR/$ARCH"/manifest.json \
      "$PKGROOT/mojolearn/$VENDOR/$ARCH"/readback.txt \
      "$PKGROOT/mojolearn/$VENDOR/$ARCH"/arch_readback.txt

say "3/5 gate (a): smoke, every family, every tier"
SMOKE_RC=0
for tier in fast deterministic identical; do
  MOJOLEARN_NUMERIC_MODE=$tier PYTHONPATH="$PKGROOT" MOJOLEARN_REPO="$REPO" \
    timeout -k 30 "${MOJOLEARN_SMOKE_TIMEOUT:-900}" \
    $PY -u packaging/linux/smoke.py --vendor "$VENDOR" --arch "$ARCH" --json "$D/gates/smoke_$tier.json" \
    > "$D/gates/smoke_$tier.log" 2>&1 || SMOKE_RC=1
  tail -3 "$D/gates/smoke_$tier.log" | sed 's/^/    /'
done

say "4/5 gate (b): sabotage"
$PY -u packaging/linux/sabotage.py --pkgroot "$PKGROOT" --vendor "$VENDOR" \
  --python "$PY" --json "$D/gates/sabotage.json" > "$D/gates/sabotage.log" 2>&1; SAB_RC=$?
tail -9 "$D/gates/sabotage.log" | sed 's/^/    /'

say "5/5 gate (d): no GPU"
$PY -u packaging/linux/nogpu.py --pkgroot "$PKGROOT" --vendor "$VENDOR" \
  --python "$PY" --json "$D/gates/nogpu.json" > "$D/gates/nogpu.log" 2>&1; NOGPU_RC=$?
tail -5 "$D/gates/nogpu.log" | sed 's/^/    /'

{
  echo "vendor=$VENDOR  build_rc=$BUILD_RC  smoke_rc=$SMOKE_RC  sabotage_rc=$SAB_RC  nogpu_rc=$NOGPU_RC"
  echo "seconds=$(( $(date +%s) - T0 ))"
  cat "$D/wheels/SIZES.txt"
  for tier in fast deterministic identical; do
    echo "smoke[$tier]: $(tail -1 "$D/gates/smoke_$tier.log" 2>/dev/null)"
  done
  echo "sabotage: $(tail -1 "$D/gates/sabotage.log" 2>/dev/null)"
  echo "nogpu: $(tail -1 "$D/gates/nogpu.log" 2>/dev/null)"
} | tee "$D/SUMMARY.txt"

rm -rf "$PKGROOT" "$D/wheels/tools-venv"
rm -rf "$SET"/_mojolearn*.so "$SET"/deterministic "$SET"/identical "$SET"/.libs
[ "$BUILD_RC" = 0 ] && [ "$SMOKE_RC" = 0 ] && [ "$SAB_RC" = 0 ]
