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

# IS THIS A CROSS-BUILD? A leg that compiles gfx90a on a gfx942 box, or
# sm_120a on an A40, produces a set that CANNOT RUN HERE, and the smoke,
# sabotage and no-GPU gates all fail for that reason alone. Reporting those
# as FAIL is how a REAL failure gets lost: on 2026-08-31 four of the six
# 0.3.1 sets were cross-built and their logs are full of expected failures
# that a reader has to know to discount.
#
# The device's own architecture is read the same way the selector reads it,
# so this asks the box rather than assuming from the GPU model. When it
# differs from the set's, the gates are recorded as NOT APPLICABLE with the
# reason, and the leg's verdict rests on build_rc and the read-back, which
# ARE meaningful for a cross-build.
DEVICE_ARCH=$($PY - "$REPO/python/mojolearn/_backend.py" "$VENDOR" <<'PYEOF' 2>/dev/null
import importlib.util, sys
# Load _backend.py STANDALONE. Importing the package would run its __init__,
# which calls select() and would fail here for the very reason we are asking.
spec = importlib.util.spec_from_file_location("_mb", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
arch, _how = m._device_arch(sys.argv[2])
print(arch or "")
PYEOF
)
CROSS_BUILD=0
if [ -n "$DEVICE_ARCH" ] && [ "$DEVICE_ARCH" != "$ARCH" ]; then
  # An `a`-suffixed build targets exactly the device that reports the bare
  # name, so sm_90 device + sm_90a set is NOT a cross-build.
  if [ "${ARCH%a}" != "$DEVICE_ARCH" ]; then
    CROSS_BUILD=1
    say "CROSS-BUILD: this box is $DEVICE_ARCH and the set is $ARCH."
    say "  The set cannot run here, so smoke, sabotage and nogpu are NOT"
    say "  APPLICABLE and are recorded as such rather than as failures."
    say "  This leg's evidence is build_rc and the architecture read-back."
  fi
fi

say "2/5 assemble the package as the wheel will lay it out"
PKGROOT="$D/pkgroot"; rm -rf "$PKGROOT"; mkdir -p "$PKGROOT/mojolearn"
cp python/mojolearn/*.py "$PKGROOT/mojolearn/"
cp python/mojolearn_diagnostics.py "$PKGROOT/"
mkdir -p "$PKGROOT/mojolearn/$VENDOR"
cp -r "$SET" "$PKGROOT/mojolearn/$VENDOR/$ARCH"
rm -f "$PKGROOT/mojolearn/$VENDOR/$ARCH"/manifest.json \
      "$PKGROOT/mojolearn/$VENDOR/$ARCH"/readback.txt \
      "$PKGROOT/mojolearn/$VENDOR/$ARCH"/arch_readback.txt

# NOT APPLICABLE IS NOT A PASS AND IT IS NOT A FAILURE. On a cross-build the
# three gates below exercise code that cannot run on this box, so their
# result carries no information about the set. They are recorded with the
# reason and skipped, and SUMMARY.txt says so, rather than filling the log
# with expected failures a reader has to know to discount. That habit is how
# a real failure gets lost.
if [ "$CROSS_BUILD" = 1 ]; then
  say "3/5..5/5 gates (a) smoke, (b) sabotage, (d) nogpu: NOT APPLICABLE"
  say "  this box is $DEVICE_ARCH, the set is $ARCH; the binaries cannot run here"
  SMOKE_RC=0; SAB_RC=0; NOGPU_RC=0
  for g in smoke_fast smoke_deterministic smoke_identical sabotage nogpu; do
    printf '{"verdict": "NOT APPLICABLE", "reason": "cross-build: box is %s, set is %s", "device_arch": "%s", "set_arch": "%s"}\n' \
      "$DEVICE_ARCH" "$ARCH" "$DEVICE_ARCH" "$ARCH" > "$D/gates/$g.json"
    echo "NOT APPLICABLE: cross-build, box is $DEVICE_ARCH and the set is $ARCH" > "$D/gates/$g.log"
  done
else

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
fi

{
  echo "vendor=$VENDOR  build_rc=$BUILD_RC  smoke_rc=$SMOKE_RC  sabotage_rc=$SAB_RC  nogpu_rc=$NOGPU_RC"
  if [ "$CROSS_BUILD" = 1 ]; then
    echo "gates=NOT APPLICABLE (cross-build: box $DEVICE_ARCH, set $ARCH); evidence is build_rc and the read-back"
  fi
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
