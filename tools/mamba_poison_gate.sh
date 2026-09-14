#!/bin/sh
# tools/mamba_poison_gate.sh: the DEVIATION 2712 gate. Build the Mamba binding
# with every scratch buffer filled with the canonical quiet NaN instead of
# zero (-D MOJOLEARN_MAMBA_POISON=1), run the four Mamba lanes of
# tools/identity_break.py COLD on that build, and require every cell to
# equal the committed three-vendor record. A kernel that reads a device
# element nobody wrote makes its lane's hash a NaN hash, which the diff
# reports as DIVERGENT and names the lane and the part.
#
#   pixi run check-mamba-poison                 the gate (must pass)
#   MOJOLEARN_MAMBA_POISON_SABOTAGE=1 pixi run check-mamba-poison
#                                               the sabotage: d_c_yoff and
#                                               d_dacs_yoff lose their zero
#                                               fill, so poison reaches the
#                                               rows the merge kernel sums;
#                                               the gate must FAIL
#
# The poison binding is built into a COPY of python/mojolearn (every other
# binding hard-linked, the Mamba one replaced), never over the binding a
# running process may have mapped (see memory: a .so rewritten in place
# under a mapped process is SIGKILL 137). PYTHONPATH points the harness at
# the copy. Environment:
#   MOJOLEARN_POISON_RECORD   record directory to diff against (default the
#                             newest record whose three vendor columns carry
#                             all four lanes: 2026-09-14_120-lanes-2711flip)
#   MOJOLEARN_BOX_LABEL       --vendor label for the column (default: the
#                             harness's own default for this box)
#   MOJOLEARN_POISON_KEEP=1   keep the package copy and the JSON
set -eu
HERE=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
cd "$HERE"
LANES="mamba1,mamba2,mamba2-dtlimit,mamba3"
RECORD="${MOJOLEARN_POISON_RECORD:-bench/results/identity_break/2026-09-14_120-lanes-2711flip}"
COLUMNS="$RECORD/apple-m4.json $RECORD/nvidia-h100-sm_90a.json $RECORD/amd-mi300x-gfx942.json"
for c in $COLUMNS; do [ -f "$c" ] || { echo "no record column at $c" >&2; exit 2; }; done
# MOJOLEARN_MAMBA_POISON_SABOTAGE
#   1  d_c_yoff and d_dacs_yoff lose their zero fill: the gate must FAIL
#   2  a planted read past silu_out's end, band present: the gate must FAIL
#   3  the same planted read with the band removed (MOJOLEARN_MAMBA_POISON_NOBAND):
#      the gate must PASS, the control that the band is what catches an over-read
SABOTAGE="${MOJOLEARN_MAMBA_POISON_SABOTAGE:-0}"
case "$SABOTAGE" in 0|1|2|3) ;; *) echo "MOJOLEARN_MAMBA_POISON_SABOTAGE must be 0, 1, 2 or 3" >&2; exit 2 ;; esac
DEFINES="-D MOJOLEARN_MAMBA_POISON=1"
[ "$SABOTAGE" = 1 ] && DEFINES="$DEFINES -D MOJOLEARN_MAMBA_POISON_SABOTAGE=1"
[ "$SABOTAGE" = 2 ] && DEFINES="$DEFINES -D MOJOLEARN_MAMBA_POISON_OVERREAD=1"
[ "$SABOTAGE" = 3 ] && DEFINES="$DEFINES -D MOJOLEARN_MAMBA_POISON_OVERREAD=1 -D MOJOLEARN_MAMBA_POISON_NOBAND=1"
MAMBA_SO=$(find python/mojolearn -name _mojolearn_mamba.so | head -1)
[ -n "$MAMBA_SO" ] || { echo "no built _mojolearn_mamba.so under python/mojolearn; build the set first" >&2; exit 2; }
SETDIR=$(dirname "$MAMBA_SO")
WORK=$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-poison.XXXXXX")
[ "${MOJOLEARN_POISON_KEEP:-0}" = 1 ] || trap 'rm -rf "$WORK"' EXIT INT TERM
PKG="$WORK/python"
mkdir -p "$PKG"
# the package copy: python sources copied, every binding hard-linked (cp on
# a filesystem that refuses links), the Mamba binding left out
(cd python && find mojolearn -type d -not -path '*/__pycache__*' -exec mkdir -p "$PKG/{}" \;)
(cd python && find mojolearn -type f -not -name '*.so' -not -path '*/__pycache__*' -exec cp -p {} "$PKG/{}" \;)
(cd python && find mojolearn -type f -name '*.so' -not -name _mojolearn_mamba.so | while read -r f; do ln "$f" "$PKG/$f" 2>/dev/null || cp -p "$f" "$PKG/$f"; done)
echo "poison gate: defines [$DEFINES] set $SETDIR record $RECORD copy $PKG"
MOJOLEARN_MAMBA_OUTDIR="$WORK/$SETDIR" MOJOLEARN_MAMBA_DEFINES="$DEFINES" sh bindings/build_mamba.sh
[ -f "$WORK/$SETDIR/_mojolearn_mamba.so" ] || { echo "poison gate: no binding at $WORK/$SETDIR/_mojolearn_mamba.so after the build" >&2; exit 2; }
cmp -s "$WORK/$SETDIR/_mojolearn_mamba.so" "$MAMBA_SO" && { echo "poison gate: the poison binding is byte-equal to the production one; the defines did not reach the build" >&2; exit 2; }
COMMIT=$(git rev-parse HEAD 2>/dev/null || cat commit.txt)
VENDOR_ARG=""
[ -n "${MOJOLEARN_BOX_LABEL:-}" ] && VENDOR_ARG="--vendor $MOJOLEARN_BOX_LABEL"
OUT="$WORK/poison.json"
env PYTHONPATH="$PKG" MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_COMMIT="$COMMIT" \
    pixi run python tools/identity_break.py --lanes "$LANES" $VENDOR_ARG --json "$OUT"
set +e
pixi run python tools/identity_break.py --diff $COLUMNS "$OUT" --lanes "$LANES" --require-columns 3 > "$WORK/diff.txt"
RC=$?
# the diff prints every lane of the record; the Mamba rows and the verdict lines are the gate's
grep -E "^\| mamba|^summary|^require-columns|DIVERGENT|MOVED|ONE-COLUMN" "$WORK/diff.txt"
set -e
# only the Mamba rows decide: the record's other lanes are not this gate's
# (2026-09-14_120-lanes-2711flip carries byte-lm-resident MOVED x9, a lane
# bug of that day, and the diff's exit code counts it)
DIVERGENT=$(grep -E "^\| mamba" "$WORK/diff.txt" | grep -c -E "DIVERGENT|MOVED|REFUSED|ONE-COLUMN" || true)
grep -q "^require-columns 3 over .*: OK" "$WORK/diff.txt" || DIVERGENT=$((DIVERGENT + 1))
# the pass condition is POSITIVE: 4 lanes x 9 fixtures = 36 training rows
# reading IDENTICAL x4 (three committed columns plus this one). A refused or
# absent column shows as fewer rows, never as a pass (a gate that cannot fail
# is not a gate; the first run of this script passed with 36 REFUSED cells).
IDENT4=$(grep -E "^\| mamba[0-9a-z/_-]* +\| IDENTICAL x4" "$WORK/diff.txt" | wc -l | tr -d ' ')
echo "poison column: $IDENT4 of 36 Mamba training rows IDENTICAL x4"
if [ "$SABOTAGE" = 1 ] || [ "$SABOTAGE" = 2 ]; then
    if [ "$IDENT4" -ne 36 ] && [ "$DIVERGENT" -gt 0 ]; then
        echo "check-mamba-poison SABOTAGE $SABOTAGE: the gate FAILED as required ($DIVERGENT Mamba rows not identical, $IDENT4 of 36 identical)"; exit 0
    fi
    echo "check-mamba-poison SABOTAGE $SABOTAGE: the gate PASSED; it does not reach the planted defect" >&2; exit 1
fi
if [ "$SABOTAGE" = 3 ]; then
    if [ "$DIVERGENT" -eq 0 ] && [ "$IDENT4" -eq 36 ]; then
        echo "check-mamba-poison SABOTAGE 3 (control): the planted over-read is INVISIBLE without the band, as expected"; exit 0
    fi
    echo "check-mamba-poison SABOTAGE 3 (control): the planted over-read moved a hash WITHOUT the band (recycled memory held NaN, or the read is not past the end); rerun cold" >&2; exit 1
fi
if [ "$DIVERGENT" -gt 0 ] || [ "$IDENT4" -ne 36 ]; then
    echo "check-mamba-poison FAILED: a Mamba lane reads a device element nobody wrote ($DIVERGENT Mamba rows not identical, $IDENT4 of 36 identical; diff exit $RC); column at $OUT" >&2
    [ "${MOJOLEARN_POISON_KEEP:-0}" = 1 ] || echo "  (set MOJOLEARN_POISON_KEEP=1 to keep the column and the copy)" >&2
    exit 1
fi
echo "check-mamba-poison PASSED: $LANES cold on the poison build equal the three committed columns of $RECORD"
