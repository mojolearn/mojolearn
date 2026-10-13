#!/bin/bash
# release_leg_changed_lanes.sh <cuda|hip> <arch> <release-build OUT>
#
# ON THE RELEASE BUILD LEG'S OWN GPU BOX, right after its bindings built (tools/release061_remote_build.sh
# calls it last). Andrew, 2026-10-13: fold light verification of what changed into compiling, so a release
# needs no separate GPU rental for re-records. Only what changed, ONE run (REPEATS=1).
#
#   1. stages the set the leg just built (<OUT>/build/sets/<vendor>/<arch>: the tiers, host/ and .libs/)
#      into python/mojolearn/, the layout of a built tree; the sets' RUNPATHs ($ORIGIN/.libs,
#      $ORIGIN/../.libs) resolve there exactly as in the wheel. Copies: the set itself is untouched
#   2. tools/changed_lanes_check.sh <OUT>/changed-lanes <vendor> with LANES=$MOJOLEARN_RELEASE_CHANGED_LANES
#      and RECORD_LABEL: the identity record of the changed lanes, then verify --lanes against the
#      committed table, one line per lane in <OUT>/changed-lanes/summary.txt
#
# Everything lands under <OUT>, which every leg already brings home with its proof; tools/release.py's
# identity-admit step admits the records (NVIDIA == AMD, 0 conflicts) before linux-pack.
# The label: MOJOLEARN_RELEASE_CHANGED_LABEL, else nvidia-<gpu>-<arch> from nvidia-smi (the NVIDIA leg walks
# GPU types on stock) or amd-gpu-<arch>.
# Never fails the build: the caller ignores this exit code and the admit reads the evidence.
set -uo pipefail
[ $# -eq 3 ] || { echo "usage: release_leg_changed_lanes.sh <cuda|hip> <arch> <release-build OUT>" >&2; exit 2; }
VENDOR=$1; ARCH=$2; OUT=$3
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT" || exit 2
DEST=$OUT/changed-lanes
mkdir -p "$DEST" || exit 2
LANES=${MOJOLEARN_RELEASE_CHANGED_LANES:-}
if [ -z "$LANES" ]; then
    echo "NOTHING-CHANGED column=$VENDOR: no changed lanes were sent to this leg" | tee -a "$DEST/summary.txt"
    exit 0
fi
SET=$OUT/build/sets/$VENDOR/$ARCH
[ -d "$SET" ] || { echo "CHANGED-LANES column=$VENDOR NOT-RUN: no built set at $SET" | tee -a "$DEST/summary.txt"; exit 1; }

# --- 1. stage the built set as a built tree -------------------------------------------
PKG=$ROOT/python/mojolearn
mkdir -p "$PKG/identical" "$PKG/host" "$PKG/.libs"
staged=0
for f in "$SET"/*.so; do [ -f "$f" ] && cp -p "$f" "$PKG/" && staged=$((staged + 1)); done
for d in identical host; do
    for f in "$SET/$d"/*.so; do [ -f "$f" ] && cp -p "$f" "$PKG/$d/" && staged=$((staged + 1)); done
done
[ -d "$SET/.libs" ] && cp -p "$SET"/.libs/* "$PKG/.libs/" 2>/dev/null
echo "staged=$staged from $SET" > "$DEST/staged.txt"

# --- 2. the label ------------------------------------------------------------------------------
LABEL=${MOJOLEARN_RELEASE_CHANGED_LABEL:-}
if [ -z "$LABEL" ]; then
    if [ "$VENDOR" = cuda ]; then
        gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n 1 | tr 'A-Z' 'a-z' \
              | sed -e 's/^nvidia //' -e 's/geforce //' -e 's/[^a-z0-9]\{1,\}/-/g' -e 's/^-//' -e 's/-$//')
        LABEL="nvidia-${gpu:-gpu}-$ARCH"
    else
        LABEL="amd-gpu-$ARCH"
    fi
fi
echo "# changed lanes on $VENDOR/$ARCH, label $LABEL"

# --- 3. record + verify, once --------------------------------------------------------------
LANES="$LANES" RECORD_LABEL="$LABEL" PYTHON="$ROOT/.pixi/envs/default/bin/python" CPU_THREADS=2 \
    bash "$ROOT/tools/changed_lanes_check.sh" "$DEST" "$VENDOR"
rc=$?
echo "changed_lanes_exit=$rc label=$LABEL" >> "$DEST/staged.txt"
exit $rc
