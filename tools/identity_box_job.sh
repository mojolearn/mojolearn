#!/bin/bash
# tools/identity_box_job.sh nvidia|amd -- the merged-bindings build, verify and
# incremental reference-column record of the tree it runs in, as a PERSISTED
# box job: tools/nvidia_central.sh submit <lane> tools/identity_box_job.sh
# (wrap the vendor in a one-line script) or tools/do_amd_steward.sh run.
# Results go under $MOJOLEARN_RESULTS_DIR (box_persist_run sets it; persists to
# R2 from the box), in a new merged-<vendor> directory that
# tools/box_merged_verify.sh requires.
set -u
V=${1:?nvidia|amd}
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
cd "$ROOT" || exit 2
RES=${MOJOLEARN_RESULTS_DIR:-$ROOT/box-out/identity-$V}
mkdir -p "$RES"
export BOX_REPO_DIR=$ROOT
# box_merged_verify refuses an output dir inside the checkout; it writes
# outside, and a mirror loop copies it into the persisted results dir every
# two minutes and once more at the end (so an interrupted run keeps its logs).
export BOX_OUT_DIR=/root/results-merged-$V-$(date -u +%Y%m%dT%H%M%SZ)
export PATH="$HOME/.pixi/bin:$PATH"
git rev-parse HEAD > "$RES/commit.txt"
mirror() { rsync -a --delete "$BOX_OUT_DIR/" "$RES/merged-$V/" 2>/dev/null || cp -r "$BOX_OUT_DIR/." "$RES/merged-$V/"; }
( while true; do sleep 120; mkdir -p "$RES/merged-$V"; mirror; done ) &
MIRROR=$!
bash tools/box_merged_verify.sh "$V" "$(git rev-parse HEAD)"
rc=$?
kill "$MIRROR" 2>/dev/null
mkdir -p "$RES/merged-$V"; mirror
echo "identity_box_job $V rc=$rc out=$BOX_OUT_DIR mirrored=$RES/merged-$V" | tee -a "$RES/job-summary.txt"
exit $rc
