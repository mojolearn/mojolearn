#!/bin/bash
# tools/identity_box_job_cpu_ptx.sh -- the Linux CPU column and (when a PTX set is supplied) the PTX
# column of the reference table, recorded in an ALREADY BUILT NVIDIA tree, as a PERSISTED box job:
#   tools/nvidia_central.sh sync identity <worktree holding this script>   (after the identity job ends)
#   tools/nvidia_central.sh submit identity --cap 240 tools/identity_box_job_cpu_ptx.sh
# The tree is the one `tools/identity_box_job.sh nvidia` (tools/box_merged_verify.sh nvidia) leaves
# behind: python/mojolearn/identical (the native merged set), python/mojolearn/host (the host set) and
# python/mojolearn/.libs/libMojolearnMath.so, activated at the tree's HEAD. This job compiles nothing.
#
#   cpu  CPU_THREADS=3 REPEATS=1 tools/record_identity_column.sh cpu <out>/cpu: incremental by
#        default (record_identity_column.sh's auto reuse: only the lanes whose closure changed since
#        the cpu class's newest good evidence; FULL=1 is never set here).
#   ptx  RETIRED 2026-11-05 (one NVIDIA column, recorded on the PTX set by box_merged_verify.sh); formerly
#        only with PTX_SET_TGZ: a cuda-sm_80 PTX set tarball (entries cuda/sm_80/..., the release
#        leg's set, as ~/mojolearn-evidence/rel0864/sets/cuda-sm_80.tar.gz), placed on the box first,
#        with PTX_SET_SHA256 (its sha256) and PTX_SET_COMMIT (the commit the set was built from; it
#        must be HEAD or code-equal to HEAD by tools/identity_code_equal.py). The box_merged_verify
#        tree carries no PTX set: it builds only the native set for the box's own architecture. The
#        set is unpacked into python/mojolearn/cuda/sm_80 AFTER the cpu record (a cuda/ directory
#        with binaries switches the loader to the wheel layout), the column is recorded as
#        nvidia-ptx-<gpu>-sm80, and the set is moved out of the tree again on exit.
#
# Environment: ID_SCOPE (SCOPE for both records, default routine, as box_merged_verify records),
# EXPECT_COMMIT (refuse unless HEAD is this commit), MOJOLEARN_IDENTITY_REUSE_RECORDS (passed through).
# Results: a NEW /root/results-cpu-ptx-* directory outside the tree, mirrored into
# $MOJOLEARN_RESULTS_DIR/cpu-ptx every two minutes and at the end. Summary lines:
#   identity_box_job cpu rc=<rc> ...
#   identity_box_job ptx rc=<rc>|SKIPPED ...
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd -P)
cd "$ROOT" || exit 2
RES=${MOJOLEARN_RESULTS_DIR:-$ROOT/box-out/identity-cpu-ptx}
O=/root/results-cpu-ptx-$(date -u +%Y%m%dT%H%M%SZ)-$$
mkdir -p "$RES" && mkdir "$O" || { echo "identity_box_job_cpu_ptx: cannot create $RES or $O" >&2; exit 2; }
export PATH="${BOX_TOOL_PATH:-$HOME/.pixi/bin}:$PATH"
mirror() { mkdir -p "$RES/cpu-ptx"; rsync -a "$O/" "$RES/cpu-ptx/" 2>/dev/null || cp -r "$O/." "$RES/cpu-ptx/"; }
summary() { echo "$*" | tee -a "$O/job-summary.txt" "$RES/job-summary.txt"; }
st() { echo "$(date -u +%H:%M:%SZ) $*" | tee -a "$O/status.txt"; }
PTX_ASIDE=''
cleanup() {
    # put the tree back as box_merged_verify left it: no cuda/ directory
    if [ -n "$PTX_ASIDE" ] && [ -d python/mojolearn/cuda ]; then
        mv python/mojolearn/cuda "$PTX_ASIDE" && st "PTX set moved out of the tree to $PTX_ASIDE"
    fi
    kill "${MIRROR:-}" 2>/dev/null
    mirror
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
( while true; do sleep 120; mirror; done ) &
MIRROR=$!

HEAD=$(git rev-parse HEAD) || { st "REFUSED not a git checkout"; exit 3; }
echo "$HEAD" > "$O/commit.txt"; cp "$O/commit.txt" "$RES/commit.txt"
st "START cpu+ptx at $HEAD in $ROOT; outputs $O"
[ -z "${EXPECT_COMMIT:-}" ] || [ "$HEAD" = "$(git rev-parse "$EXPECT_COMMIT^{commit}" 2>/dev/null)" ] \
    || { st "REFUSED HEAD $HEAD is not EXPECT_COMMIT $EXPECT_COMMIT"; exit 3; }
# never beside a running qualification of this checkout (box_merged_verify holds this lock)
exec 9> "$(git rev-parse --git-path box-merged-verify.lock)" || { st "REFUSED cannot open the checkout lock"; exit 3; }
flock -n 9 || { st "REFUSED another qualification runner holds this checkout"; exit 3; }
has_so() { [ -d "$1" ] && [ -n "$(find "$1" -name '*.so' -print -quit)" ]; }
for d in python/mojolearn/identical python/mojolearn/host; do
    has_so "$d" || { st "REFUSED $d has no binaries: run tools/identity_box_job.sh nvidia in this tree first"; exit 3; }
done
[ -f python/mojolearn/.libs/libMojolearnMath.so ] || { st "REFUSED no host math helper (python/mojolearn/.libs/libMojolearnMath.so)"; exit 3; }
! has_so python/mojolearn/cuda || { st "REFUSED python/mojolearn/cuda already holds binaries; a native or PTX set there would shadow the built tree"; exit 3; }
command -v pixi > /dev/null || { st "REFUSED pixi missing"; exit 3; }
SCOPE=${ID_SCOPE:-routine}; export SCOPE
unset FULL

# --- the cpu column ------------------------------------------------------------------------
st "CPU record (scope $SCOPE, CPU_THREADS=3, REPEATS=1, incremental)"
CPU_THREADS=3 REPEATS=1 bash tools/record_identity_column.sh cpu "$O/cpu" > "$O/cpu-record.log" 2>&1
crc=$?
mirror
summary "identity_box_job cpu rc=$crc $(grep -m 1 -E '^(REUSED|NOTHING-CHANGED)' "$O/cpu-record.log" | cut -c1-120) | $(tail -n 1 "$O/cpu-record.log" | cut -c1-200)"

# --- the PTX column: RETIRED (Andrew 2026-11-05) -------------------------------------------
# One NVIDIA column, recorded on the sm_80 PTX set by tools/box_merged_verify.sh (PTX_SET_TGZ there).
# This job records the cpu column only; a PTX_SET_TGZ given here is ignored.
prc=SKIPPED
summary "identity_box_job ptx rc=SKIPPED the PTX column is retired; the NVIDIA column records on the PTX set"
st "DONE cpu=$crc ptx=$prc"
[ "$crc" = 0 ] && { [ "$prc" = 0 ] || [ "$prc" = SKIPPED ]; }
