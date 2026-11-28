#!/bin/bash
# record_identity_column.sh <vendor-label> <outdir> [--models]
#
# ONE ADMISSIBLE COLUMN RECORD for the verifier's reference table
# (python/mojolearn/verify_reference/table.json), written the way
# _verify_reference.admit() and docs/VERIFY.md "Regenerating the reference
# table" require: tools/identity_break.py --json in the identical tier, a
# commit witness, the default fixture size, one device (par_devices 0), no
# sabotage, every part collected (never --partial-column), every fixture,
# --repeats 1 (one sample per part; the second witness is the other GPU
# vendor's column, never a second fit here), input and held-out witnesses
# and the batch/property protocols recorded by the harness itself.
#
# Runs in a BUILT checkout of a committed tree (every binding built for this
# box; the host bindings too for the cpu column). Runs nothing else. The
# orchestrator runs it on the box; lanes never do.
#
#   <vendor-label>  nvidia-<gpu>-<arch>  -> --require-backend cuda
#                   nvidia-ptx-<gpu>-sm80 -> --require-backend cuda on the PTX set
#                                           (MOJOLEARN_GPU_ARCH=sm_80): THE PTX COLUMN,
#                                           Andrew 2026-10-10: PTX is a normal target; no
#                                           flag. Recorded ONCE on any NVIDIA box; the tree
#                                           must carry the PTX set at python/mojolearn/cuda/sm_80
#                                           (the cuda-sm_80 leg's sets/cuda/sm_80). The
#                                           admission refuses a PTX digest that differs
#                                           from the NVIDIA == AMD reference
#                   amd-<gpu>-<arch>     -> --require-backend hip
#                   apple-<chip>         -> --require-backend metal
#                   cpu                  -> --side cpu: every public call with device="cpu"
#                                           (2026-11-24: no MOJOLEARN_VENDOR=cpu process), on a
#                                           CPU-only runner or on a GPU box alike; the label is
#                                           derived by the harness (cpu-<cpu model slug>), never typed
#                   A GPU label records the GPU column (--side gpu, device="gpu"); with
#                   WITH_CPU=1 the SAME process then records the CPU column too (--side both):
#                   <file>.identical.json is the GPU column, cpu<suffix>.identical.json the CPU one
#   <outdir>        an ABSOLUTE directory OUTSIDE the tree (an lq CMD tree is
#                   deleted after the job); created if missing
#   --models        instead of a column: save the portable models the verifier
#                   bundle ships (verify --emit-models) on this GPU install,
#                   against the reference table committed in THIS tree. Run it
#                   only after admit_identity_columns.sh's table is committed
#                   and pushed; it writes <outdir>/models/ (files + models.json)
#
# Environment:
#   SCOPE=routine|all       lanes (default routine = what `verify` runs by
#                           default; `all` adds the neural-training lanes)
#   SHARD=i/N               only every N-th routine lane from i, into its own
#                           complete record <label>.s<i>of<N>.identical.json;
#                           each shard is admissible alone
#   LANES=a,b               an explicit lane list instead of SCOPE/SHARD
#   REPEATS=1               fits per cell (default 1). Andrew 2026-10-10: identity runs ONCE; a mismatch is a bug to fix, never a reason to rerun.
#                           NVIDIA == AMD is the identity rule; a second fit on
#                           one box adds nothing the other vendor's column does not
#   (default)               THE FAMILIES THAT RECORD (AGENTS.md part 4, Andrew 2026-11-28): with no
#                           LANES, SHARD or ONLY_CHANGED, the lanes of the release families that
#                           record (tools/release_families.py: a family records when one of its
#                           binding identities changed, it shares a merged file with one that did,
#                           shared changed, or its kept columns do not rest on the shipped bytes);
#                           every other lane keeps its admitted rows. NOTHING-CHANGED (exit 0) when
#                           no family records. The plan is HEAD's against the previous release
#                           (<outdir>/<label>.family-plan.json); it needs the release evidence set
#                           and the history of the previous release's commit in this tree
#   FAMILIES=a,b            the families to record instead of the plan (tools/release.py passes its own
#                           decision); FAMILIES=none records nothing
#   REBUILD_ALL=1           every lane of the scope (tools/release.py --rebuild-all). REUSE=1 and
#                           SELECT_ONLY=1 (the per-lane selector) are removed and refused; FULL=1 is retired
#   ONLY_CHANGED=table|<sha> INCREMENTAL RE-RECORD: only the lanes whose source
#                           closure (tools/lane_select.py) changed between the
#                           commit each lane's admitted rows rest on (`table`)
#                           or one named <sha>, and HEAD; re-record
#                           only what changed; every other row is kept (the
#                           table builder takes the newest commit per cell, so
#                           the unchanged lanes keep their admitted rows).
#                           Prints NOTHING-CHANGED and exits 0 when no lane moved
#   WITH_CPU=1              with a GPU label: the CPU column from the same process (--side both,
#                           device="cpu" after the GPU column's device="gpu"), written as
#                           cpu<the GPU record's suffix>.identical.json beside it, with its own
#                           provenance, check line and RECORDED line. Excludes ONLY_CHANGED (its
#                           lane selection is per device class: record the cpu label on its own);
#                           the family selection is the same on every class
#   CPU_THREADS=3           every CPU thread pool (MOJOLEARN_CPU_THREADS, OMP...)
#   PYTHON=...              the interpreter (default: `pixi run -e default python`
#                           when pixi and pixi.toml are present, else python3)
#   ALLOW_DIRTY=1           diagnostic recording from modified tracked files;
#                           guarded release launches refuse this override
#   MOJOLEARN_IDENTITY_LAUNCH  optional guarded release launch from
#                           release_identity_run.py; validated when supplied
#   RECORD_FROM_INSTALLED=1 THE RECORD IS THE GATE (Andrew 2026-11-05: one execution of the identity
#                           lanes per device class, on the built release wheels). Default off. The lanes
#                           run against the mojolearn INSTALLED in PYTHON's environment (the exact release
#                           wheel bytes), never against $ROOT/python: PYTHONPATH is not extended, the
#                           preflight refuses a package under $ROOT or outside site-packages, and the
#                           installed identity_columns/COMMIT must equal the tree commit (tree_commit.sh; the harness in tools/ is the
#                           same source). Needs PYTHON (the venv interpreter) and INSTALLED_WHEELS (a file of
#                           "<wheel file name> <sha256>" lines the installer verified); the digests land in the
#                           provenance as installed_wheels=. The lane selection (every lane, or REUSE=1) is unchanged.
#   IDENTITY_RECORD_BOX     <provider>:<id> of this machine (runpod:<pod id>, digitalocean:<droplet id>,
#                           github:<run id>/<job>, mac:<host>), written to the record's run.box and the
#                           provenance's run_box= line. Unset: the harness derives it (GITHUB_RUN_ID/GITHUB_JOB,
#                           RUNPOD_POD_ID, the Mac's hostname hash) or writes null with the reason
#   IDENTITY_RECORD_WHEELS  set here from INSTALLED_WHEELS under RECORD_FROM_INSTALLED=1 (never by hand): the
#                           record's run.wheels [{name, sha256, version}]. Neither variable starts with MOJO, so
#                           the resume signature never sees them; the run block is provenance, never hashed
#   LANE_SELECTION=family-chunk  with LANES: one chunk of the recording families' lanes (identity_wheel_record.py
#                           --record-families a,b: tools/release.py's family plan)
#   LANE_SELECTION=full-chunk  with LANES: one chunk of a full column (every lane; --record-families all,
#                           tools/release.py --rebuild-all)
#                           (tools/identity_wheel_record.py runs the chunks in parallel)
#
# Start ledger-managed release columns through release_identity_run.py after
# reconciling retained evidence. Legacy per-run controllers remain supported.
# The ledger retains finished work and refuses duplicate active runs. A refused resume is retained
# for diagnosis; it never silently starts every completed fit again.
#
# Output, per record: <outdir>/<file>.identical.json, its .log, an IDCOLUMN
# line from tools/identity_columns.py check (admissible=yes|NO), and
# <outdir>/<file>.provenance.txt. The last line of stdout per record is
#   RECORDED <file> rc=<harness rc> admissible=<yes|NO>
# (two such lines with WITH_CPU=1: the GPU record, then the CPU one; exit 1
# unless both are admissible)
set -uo pipefail

die() { echo "record_identity_column: $*" >&2; exit 2; }
[ $# -ge 2 ] || die "usage: record_identity_column.sh <vendor-label> <outdir> [--models]"
LABEL=$1; OUT=$2; MODE=${3:-column}
case $MODE in column|--models) ;; *) die "unknown third argument $MODE (only --models)";; esac
case $OUT in /*) ;; *) die "<outdir> must be absolute (an lq CMD tree is deleted after the job)";; esac

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT" || die "cannot enter $ROOT"
case $OUT in "$ROOT"|"$ROOT"/*) die "<outdir> must be outside the tree $ROOT";; esac

PTX=0
case $LABEL in
    nvidia-ptx-*) BACKEND=cuda; PTX=1 ;;
    nvidia-*) BACKEND=cuda ;;
    amd-*) BACKEND=hip ;;
    apple-*) BACKEND=metal ;;
    cpu) BACKEND=cpu ;;
    *) die "label $LABEL: must start with nvidia-, amd-, apple- or be exactly cpu" ;;
esac
[[ $LABEL =~ ^[a-z0-9][a-z0-9_.-]*$ ]] || die "label $LABEL: lowercase letters, digits, _ . - only"
if [ "$MODE" = --models ] && [ "$BACKEND" = cpu ]; then
    die "--models saves GPU-trained models; run it on a GPU box"
fi
WITH_CPU=${WITH_CPU:-0}
case $WITH_CPU in 0|1) ;; *) die "WITH_CPU is 0 or 1, not $WITH_CPU";; esac
if [ "$WITH_CPU" = 1 ]; then
    [ "$BACKEND" != cpu ] || die "WITH_CPU=1 adds the CPU column to a GPU label's record; the cpu label records it alone"
    [ -z "${ONLY_CHANGED:-}" ] \
        || die "WITH_CPU=1 excludes ONLY_CHANGED (its selection is per device class); record the cpu label on its own"
fi
mkdir -p "$OUT" || die "cannot create $OUT"

# --- the interpreter -------------------------------------------------------
if [ -n "${PYTHON:-}" ]; then
    read -r -a PY <<< "$PYTHON"
elif command -v pixi > /dev/null 2>&1 && [ -f pixi.toml ]; then
    PY=(pixi run -e default python)
else
    PY=(python3)
fi

# Validate a supplied release claim before any product import. Do not impose a
# new ledger on existing remote controllers until their orchestration migrates.
if [ "$MODE" = column ] && [ -n "${MOJOLEARN_IDENTITY_LAUNCH:-}" ]; then
    "${PY[@]}" "${MOJOLEARN_IDENTITY_GUARD:-$ROOT/tools/release_identity_run.py}" validate \
        --launch "$MOJOLEARN_IDENTITY_LAUNCH" --root "$ROOT" --label "$LABEL" --out "$OUT" \
        || die "release identity launch validation failed"
fi

# --- the environment the record claims -------------------------------------
for v in MOJOLEARN_IDENTITY_N MOJOLEARN_IDENTITY_WIDE MOJOLEARN_PAR_DEVICES; do
    [ -z "${!v:-}" ] || die "$v is set; a reference record runs the default fixture on one device"
done
for v in $(env | sed -n 's/^\(MOJOLEARN_[A-Z0-9_]*SABOTAGE[A-Z0-9_]*\)=.*/\1/p'); do
    [ -z "${!v:-}" ] || [ "${!v}" = 0 ] || die "$v is set; a sabotage run is never a record"
done
export MOJOLEARN_NUMERIC_MODE=identical PYTHONUNBUFFERED=1
INSTALLED=${RECORD_FROM_INSTALLED:-0}
case $INSTALLED in 0|1) ;; *) die "RECORD_FROM_INSTALLED is 0 or 1, not $INSTALLED";; esac
INSTALLED_LINE=
if [ "$INSTALLED" = 1 ]; then
    [ -n "${PYTHON:-}" ] || die "RECORD_FROM_INSTALLED=1 needs PYTHON=<the venv interpreter the wheels are installed in>"
    [ -n "${INSTALLED_WHEELS:-}" ] && [ -f "$INSTALLED_WHEELS" ] \
        || die "RECORD_FROM_INSTALLED=1 needs INSTALLED_WHEELS=<file of '<wheel> <sha256>' lines>"
    # no regex interval ({64}): mawk (Debian/Ubuntu's default awk) does not support it
    INSTALLED_LINE=$(awk 'NF == 2 && length($2) == 64 && $2 ~ /^[0-9a-f]+$/ {printf "%s%s:%s", (n++ ? "," : ""), $1, $2} END {if (!n) exit 1}' \
        "$INSTALLED_WHEELS") || die "INSTALLED_WHEELS $INSTALLED_WHEELS holds no '<wheel> <sha256>' line"
    unset PYTHONPATH   # the installed package, never the source tree's python/
    # the record's run.wheels (tools/identity_break.py recorded_wheels): the exact files installed
    export IDENTITY_RECORD_WHEELS="$INSTALLED_WHEELS"
else
    export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
fi
T=${CPU_THREADS:-3}
for v in MOJOLEARN_CPU_THREADS OMP_NUM_THREADS OMP_THREAD_LIMIT OPENBLAS_NUM_THREADS MKL_NUM_THREADS \
         VECLIB_MAXIMUM_THREADS NUMEXPR_NUM_THREADS NUMEXPR_MAX_THREADS BLIS_NUM_THREADS; do
    export "$v=$T"
done
export OMP_MAX_ACTIVE_LEVELS=1 OMP_DYNAMIC=FALSE
# THE SIDE IS THE PUBLIC device= PARAMETER (2026-11-24): the cpu label runs the
# harness with --side cpu in this installation's own process (no
# MOJOLEARN_VENDOR=cpu); a GPU label runs --side gpu (or both, WITH_CPU=1).
# THE PTX COLUMN: the PTX set by name, like any architecture directory; a native
# column must not inherit a forced architecture from the caller.
if [ "$PTX" = 1 ]; then export MOJOLEARN_GPU_ARCH=sm_80; elif [ "$BACKEND" = cuda ]; then unset MOJOLEARN_GPU_ARCH; fi
# identity_break refuses a Metal matrix of more than one lane unless it is an
# intentional full run; a reference column is exactly that.
if [ "$BACKEND" = metal ]; then export MOJOLEARN_APPLE_FULL_DIAGNOSTIC=1; fi

HEAD_COMMIT=$(git rev-parse HEAD 2>/dev/null) || die "not a git checkout; the record needs a commit witness"
# THE TREE'S COMMIT (rel-fix-1, 2026-11-26): a box tree patch-synced by tools/dev_pod.sh has its HEAD at the merge
# base with main and the synced commit in .git/devpod_synced; `git rev-parse HEAD` named main, so every chunk of the
# L40S record refused the wheel ("installed wheel source 9bdb1a9e8c34, harness tree 3423ae229e0b", rc 2). The commit
# is tools/ops/tree_commit.sh's, the one rule every box job reads. For a synced tree the patch is the difference
# from HEAD by design, so the source check is the sync record's own (worktree_dirty=0: the synced commit is exactly
# what was shipped); MOJOLEARN_COMMIT carries it to the harness (tools/identity_break.py commit_witness).
. "$ROOT/tools/ops/tree_commit.sh"
COMMIT=$(tree_commit "$ROOT")
[ -n "$COMMIT" ] || die "no commit witness for $ROOT"
if [ "$COMMIT" != "$HEAD_COMMIT" ]; then
    SYNC_DIRTY=$(sed -n 's/^worktree_dirty=//p' "$(git rev-parse --absolute-git-dir)/devpod_synced" | head -n 1)
    if [ "$SYNC_DIRTY" != 0 ] && [ "${ALLOW_DIRTY:-0}" != 1 ]; then
        die "the tree was synced from a worktree with uncommitted changes over $COMMIT (ALLOW_DIRTY=1 overrides)"
    fi
    export MOJOLEARN_COMMIT=$COMMIT
else
    # A build may touch a lock file or a generated table; a changed SOURCE file
    # means the record would name a commit that is not what ran.
    DIRTY=$(git status --porcelain --untracked-files=no 2>/dev/null | awk '{print $NF}' | tr '\n' ' ')
    SRC_DIRTY=$(tr ' ' '\n' <<< "$DIRTY" | grep -E '^(python/|tools/|bindings/|.*\.mojo$)' \
                | grep -v -x 'tokenizer/impl/unicode_table_generated.mojo' | tr '\n' ' ')
    if [ -n "${SRC_DIRTY// /}" ] && [ "${ALLOW_DIRTY:-0}" != 1 ]; then
        die "source files differ from $COMMIT: $SRC_DIRTY(ALLOW_DIRTY=1 overrides)"
    fi
fi

# --- preflight: the loaded backend and tier, before any fit ----------------
if [ "$INSTALLED" = 1 ]; then
    where=$("${PY[@]}" - "$ROOT" "$COMMIT" <<'PYEOF' 2>&1 | tail -1
import pathlib, sys
import mojolearn
root, commit = pathlib.Path(sys.argv[1]).resolve(), sys.argv[2]
pkg = pathlib.Path(mojolearn.__file__).resolve().parent
assert not pkg.is_relative_to(root), "imported %s from the source tree" % pkg
assert "site-packages" in pkg.parts, "%s is not an installed package" % pkg
got = (pkg / "identity_columns" / "COMMIT").read_text().strip()
assert got == commit, "installed wheel source %s, harness tree %s" % (got[:12], commit[:12])
print("INSTALLED", pkg)
PYEOF
)
    case $where in "INSTALLED "*) echo "record_identity_column: lanes run against the $where" ;;
        *) die "preflight (RECORD_FROM_INSTALLED=1): $where" ;; esac
fi
got=$("${PY[@]}" -c 'import mojolearn as m, mojolearn._backend as b; print(m.vendor(), b.numeric_mode())' 2>&1 | tail -1)
if [ "$BACKEND" = cpu ]; then
    # The CPU column runs through device="cpu" on whatever this installation
    # loaded (a CPU-only runner, or a GPU box whose host bindings are built).
    case "$got" in *" identical") ;; *) die "preflight: expected the identical tier, loaded '$got' (are the bindings built in $ROOT?)";; esac
else
    [ "$got" = "$BACKEND identical" ] || die "preflight: expected '$BACKEND identical', loaded '$got' (are the bindings built in $ROOT?)"
fi
if [ "$BACKEND" = cuda ]; then
    set_got=$("${PY[@]}" -c 'import mojolearn as m, mojolearn._backend as b; print(b.gpu_arch(), "ptx" if b.baseline_selection_receipt() else "native")' 2>&1 | tail -1)
    if [ "$PTX" = 1 ]; then
        [ "$set_got" = "sm_80 ptx" ] || die "preflight: a PTX column needs the PTX set loaded (sm_80 ptx), loaded '$set_got' (is the cuda-sm_80 leg's set at python/mojolearn/cuda/sm_80?)"
    else
        case "$set_got" in *" native") ;; *) die "preflight: a native NVIDIA column loaded '$set_got'; label a PTX record nvidia-ptx-<gpu>-sm80" ;; esac
    fi
fi

if [ "$MODE" = --models ]; then
    M=$OUT/models
    [ -e "$M" ] && mv "$M" "$M.old-$(date -u +%Y%m%dT%H%M%SZ)"
    "${PY[@]}" -m mojolearn verify --emit-models "$M" --cpu-threads "$T" > "$OUT/models.log" 2>&1
    rc=$?
    {
        echo "commit=$COMMIT"; echo "label=$LABEL"; echo "backend=$BACKEND"; echo "date=$(date -u +%FT%TZ)"
        echo "table_sha256=$(shasum -a 256 python/mojolearn/verify_reference/table.json | cut -d' ' -f1)"
    } > "$M/provenance.txt" 2>/dev/null
    n=$(grep -c '"model_hash"' "$M/models.json" 2>/dev/null || echo 0)
    echo "MODELS $LABEL commit=${COMMIT:0:12} rc=$rc saved=$n dir=$M (problems: grep -A3 problems $OUT/models.log)"
    exit $rc
fi

# --- which lanes -----------------------------------------------------------
NAME=$LABEL
SELECTION=explicit
# FAMILY REUSE IS THE DEFAULT (AGENTS.md part 4, "A release rebuilds and re-records
# only the families that changed", Andrew 2026-11-28; tools/release_families.py).
# With no LANES, SHARD or ONLY_CHANGED the column records the lanes of the release
# families that record: FAMILIES=a,b names them (`none`: no family; what
# tools/release.py decides from its plan), else the family plan of HEAD against the
# previous release decides (tools/release_families.py plan + lanes --settle: the
# binding identities of tools/release_reuse.py, whole families, and the kept-column
# check), every other lane keeping its admitted rows. REBUILD_ALL=1 records every
# lane of the scope. Removed 2026-11-28 with the opt-in it served: REUSE=1 (the
# per-lane closure selector) and SELECT_ONLY=1; FULL=1 stays retired.
[ -z "${FULL:-}" ] || die "FULL=$FULL is retired: REBUILD_ALL=1 records every lane"
[ -z "${REUSE:-}" ] || die "REUSE=$REUSE is removed (2026-11-28): family reuse is the default; FAMILIES=a,b names the families, REBUILD_ALL=1 records every lane"
[ -z "${SELECT_ONLY:-}" ] || die "SELECT_ONLY is removed (2026-11-28): tools/release_families.py lanes prints a family selection"
case "${REBUILD_ALL:-0}" in 0|1) ;; *) die "REBUILD_ALL=$REBUILD_ALL: 1 records every lane, 0 (default) the families that record" ;; esac
if [ -z "${ONLY_CHANGED:-}" ] && [ -z "${LANES:-}" ] && [ -z "${SHARD:-}" ]; then
    if [ "${REBUILD_ALL:-0}" = 1 ]; then
        SELECTION=full
        echo "FULL $LABEL: every ${SCOPE:-routine} lane is recorded (REBUILD_ALL=1)"
    else
        SELECTION=family
        "${PY[@]}" tools/identity_columns.py lanes --scope "${SCOPE:-routine}" > "$OUT/$LABEL.scope-lanes.txt" \
            || die "could not list the ${SCOPE:-routine} lanes"
        if [ -n "${FAMILIES+set}" ]; then
            fam_args=(--families "${FAMILIES:-none}")
        else
            "${PY[@]}" tools/release_families.py plan --commit HEAD --json "$OUT/$LABEL.family-plan.json" \
                > "$OUT/$LABEL.family-plan.txt" 2>&1 \
                || die "no family plan for HEAD (see $OUT/$LABEL.family-plan.txt); FAMILIES=a,b or REBUILD_ALL=1"
            fam_args=(--plan "$OUT/$LABEL.family-plan.json" --settle)
        fi
        FAMILY_LANES=$("${PY[@]}" tools/release_families.py lanes "${fam_args[@]}" \
                --lanes-file "$OUT/$LABEL.scope-lanes.txt" 2> "$OUT/$LABEL.family-lanes.txt" | tail -n 1) \
            || die "could not select the families' lanes (see $OUT/$LABEL.family-lanes.txt)"
        grep -m 12 -E '^(FAMILY|== release families)' "$OUT/$LABEL.family-lanes.txt" || true
        if [ -z "$FAMILY_LANES" ]; then
            echo "NOTHING-CHANGED $LABEL: no release family records; every row is kept"
            exit 0
        fi
        ALL_LANES=$(tail -n 1 "$OUT/$LABEL.scope-lanes.txt")
        [ "$FAMILY_LANES" = "$ALL_LANES" ] || LANES=$FAMILY_LANES
    fi
fi
if [ -n "${LANE_SELECTION:-}" ]; then
    case "$LANE_SELECTION" in family-chunk|full-chunk) ;; *) die "LANE_SELECTION is family-chunk or full-chunk";; esac
    [ "$SELECTION" = explicit ] && [ -n "${LANES:-}" ] \
        || die "LANE_SELECTION=$LANE_SELECTION names a chunk given as LANES"
    SELECTION=$LANE_SELECTION
fi
if [ -n "${ONLY_CHANGED:-}" ]; then
    SELECTION=only-changed-$ONLY_CHANGED
    # INCREMENTAL RE-RECORD: re-record only what changed; every other row is kept.
    # The lanes whose source closure (tools/lane_select.py, the release lane
    # selection's map) changed since the commit each lane's admitted rows rest
    # on (ONLY_CHANGED=table), or since one named commit (ONLY_CHANGED=<sha>).
    # admit_identity_columns.sh then rebuilds the table from every committed
    # column, newest commit per cell, so the rows of every other lane are kept.
    [ -z "${LANES:-}" ] && [ -z "${SHARD:-}" ] || die "ONLY_CHANGED excludes LANES and SHARD"
    since_arg=()
    [ "$ONLY_CHANGED" = table ] || since_arg=(--since "$ONLY_CHANGED")
    # The column's GPU backend: a host-binding-only change cannot move a device
    # column's bits, so lane_select leaves it out of an NVIDIA or AMD record.
    case "$LABEL" in
        nvidia-*) since_arg+=(--backend cuda) ;;
        amd-*) since_arg+=(--backend hip) ;;
    esac
    LANES=$("${PY[@]}" tools/identity_columns.py changed-lanes --scope "${SCOPE:-routine}" ${since_arg[@]+"${since_arg[@]}"} \
            2> "$OUT/$LABEL.changed-lanes.txt" | tail -n 1) || die "could not list the changed lanes (see $OUT/$LABEL.changed-lanes.txt)"
    grep -m 1 'CHANGED-LANES' "$OUT/$LABEL.changed-lanes.txt"
    grep -m 1 '^# release floor' "$OUT/$LABEL.changed-lanes.txt" | cut -c1-240 || true
    if [ -z "$LANES" ]; then
        echo "NOTHING-CHANGED $LABEL: no lane's closure moved since the admitted table; every row is kept"
        exit 0
    fi
fi
if [ -n "${LANES:-}" ]; then
    LANE_LIST=$LANES
    NAME="$LABEL.lanes-$(printf '%s' "$LANES" | shasum -a 256 | cut -c1-8)"
else
    shard_arg=()
    if [ -n "${SHARD:-}" ]; then
        [[ $SHARD =~ ^[0-9]+/[0-9]+$ ]] || die "SHARD=$SHARD: expected i/N"
        shard_arg=(--shard "$SHARD")
        NAME="$LABEL.s${SHARD%/*}of${SHARD#*/}"
    fi
    LANE_LIST=$("${PY[@]}" tools/identity_columns.py lanes --scope "${SCOPE:-routine}" "${shard_arg[@]}") \
        || die "could not list the ${SCOPE:-routine} lanes"
fi
[ -n "$LANE_LIST" ] || die "empty lane list"
# Andrew 2026-10-10: identity runs ONCE; a mismatch is a bug to fix, never a reason to rerun.
# One sample per part (the old ">= 2" refusal is gone): the second witness of a
# reference is the other GPU vendor's column, and NVIDIA == AMD is the rule.
REPEATS=${REPEATS:-1}
[ "$REPEATS" -ge 1 ] 2>/dev/null || die "REPEATS=$REPEATS: must be a positive integer"

JSON=$OUT/$NAME.identical.json
LOG=$OUT/$NAME.log
# WITH_CPU=1: the CPU column of the same lanes, from the same process, named
# like the GPU record with the label replaced by cpu.
CPU_NAME=; CPU_JSON=
if [ "$WITH_CPU" = 1 ]; then
    CPU_NAME="cpu${NAME#"$LABEL"}"
    CPU_JSON=$OUT/$CPU_NAME.identical.json
fi
complete_at_commit() {
    [ -f "$1" ] && "${PY[@]}" -c "import json,sys; j=json.load(open(sys.argv[1])); sys.exit(0 if j.get('complete') is True and j.get('commit')==sys.argv[2] else 1)" "$1" "$COMMIT" 2>/dev/null
}
done_already() {
    complete_at_commit "$JSON" && { [ -z "$CPU_JSON" ] || complete_at_commit "$CPU_JSON"; }
}
if done_already; then
    echo "# $JSON${CPU_JSON:+ and $CPU_JSON} already complete at ${COMMIT:0:12}; left untouched"
else
    for f in "$JSON" ${CPU_JSON:+"$CPU_JSON"}; do
        if [ -f "$f" ] && ! "${PY[@]}" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('commit')==sys.argv[2] else 1)" "$f" "$COMMIT" 2>/dev/null; then
            mv "$f" "$f.other-commit-$(date -u +%Y%m%dT%H%M%SZ)"
        fi
    done
    args=(tools/identity_break.py --json "$JSON" --lanes "$LANE_LIST" --repeats "$REPEATS")
    if [ "$BACKEND" = cpu ]; then
        args+=(--side cpu)
    elif [ "$WITH_CPU" = 1 ]; then
        args+=(--require-backend "$BACKEND" --vendor "$LABEL" --side both --cpu-json "$CPU_JSON")
    else
        args+=(--require-backend "$BACKEND" --vendor "$LABEL" --side gpu)
    fi
    if [ -n "${FIXTURES:-}" ]; then die "FIXTURES is not supported: a record covers every fixture"; fi
    {
        echo "# $(date -u +%FT%TZ) record $NAME${CPU_NAME:+ + $CPU_NAME} commit=$COMMIT backend=$BACKEND threads=$T lanes=$(tr ',' '\n' <<< "$LANE_LIST" | wc -l | tr -d ' ')"
    } >> "$LOG"
    # A resume reads every column file it writes: with WITH_CPU=1 only when both
    # exist (a run that died before its CPU column starts both again).
    if [ -f "$JSON" ] && { [ -z "$CPU_JSON" ] || [ -f "$CPU_JSON" ]; }; then
        "${PY[@]}" -u "${args[@]}" --resume >> "$LOG" 2>&1
        rc=$?
        if grep -q "REFUSING TO RESUME" <(tail -n 5 "$LOG"); then
            echo "# resume refused; retained completed work for explicit provenance repair" >> "$LOG"
            die "resume refused; see $LOG (no completed fits restarted)"
        fi
    else
        "${PY[@]}" -u "${args[@]}" >> "$LOG" 2>&1
        rc=$?
    fi
    echo "# harness rc=$rc" >> "$LOG"
fi
rc=${rc:-0}

# provenance, check and retention of one record: $1 name, $2 json, $3 label, $4 backend
finish_record() {
    local name=$1 json=$2 label=$3 backend=$4 line adm
    {
        echo "commit=$COMMIT"
        echo "label=$label"
        echo "backend=$backend"
        echo "side=$([ "$backend" = cpu ] && echo cpu || echo gpu) process=$NAME"
        echo "scope=${SCOPE:-routine} shard=${SHARD:-} lanes_override=${LANES:+yes} lane_selection=$SELECTION"
        echo "repeats=$REPEATS cpu_threads=$T"
        echo "host=$(uname -srm) $(hostname 2>/dev/null)"
        echo "date=$(date -u +%FT%TZ)"
        echo "dirty_tracked=${DIRTY:-none}"
        [ "$INSTALLED" = 1 ] && echo "installed_wheels=$INSTALLED_LINE"
        # THE RUN LINES (lane post-record-device-time): device, driver, Mojo, Python, start/end, seconds total and
        # per lane, wheels, box and host, from the record's own run block (identity_break.py run_provenance_lines)
        if [ -f "$json" ]; then
            "${PY[@]}" "$ROOT/tools/identity_break.py" --provenance-lines "$json" 2>/dev/null \
                || echo "run_facts=unreadable (identity_break.py --provenance-lines failed)"
        fi
        [ -f "$json" ] && echo "record_sha256=$(shasum -a 256 "$json" | cut -d' ' -f1)"
    } > "$OUT/$name.provenance.txt"

    line=$("${PY[@]}" tools/identity_columns.py check "$json" 2>&1 | tail -1)
    echo "$line" | cut -c1-600
    adm=NO; case $line in *admissible=yes*) adm=yes;; esac
    # Retain the exact record in MojoLearn as well as the original run directory.
    # Collection does not admit it or regenerate the wheel's reference table.
    if [ -f "$json" ]; then
        "${PY[@]}" tools/retain_run_summary.py identity "$json" || \
            echo "RETENTION FAILED: identity record remains at $json; collect it before teardown" >&2
    fi
    echo "RECORDED $name rc=$rc admissible=$adm json=$json"
    [ "$adm" = yes ]
}

all_adm=0
finish_record "$NAME" "$JSON" "$LABEL" "$BACKEND" || all_adm=1
if [ -n "$CPU_JSON" ]; then
    finish_record "$CPU_NAME" "$CPU_JSON" cpu cpu || all_adm=1
fi
exit $all_adm
