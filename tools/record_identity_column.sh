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
#                   cpu                  -> MOJOLEARN_VENDOR=cpu, --require-backend cpu;
#                                           the label is derived by the harness
#                                           (cpu-<cpu model slug>), never typed
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
#   ONLY_CHANGED=table|<sha> INCREMENTAL RE-RECORD: only the lanes whose source
#                           closure (tools/lane_select.py) changed between the
#                           commit each lane's admitted rows rest on (`table`)
#                           or one named <sha>, and HEAD; re-record
#                           only what changed; every other row is kept (the
#                           table builder takes the newest commit per cell, so
#                           the unchanged lanes keep their admitted rows).
#                           Prints NOTHING-CHANGED and exits 0 when no lane moved
#   CPU_THREADS=3           every CPU thread pool (MOJOLEARN_CPU_THREADS, OMP...)
#   PYTHON=...              the interpreter (default: `pixi run -e default python`
#                           when pixi and pixi.toml are present, else python3)
#   ALLOW_DIRTY=1           record from a tree with modified tracked files
#                           (the record then names a commit that is not its
#                           source; admission tooling will not know)
#
# Rerunning the same command in the SAME tree resumes (identity_break --resume);
# a refused resume (new tree path, rebuilt bindings) moves the old file aside
# and starts that record again. A finished record is left untouched.
#
# Output, per record: <outdir>/<file>.identical.json, its .log, an IDCOLUMN
# line from tools/identity_columns.py check (admissible=yes|NO), and
# <outdir>/<file>.provenance.txt. The last line of stdout is
#   RECORDED <file> rc=<harness rc> admissible=<yes|NO>
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
mkdir -p "$OUT" || die "cannot create $OUT"

# --- the interpreter -------------------------------------------------------
if [ -n "${PYTHON:-}" ]; then
    read -r -a PY <<< "$PYTHON"
elif command -v pixi > /dev/null 2>&1 && [ -f pixi.toml ]; then
    PY=(pixi run -e default python)
else
    PY=(python3)
fi

# --- the environment the record claims -------------------------------------
for v in MOJOLEARN_IDENTITY_N MOJOLEARN_IDENTITY_WIDE MOJOLEARN_PAR_DEVICES; do
    [ -z "${!v:-}" ] || die "$v is set; a reference record runs the default fixture on one device"
done
for v in $(env | sed -n 's/^\(MOJOLEARN_[A-Z0-9_]*SABOTAGE[A-Z0-9_]*\)=.*/\1/p'); do
    [ -z "${!v:-}" ] || [ "${!v}" = 0 ] || die "$v is set; a sabotage run is never a record"
done
export MOJOLEARN_NUMERIC_MODE=identical PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/python${PYTHONPATH:+:$PYTHONPATH}"
T=${CPU_THREADS:-3}
for v in MOJOLEARN_CPU_THREADS OMP_NUM_THREADS OMP_THREAD_LIMIT OPENBLAS_NUM_THREADS MKL_NUM_THREADS \
         VECLIB_MAXIMUM_THREADS NUMEXPR_NUM_THREADS NUMEXPR_MAX_THREADS BLIS_NUM_THREADS; do
    export "$v=$T"
done
export OMP_MAX_ACTIVE_LEVELS=1 OMP_DYNAMIC=FALSE
if [ "$BACKEND" = cpu ]; then export MOJOLEARN_VENDOR=cpu; fi
# THE PTX COLUMN: the PTX set by name, like any architecture directory; a native
# column must not inherit a forced architecture from the caller.
if [ "$PTX" = 1 ]; then export MOJOLEARN_GPU_ARCH=sm_80; elif [ "$BACKEND" = cuda ]; then unset MOJOLEARN_GPU_ARCH; fi
# identity_break refuses a Metal matrix of more than one lane unless it is an
# intentional full run; a reference column is exactly that.
if [ "$BACKEND" = metal ]; then export MOJOLEARN_APPLE_FULL_DIAGNOSTIC=1; fi

COMMIT=$(git rev-parse HEAD 2>/dev/null) || die "not a git checkout; the record needs a commit witness"
# A build may touch a lock file or a generated table; a changed SOURCE file
# means the record would name a commit that is not what ran.
DIRTY=$(git status --porcelain --untracked-files=no 2>/dev/null | awk '{print $NF}' | tr '\n' ' ')
SRC_DIRTY=$(tr ' ' '\n' <<< "$DIRTY" | grep -E '^(python/|tools/|bindings/|.*\.mojo$)' \
            | grep -v -x 'tokenizer/impl/unicode_table_generated.mojo' | tr '\n' ' ')
if [ -n "${SRC_DIRTY// /}" ] && [ "${ALLOW_DIRTY:-0}" != 1 ]; then
    die "source files differ from $COMMIT: $SRC_DIRTY(ALLOW_DIRTY=1 overrides)"
fi

# --- preflight: the loaded backend and tier, before any fit ----------------
got=$("${PY[@]}" -c 'import mojolearn as m, mojolearn._backend as b; print(m.vendor(), b.numeric_mode())' 2>&1 | tail -1)
[ "$got" = "$BACKEND identical" ] || die "preflight: expected '$BACKEND identical', loaded '$got' (are the bindings built in $ROOT?)"
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
if [ -n "${ONLY_CHANGED:-}" ]; then
    # INCREMENTAL RE-RECORD: re-record only what changed; every other row is kept.
    # The lanes whose source closure (tools/lane_select.py, the release lane
    # selection's map) changed since the commit each lane's admitted rows rest
    # on (ONLY_CHANGED=table), or since one named commit (ONLY_CHANGED=<sha>).
    # admit_identity_columns.sh then rebuilds the table from every committed
    # column, newest commit per cell, so the rows of every other lane are kept.
    [ -z "${LANES:-}" ] && [ -z "${SHARD:-}" ] || die "ONLY_CHANGED excludes LANES and SHARD"
    since_arg=()
    [ "$ONLY_CHANGED" = table ] || since_arg=(--since "$ONLY_CHANGED")
    LANES=$("${PY[@]}" tools/identity_columns.py changed-lanes --scope "${SCOPE:-routine}" ${since_arg[@]+"${since_arg[@]}"} \
            2> "$OUT/$LABEL.changed-lanes.txt" | tail -n 1) || die "could not list the changed lanes (see $OUT/$LABEL.changed-lanes.txt)"
    grep -m 1 'CHANGED-LANES' "$OUT/$LABEL.changed-lanes.txt"
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
done_already() {
    [ -f "$JSON" ] && "${PY[@]}" -c "import json,sys; j=json.load(open(sys.argv[1])); sys.exit(0 if j.get('complete') is True and j.get('commit')==sys.argv[2] else 1)" "$JSON" "$COMMIT" 2>/dev/null
}
if done_already; then
    echo "# $JSON is already complete at ${COMMIT:0:12}; left untouched"
else
    if [ -f "$JSON" ] && ! "${PY[@]}" -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('commit')==sys.argv[2] else 1)" "$JSON" "$COMMIT" 2>/dev/null; then
        mv "$JSON" "$JSON.other-commit-$(date -u +%Y%m%dT%H%M%SZ)"
    fi
    args=(tools/identity_break.py --json "$JSON" --lanes "$LANE_LIST" --repeats "$REPEATS"
          --require-backend "$BACKEND")
    if [ "$BACKEND" = cpu ]; then args+=(--require-cpu); else args+=(--vendor "$LABEL"); fi
    if [ -n "${FIXTURES:-}" ]; then die "FIXTURES is not supported: a record covers every fixture"; fi
    {
        echo "# $(date -u +%FT%TZ) record $NAME commit=$COMMIT backend=$BACKEND threads=$T lanes=$(tr ',' '\n' <<< "$LANE_LIST" | wc -l | tr -d ' ')"
    } >> "$LOG"
    if [ -f "$JSON" ]; then
        "${PY[@]}" -u "${args[@]}" --resume >> "$LOG" 2>&1
        rc=$?
        if grep -q "REFUSING TO RESUME" <(tail -n 5 "$LOG"); then
            mv "$JSON" "$JSON.unresumable-$(date -u +%Y%m%dT%H%M%SZ)"
            echo "# resume refused (new tree or rebuilt bindings); recording $NAME again" >> "$LOG"
            "${PY[@]}" -u "${args[@]}" >> "$LOG" 2>&1
            rc=$?
        fi
    else
        "${PY[@]}" -u "${args[@]}" >> "$LOG" 2>&1
        rc=$?
    fi
    echo "# harness rc=$rc" >> "$LOG"
fi
rc=${rc:-0}

{
    echo "commit=$COMMIT"
    echo "label=$LABEL"
    echo "backend=$BACKEND"
    echo "scope=${SCOPE:-routine} shard=${SHARD:-} lanes_override=${LANES:+yes}"
    echo "repeats=$REPEATS cpu_threads=$T"
    echo "host=$(uname -srm) $(hostname 2>/dev/null)"
    echo "date=$(date -u +%FT%TZ)"
    echo "dirty_tracked=${DIRTY:-none}"
    [ -f "$JSON" ] && echo "record_sha256=$(shasum -a 256 "$JSON" | cut -d' ' -f1)"
} > "$OUT/$NAME.provenance.txt"

line=$("${PY[@]}" tools/identity_columns.py check "$JSON" 2>&1 | tail -1)
echo "$line" | cut -c1-600
adm=NO; case $line in *admissible=yes*) adm=yes;; esac
echo "RECORDED $NAME rc=$rc admissible=$adm json=$JSON"
[ "$adm" = yes ] || exit 1
exit 0
