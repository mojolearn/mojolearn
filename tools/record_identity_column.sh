#!/bin/bash
# record_identity_column.sh <vendor-label> <outdir> [--models]
#
# ONE ADMISSIBLE COLUMN RECORD for the verifier's reference table
# (python/mojolearn/verify_reference/table.json), written the way
# _verify_reference.admit() and docs/VERIFY.md "Regenerating the reference
# table" require: tools/identity_break.py --json in the identical tier, a
# commit witness, the default fixture size, one device (par_devices 0), no
# sabotage, every part collected (never --partial-column), every fixture,
# --repeats 2 (two identical samples per part), input and held-out witnesses
# and the batch/property protocols recorded by the harness itself.
#
# Runs in a BUILT checkout of a committed tree (every binding built for this
# box; the host bindings too for the cpu column). Runs nothing else. The
# orchestrator runs it on the box; lanes never do.
#
#   <vendor-label>  nvidia-<gpu>-<arch>  -> --require-backend cuda
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
#   REPEATS=2               fits per cell (default 2; never below 2 for a table)
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
mkdir -p "$OUT" || die "cannot create $OUT"

case $LABEL in
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
REPEATS=${REPEATS:-2}
[ "$REPEATS" -ge 2 ] 2>/dev/null || die "REPEATS=$REPEATS: a reference record needs two samples per part"

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
