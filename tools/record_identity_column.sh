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
#   (default)               EVERY LANE (Andrew 2026-11-05, decided 2026-11-09: reuse of old
#                           runs is opt-in): with no LANES, SHARD, ONLY_CHANGED or REUSE, every
#                           lane of the scope is recorded; nothing is reused
#   REUSE=1                 opt into reuse: with no LANES, SHARD or ONLY_CHANGED, only the lanes
#                           whose source closure (tools/lane_select.py, this column's backend)
#                           changed since the newest good evidence of THIS device class are
#                           recorded: the admitted table's rows and every admissible record under
#                           bench/results/identity_break (retained/ included), <outdir> and
#                           MOJOLEARN_IDENTITY_REUSE_RECORDS=dir1:dir2. Prints REUSED n / RECORD m,
#                           and NOTHING-CHANGED (exit 0) when no lane is owed; decisions in
#                           <outdir>/<label>.reuse.json. A depth-1 box tree is enough: the reuse
#                           step fetches HEAD's history back to the oldest reuse commit from the
#                           tree's own remote (MOJOLEARN_IDENTITY_REUSE_REMOTE, default origin)
#                           and names the release tags it holds. (FULL=1, the old way to record
#                           every lane, is retired with the reuse default and refused.)
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
#   ALLOW_DIRTY=1           diagnostic recording from modified tracked files;
#                           guarded release launches refuse this override
#   MOJOLEARN_IDENTITY_LAUNCH  optional guarded release launch from
#                           release_identity_run.py; validated when supplied
#   RECORD_FROM_INSTALLED=1 THE RECORD IS THE GATE (Andrew 2026-11-05: one execution of the identity
#                           lanes per device class, on the built release wheels). Default off. The lanes
#                           run against the mojolearn INSTALLED in PYTHON's environment (the exact release
#                           wheel bytes), never against $ROOT/python: PYTHONPATH is not extended, the
#                           preflight refuses a package under $ROOT or outside site-packages, and the
#                           installed identity_columns/COMMIT must equal HEAD (the harness in tools/ is the
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
#   SELECT_ONLY=1           with REUSE=1, the reuse selection only: prints LANES-SELECTED <csv> (or
#                           NOTHING-CHANGED) and records nothing
#   LANE_SELECTION=auto-reuse-chunk  with LANES: this record is one chunk of a SELECT_ONLY selection
#   LANE_SELECTION=full-chunk  with LANES: one chunk of a full column (every lane; release.py --identity-on-wheels
#                           default, Andrew 2026-11-05: run everything once per release, no selector)
#                           (tools/identity_wheel_record.py runs the chunks in parallel)
#
# Start ledger-managed release columns through release_identity_run.py after
# reconciling retained evidence. Legacy per-run controllers remain supported.
# The ledger retains finished work and refuses duplicate active runs. A refused resume is retained
# for diagnosis; it never silently starts every completed fit again.
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
SELECTION=explicit
# EVERY LANE BY DEFAULT; REUSE IS OPT-IN (Andrew 2026-11-05, decided 2026-11-09,
# AGENTS.md "Reuse of old runs is opt-in"). With no LANES, SHARD or ONLY_CHANGED
# the column records every lane of the scope. REUSE=1 records only the lanes
# whose source closure changed since the newest good evidence THIS device class
# already holds and reuses the rest. FULL=1 (the old opt-out of the reuse
# default) is retired with it.
[ -z "${FULL:-}" ] || die "FULL=$FULL is retired: every lane is recorded by default; REUSE=1 opts into reuse"
case "${REUSE:-0}" in 0|1) ;; *) die "REUSE=$REUSE: 1 reuses unchanged lanes, 0 (default) records every lane" ;; esac
if [ "${REUSE:-0}" = 0 ]; then
    if [ -z "${ONLY_CHANGED:-}" ] && [ -z "${LANES:-}" ] && [ -z "${SHARD:-}" ]; then
        SELECTION=full
        echo "FULL $LABEL: every ${SCOPE:-routine} lane is recorded; no lane is reused (REUSE=1 opts into reuse)"
    fi
else
    [ -z "${ONLY_CHANGED:-}" ] && [ -z "${LANES:-}" ] && [ -z "${SHARD:-}" ] \
        || die "REUSE=1 excludes LANES, SHARD and ONLY_CHANGED"
    SELECTION=auto-reuse
    CLASS=$("${PY[@]}" -c 'import sys; sys.path.insert(0, "tools"); import identity_columns as c; print(c.label_class(sys.argv[1]))' "$LABEL") \
        || die "no device class for label $LABEL"
    reuse_args=(--class "$CLASS" --records "$ROOT/bench/results/identity_break" --records "$OUT"
                --summary-json "$OUT/$LABEL.reuse.json")
    # Admissible records that are not committed yet (a previous box run, a
    # retained archive): MOJOLEARN_IDENTITY_REUSE_RECORDS=dir1:dir2.
    IFS=: read -r -a extra_dirs <<< "${MOJOLEARN_IDENTITY_REUSE_RECORDS:-}"
    for d in ${extra_dirs[@]+"${extra_dirs[@]}"}; do
        [ -z "$d" ] && continue
        [ -d "$d" ] || die "MOJOLEARN_IDENTITY_REUSE_RECORDS: no such directory $d"
        reuse_args+=(--records "$d")
    done
    # The column's backend (lane_select's derived rules): a host-binding-only
    # change cannot move an NVIDIA or AMD column; the other platform's packaging
    # tree cannot reach binaries built here (Apple and a Mac cpu column build on
    # macOS, the NVIDIA and AMD columns from the Linux sets).
    case "$LABEL" in
        nvidia-*) reuse_args+=(--backend cuda) ;;
        amd-*) reuse_args+=(--backend hip) ;;
        apple-*) reuse_args+=(--backend metal) ;;
        cpu) reuse_args+=(--backend cpu) ;;
    esac
    AUTO_LANES=$("${PY[@]}" tools/identity_columns.py changed-lanes --scope "${SCOPE:-routine}" "${reuse_args[@]}" \
            2> "$OUT/$LABEL.changed-lanes.txt" | tail -n 1) \
        || die "could not decide which lanes to reuse (see $OUT/$LABEL.changed-lanes.txt)"
    summary=$(grep -m 1 'REUSE-SUMMARY' "$OUT/$LABEL.changed-lanes.txt") \
        || die "no reuse summary (see $OUT/$LABEL.changed-lanes.txt)"
    reused=$(sed -n 's/.* REUSED \([0-9]*\) .*/\1/p' <<< "$summary")
    record=$(sed -n 's/.* RECORD \([0-9]*\) .*/\1/p' <<< "$summary")
    bad=$(sed -n 's/.* KNOWN-BAD \([0-9]*\) .*/\1/p' <<< "$summary")
    # n/a: lanes the harness declares have no route on this class (GPU_ONLY_LANES on cpu), not known-bad.
    na=$(sed -n 's|.* N/A \([0-9]*\).*|\1|p' <<< "$summary")
    echo "REUSED $reused / RECORD $record $LABEL (class $CLASS, known-bad $bad, n/a ${na:-0}; decisions $OUT/$LABEL.reuse.json; without REUSE=1 every lane is recorded)"
    # RELEASE FLOOR = THE LEDGER'S RULE (lane ledger-floor-receipts): changed-lanes measures a row
    # from the release floor only with an admitted receipt at the floor (identity_columns.py
    # floor_receipts_for -> release_identity_reference.floor_receipts); a tag alone is no floor.
    grep -m 1 '^# release floor' "$OUT/$LABEL.changed-lanes.txt" | cut -c1-240 || true
    # Reused rows from records that are not committed yet must reach admission.
    "${PY[@]}" -c '
import json, sys
doc = json.load(open(sys.argv[1]))
src = sorted({r["source"] for r in doc["reused"].values() if r.get("source") not in (None, "table") and not r.get("committed")})
if src:
    print("ADMIT-WITH %d uncommitted reuse source(s): pass their directories to tools/admit_identity_columns.sh: %s"
          % (len(src), " ".join(src[:8])))
' "$OUT/$LABEL.reuse.json"
    if [ -z "$AUTO_LANES" ]; then
        echo "NOTHING-CHANGED $LABEL: no lane's closure moved since this class's newest good evidence; every row is reused"
        exit 0
    fi
    ALL_LANES=$("${PY[@]}" tools/identity_columns.py lanes --scope "${SCOPE:-routine}") \
        || die "could not list the ${SCOPE:-routine} lanes"
    # Nothing to reuse: an ordinary full column under its own label.
    [ "$AUTO_LANES" = "$ALL_LANES" ] || LANES=$AUTO_LANES
    # SELECT_ONLY=1 (tools/identity_wheel_record.py): print the owed lanes and stop, so one selection by THIS
    # rule feeds every parallel worker (each then records its chunk with LANES=... LANE_SELECTION=auto-reuse-chunk)
    if [ "${SELECT_ONLY:-0}" = 1 ]; then echo "LANES-SELECTED $AUTO_LANES"; exit 0; fi
fi
[ "${SELECT_ONLY:-0}" = 0 ] || [ "$SELECTION" = auto-reuse ] || die "SELECT_ONLY=1 needs REUSE=1 (the reuse selection)"
if [ -n "${LANE_SELECTION:-}" ]; then
    case "$LANE_SELECTION" in auto-reuse-chunk|full-chunk) ;; *) die "LANE_SELECTION is auto-reuse-chunk or full-chunk";; esac
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

{
    echo "commit=$COMMIT"
    echo "label=$LABEL"
    echo "backend=$BACKEND"
    echo "scope=${SCOPE:-routine} shard=${SHARD:-} lanes_override=${LANES:+yes} lane_selection=$SELECTION"
    echo "repeats=$REPEATS cpu_threads=$T"
    echo "host=$(uname -srm) $(hostname 2>/dev/null)"
    echo "date=$(date -u +%FT%TZ)"
    echo "dirty_tracked=${DIRTY:-none}"
    [ "$INSTALLED" = 1 ] && echo "installed_wheels=$INSTALLED_LINE"
    # THE RUN LINES (lane post-record-device-time): device, driver, Mojo, Python, start/end, seconds total and
    # per lane, wheels, box and host, from the record's own run block (identity_break.py run_provenance_lines)
    if [ -f "$JSON" ]; then
        "${PY[@]}" "$ROOT/tools/identity_break.py" --provenance-lines "$JSON" 2>/dev/null \
            || echo "run_facts=unreadable (identity_break.py --provenance-lines failed)"
    fi
    [ -f "$JSON" ] && echo "record_sha256=$(shasum -a 256 "$JSON" | cut -d' ' -f1)"
} > "$OUT/$NAME.provenance.txt"

line=$("${PY[@]}" tools/identity_columns.py check "$JSON" 2>&1 | tail -1)
echo "$line" | cut -c1-600
adm=NO; case $line in *admissible=yes*) adm=yes;; esac
# Retain the exact record in MojoLearn as well as the original run directory.
# Collection does not admit it or regenerate the wheel's reference table.
if [ -f "$JSON" ]; then
    "${PY[@]}" tools/retain_run_summary.py identity "$JSON" || \
        echo "RETENTION FAILED: identity record remains at $JSON; collect it before teardown" >&2
fi
echo "RECORDED $NAME rc=$rc admissible=$adm json=$JSON"
[ "$adm" = yes ] || exit 1
exit 0
