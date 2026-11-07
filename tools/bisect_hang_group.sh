#!/bin/bash
# tools/bisect_hang_group.sh -- find the file(s) whose change makes one development group hang in the Mojo compiler.
#
#   bash tools/bisect_hang_group.sh --group <stem> --set <host|identical> --base <commit> --out <dir>
#        [--suspects <file>] [--parallel N] [--stall 420] [--deadline 3600] [--base-archive <tgz>]
#        [--work <dir>] [--keep-trees] [--dry-run]
#   bash tools/bisect_hang_group.sh --export-base <tgz> --base <commit> [--suspects <file>]   # git only, no compile
#
# On a RunPod CPU pod (cwd = the unpacked repo, the default pixi env installed):
#   bash tools/runpod_cpu_leg.sh --lane bisect-hang --worktree ~/mojolearn-wt/ne-bisect-hang \
#     --cmd 'bash tools/bisect_hang_group.sh --group _mojolearn_dev_runtime_classical_host --set host --base e49ac7df70 --out "$LEG_OUT/bisect"' \
#     --envs default --vcpu 32 --flavors cpu5m,cpu3m --disk 80 --lease 120 --out ~/mojolearn-evidence/bisect-hang --rent
#
# One candidate tree per suspect F: a copy of this checkout with F alone put back to its --base content (a file the
# base does not have is deleted). Two controls: control-head (nothing reverted: must HANG, or the bisect says nothing)
# and control-all (every suspect reverted). Each candidate runs
#   MOJOLEARN_COMPILE_JOBS=1 <default env python> tools/development_build.py build --sets <set> --group <stem>
#       --parallel 1 --stall-seconds <stall> --group-seconds <deadline> --cache <work>/cache/<id>
#       --evidence <out>/cand/<id> --json <out>/cand/<id>.json
# inside its own tree, so every candidate compiles its own sources with its own cache (no reuse between candidates).
# Verdict from the group's <mode>-<stem>.log.execution.json status:
#   HANG      stalled (no CPU progress for --stall s) or timeout (--deadline s reached)
#   COMPILED  passed (development_build.py exits 3 on a --group run: the set is not activated; that is expected)
#   ERROR     failed: the first `error:` line of the group log is recorded
# A file whose revert gives COMPILED or ERROR breaks the hang: it is a culprit (or part of one).
# Other statuses (rss-limit, memory-pressure, interrupted, controller-error, no receipt) are recorded in
# <out>/cand/<id>.attempt and get no verdict, so a re-run retries them. A candidate with <out>/cand/<id>.verdict is
# skipped: the script is safe to re-run on the same --out.
#
# Why a copy and not `git worktree add`: the CPU leg ships tracked files as a tarball, so the pod has no .git.
# Base content comes from git when this checkout has <base> (git show <base>:F), else from --base-archive
# (default tools/bisect_hang_base/<base>.tgz, made on the Mac by --export-base and committed beside this script).
# Each tree shares the main checkout's pixi env through a .pixi symlink. Trees and caches live under --work
# (default $HOME/bisect-hang-work, never under --out: $LEG_OUT is fetched to the Mac) and are deleted after a
# verdict unless --keep-trees.
#
# --parallel default: every core, lowered to 0.8 * MemAvailable / BISECT_GIB_PER_COMPILE (default 3; one compile of
# the group peaks near 2.5 GiB), at most the candidate count. The chosen count and its reason go to <out>/summary.txt.
# <out>/load.tsv: one sample per minute (busy cores from /proc/stat, cores, MemAvailable, candidates running).
#
# Outputs: <out>/plan.tsv, <out>/cand/<id>.{log,json,verdict}, <out>/cand/<id>/ (the build evidence and the group
# log), <out>/bisect.tsv (file, verdict, seconds, first_error, status, id; ERROR, then COMPILED, then HANG),
# <out>/summary.txt. Exit: 0 every candidate has a verdict and control-head hung; 4 control-head did not hang;
# 5 a candidate has no verdict; 2 usage or setup error.
set -u

DEFAULT_SUSPECTS="bindings/admitted_call.mojo
bindings/classical_archive_read.mojo
bindings/classical_native.mojo
bindings/classical_pandas.mojo
bindings/family_native.mojo
bindings/gp_classifier_native.mojo
bindings/host_inference_native.mojo
bindings/logistic_native.mojo
bindings/mcd_native.mojo
bindings/mcls_cluster.mojo
bindings/mcls_common.mojo
bindings/native_scalar_decode.mojo
bindings/neighbors_ann_native.mojo
bindings/neighbors_semi_native.mojo
bindings/pca_fit_native.mojo
bindings/prep_classify_native.mojo
bindings/prep_composition_lifecycle.mojo
bindings/prep_labels_native.mojo
bindings/prep_lda_admission.mojo
bindings/prep_stateless_native.mojo
bindings/trees_bagging_members_native.mojo
bindings/trees_public_meta.mojo
bindings/trees_public_native.mojo
providers/python/estimator_lifecycle_pyapi.mojo
providers/python/pca_input_pyapi.mojo"

die() { echo "bisect_hang_group: $*" >&2; exit 2; }
usage() { sed -n '2,8p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

R=$(cd "$(dirname "$0")/.." && pwd)
GROUP="" SET="" BASE="" SUSPECTS_FILE="" OUT="" PARALLEL="" STALL=420 DEADLINE=3600
BASE_ARCHIVE="" WORK="${BISECT_WORK:-$HOME/bisect-hang-work}" KEEP=0 DRY=0 EXPORT=""
while [ $# -gt 0 ]; do
    case "$1" in
        --group) shift; GROUP="${1:-}" ;;
        --set) shift; SET="${1:-}" ;;
        --base) shift; BASE="${1:-}" ;;
        --suspects) shift; SUSPECTS_FILE="${1:-}" ;;
        --out) shift; OUT="${1:-}" ;;
        --parallel) shift; PARALLEL="${1:-}" ;;
        --stall) shift; STALL="${1:-}" ;;
        --deadline) shift; DEADLINE="${1:-}" ;;
        --base-archive) shift; BASE_ARCHIVE="${1:-}" ;;
        --work) shift; WORK="${1:-}" ;;
        --keep-trees) KEEP=1 ;;
        --dry-run) DRY=1 ;;
        --export-base) shift; EXPORT="${1:-}" ;;
        -h|--help) usage ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
    shift
done

[ -n "$BASE" ] || die "--base <commit> is required"
if [ -n "$SUSPECTS_FILE" ]; then
    [ -f "$SUSPECTS_FILE" ] || die "--suspects $SUSPECTS_FILE does not exist"
    SUSPECTS=$(grep -v -E '^[[:space:]]*(#|$)' "$SUSPECTS_FILE")
else
    SUSPECTS=$DEFAULT_SUSPECTS
fi
[ -n "$SUSPECTS" ] || die "the suspect list is empty"
FILES=()
while IFS= read -r f; do FILES+=("$f"); done <<< "$SUSPECTS"  # no mapfile: --export-base runs on macOS bash 3.2
for f in "${FILES[@]}"; do
    case "$f" in /*|*..*) die "suspect paths are repo-relative: $f" ;; esac
done

git_has_base() { git -C "$R" rev-parse --verify --quiet "$BASE^{commit}" > /dev/null 2>&1; }

# --export-base: the base content of every suspect into one archive (+ .bisect_absent for files the base lacks)
if [ -n "$EXPORT" ]; then
    git_has_base || die "this checkout has no commit $BASE"
    stage=$(mktemp -d "${TMPDIR:-/tmp}/bisect-base.XXXXXX") || die "mktemp failed"
    : > "$stage/.bisect_absent"
    git -C "$R" rev-parse "$BASE^{commit}" > "$stage/.bisect_base"
    for f in "${FILES[@]}"; do
        if git -C "$R" cat-file -e "$BASE:$f" 2> /dev/null; then
            mkdir -p "$stage/$(dirname "$f")"
            git -C "$R" show "$BASE:$f" > "$stage/$f" || die "git show $BASE:$f failed"
        else
            echo "$f" >> "$stage/.bisect_absent"
        fi
    done
    mkdir -p "$(dirname "$EXPORT")"
    COPYFILE_DISABLE=1 tar -C "$stage" -czf "$EXPORT" . || die "tar failed"
    echo "exported $(( ${#FILES[@]} - $(wc -l < "$stage/.bisect_absent") )) files, $(wc -l < "$stage/.bisect_absent" | tr -d ' ') absent at $BASE -> $EXPORT"
    rm -rf "$stage"
    exit 0
fi

[ -n "$GROUP" ] || die "--group <stem> is required"
case "$SET" in host|identical) ;; *) die "--set must be host or identical" ;; esac
[ -n "$OUT" ] || die "--out <dir> is required"
case "$STALL$DEADLINE" in *[!0-9]*) die "--stall and --deadline are whole seconds" ;; esac
[ -n "$PARALLEL" ] && case "$PARALLEL" in *[!0-9]*|0) die "--parallel is a positive integer" ;; esac
OUT=$(mkdir -p "$OUT" && cd "$OUT" && pwd) || die "cannot create --out $OUT"
case "$WORK" in /*) ;; *) WORK="$R/$WORK" ;; esac
case "$WORK/" in "$OUT"/*) die "--work must not be inside --out (the trees would be fetched with the results)" ;; esac

# candidates: id <TAB> label <TAB> revert set ("-" nothing, "ALL" every suspect, else one file)
PLAN="$OUT/plan.tsv"
{
    printf 'control-head\t(none reverted)\t-\n'
    printf 'control-all\t(all suspects reverted)\tALL\n'
    i=0
    for f in "${FILES[@]}"; do printf 'c%02d\t%s\t%s\n' "$i" "$f" "$f"; i=$((i + 1)); done
} > "$PLAN"
NCAND=$(wc -l < "$PLAN" | tr -d ' ')

CORES=$(nproc 2> /dev/null || sysctl -n hw.ncpu)
GIB=${BISECT_GIB_PER_COMPILE:-3}
if [ -z "$PARALLEL" ]; then
    PARALLEL=$CORES REASON="every core ($CORES)"
    if [ -r /proc/meminfo ]; then
        BY_MEM=$(awk -v g="$GIB" '/^MemAvailable:/ { n = int(0.8 * $2 / (g * 1048576)); print (n < 1 ? 1 : n) }' /proc/meminfo)
        [ "$BY_MEM" -lt "$PARALLEL" ] && { PARALLEL=$BY_MEM; REASON="memory: 0.8*MemAvailable / ${GIB} GiB per compile"; }
    fi
    [ "$NCAND" -lt "$PARALLEL" ] && { PARALLEL=$NCAND; REASON="$REASON, capped at $NCAND candidates"; }
else
    REASON="--parallel"
fi

PY="$R/.pixi/envs/default/bin/python3"
command_for() {  # $1 id, $2 tree
    printf 'cd %s && MOJOLEARN_COMPILE_JOBS=1 %s tools/development_build.py build --sets %s --group %s --parallel 1 --stall-seconds %s --group-seconds %s --cache %s --evidence %s --json %s' \
        "$2" "$PY" "$SET" "$GROUP" "$STALL" "$DEADLINE" "$WORK/cache/$1" "$OUT/cand/$1" "$OUT/cand/$1.json"
}

if git_has_base; then
    BASE_SRC="git ($BASE in $R)"
else
    [ -n "$BASE_ARCHIVE" ] || BASE_ARCHIVE="$R/tools/bisect_hang_base/$BASE.tgz"
    BASE_SRC="archive $BASE_ARCHIVE"
fi

if [ "$DRY" = 1 ]; then
    echo "DRY RUN: nothing copied, nothing compiled"
    echo "repo     $R"
    echo "group    $GROUP (set $SET)"
    echo "base     $BASE from $BASE_SRC"
    if [ -n "$BASE_ARCHIVE" ]; then
        if [ -f "$BASE_ARCHIVE" ]; then echo "         archive present"; else echo "         ARCHIVE MISSING"; fi
    fi
    echo "out      $OUT"
    echo "work     $WORK"
    echo "parallel $PARALLEL of $NCAND candidates ($REASON); stall ${STALL}s, deadline ${DEADLINE}s"
    while IFS=$'\t' read -r id label revert; do
        printf 'PLAN %s\trevert=%s\t%s\n' "$id" "$revert" "$label"
        printf '     %s\n' "$(command_for "$id" "$WORK/wt/$id")"
    done < "$PLAN"
    exit 0
fi

[ -x "$PY" ] || die "the default pixi env is not installed ($PY)"
[ -d /root/.pixi/bin ] && PATH="/root/.pixi/bin:$PATH"
command -v pixi > /dev/null || die "pixi is not on PATH (bindings/merged_build.sh runs the compiler through it)"
export PATH
export MOJOLEARN_NUMERIC_MODE=identical
[ "$SET" = identical ] && export MOJOLEARN_GPU_ARCHS="${MOJOLEARN_GPU_ARCHS:-${DEV_GPU_ARCH:-sm_89}}"
mkdir -p "$OUT/cand" "$WORK/wt" "$WORK/cache"
note() { printf '%s\t%s\n' "$(date -u +%H:%M:%SZ)" "$*" | tee -a "$OUT/summary.txt"; }

# base content, once
BASED="$WORK/base"
if [ ! -f "$BASED/.ready" ]; then
    rm -rf "$BASED"; mkdir -p "$BASED"; : > "$BASED/.bisect_absent"
    if git_has_base; then
        for f in "${FILES[@]}"; do
            if git -C "$R" cat-file -e "$BASE:$f" 2> /dev/null; then
                mkdir -p "$BASED/$(dirname "$f")"
                git -C "$R" show "$BASE:$f" > "$BASED/$f" || die "git show $BASE:$f failed"
            else
                echo "$f" >> "$BASED/.bisect_absent"
            fi
        done
    else
        [ -f "$BASE_ARCHIVE" ] || die "no git history for $BASE and no archive $BASE_ARCHIVE (make it with --export-base)"
        tar -C "$BASED" -xzf "$BASE_ARCHIVE" || die "cannot unpack $BASE_ARCHIVE"
        [ -f "$BASED/.bisect_base" ] && case "$(cat "$BASED/.bisect_base")" in
            "$BASE"*) ;; *) die "$BASE_ARCHIVE holds base $(cat "$BASED/.bisect_base"), not $BASE" ;; esac
    fi
    for f in "${FILES[@]}"; do
        [ -f "$BASED/$f" ] || grep -qxF "$f" "$BASED/.bisect_absent" || die "the base content has no entry for $f"
    done
    touch "$BASED/.ready"
fi

# the busy-cores sampler, one sample per minute (same arithmetic as tools/cpu_box_dev_build_and_tests.sh)
sample_load() {
    [ -r /proc/stat ] || return 0
    [ -f "$OUT/load.tsv" ] || printf 'utc\tbusy_cores\tcores\tmem_available_kib\trunning\n' > "$OUT/load.tsv"
    local a1 b1 c1 d1 e1 f1 g1 h1 a2 b2 c2 d2 e2 f2 g2 h2 busy total
    read -r _ a1 b1 c1 d1 e1 f1 g1 h1 _ < /proc/stat
    while :; do
        sleep 60
        read -r _ a2 b2 c2 d2 e2 f2 g2 h2 _ < /proc/stat
        busy=$(( (a2 + b2 + c2 + f2 + g2 + h2) - (a1 + b1 + c1 + f1 + g1 + h1) ))
        total=$(( busy + (d2 + e2) - (d1 + e1) ))
        awk -v b="$busy" -v t="$total" -v n="$CORES" -v m="$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)" \
            -v r="$(find "$OUT/cand" -maxdepth 1 -name '*.running' 2> /dev/null | wc -l | tr -d ' ')" -v u="$(date -u +%H:%M:%SZ)" \
            'BEGIN { printf "%s\t%.1f\t%d\t%s\t%s\n", u, (t > 0 ? n * b / t : 0), n, m, r }' >> "$OUT/load.tsv"
        a1=$a2 b1=$b2 c1=$c2 d1=$d2 e1=$e2 f1=$f2 g1=$g2 h1=$h2
    done
}
sample_load &
SAMPLER=$!
trap 'kill "$SAMPLER" 2> /dev/null' EXIT

make_tree() {  # $1 id, $2 revert set
    local wt="$WORK/wt/$1" f
    rm -rf "$wt"; mkdir -p "$wt"
    # The repo minus its pixi env, its git dir and anything under --work (when --work is inside the repo).
    local ex=(--exclude=./.pixi --exclude=./.git)
    case "$WORK/" in "$R"/*) ex+=("--exclude=./${WORK#"$R"/}") ;; esac
    case "$OUT/" in "$R"/*) ex+=("--exclude=./${OUT#"$R"/}") ;; esac
    tar -C "$R" "${ex[@]}" -cf - . | tar -C "$wt" -xf - || return 1
    ln -s "$R/.pixi" "$wt/.pixi" || return 1
    case "$2" in
        -) return 0 ;;
        ALL) set -- "${FILES[@]}" ;;
        *) set -- "$2" ;;
    esac
    for f in "$@"; do
        if grep -qxF "$f" "$BASED/.bisect_absent"; then
            rm -f "$wt/$f"
        else
            mkdir -p "$wt/$(dirname "$f")" && cp "$BASED/$f" "$wt/$f" || return 1
        fi
    done
}

receipt_status() {  # $1 execution.json -> its status field
    "$PY" -c 'import json, sys; print(json.load(open(sys.argv[1])).get("status", "unknown"))' "$1" 2> /dev/null || echo unreadable
}

run_candidate() {  # $1 id, $2 label, $3 revert set
    local id=$1 label=$2 revert=$3 wt="$WORK/wt/$1" ev="$OUT/cand/$1" t0 rc rec st secs verdict err
    [ -f "$OUT/cand/$id.verdict" ] && return 0
    touch "$OUT/cand/$id.running"
    rm -rf "$ev" "$WORK/cache/$id" "$OUT/cand/$id.json"
    if ! make_tree "$id" "$revert" > "$OUT/cand/$id.log" 2>&1; then
        printf '%s\ttree copy failed\n' "$(date -u +%H:%M:%SZ)" >> "$OUT/cand/$id.attempt"
        rm -f "$OUT/cand/$id.running"; return 0
    fi
    t0=$(date +%s)
    ( cd "$wt" && MOJOLEARN_COMPILE_JOBS=1 PYTHONPATH="$wt/python" "$PY" tools/development_build.py build \
        --sets "$SET" --group "$GROUP" --parallel 1 --stall-seconds "$STALL" --group-seconds "$DEADLINE" \
        --cache "$WORK/cache/$id" --evidence "$ev" --json "$OUT/cand/$id.json" ) >> "$OUT/cand/$id.log" 2>&1
    rc=$?
    secs=$(( $(date +%s) - t0 ))
    rec=$(find "$ev" -maxdepth 1 -name "*-$GROUP.log.execution.json" 2> /dev/null | head -n 1)
    [ -n "$rec" ] || rec=$(find "$ev" -maxdepth 1 -name '*.log.execution.json' ! -name 'generate-unicode*' 2> /dev/null | head -n 1)
    err="-"
    if [ -n "$rec" ]; then
        st=$(receipt_status "$rec")
        case "$st" in
            stalled|timeout) verdict=HANG ;;
            passed) verdict=COMPILED ;;
            failed) verdict=ERROR
                    err=$(grep -m 1 -E 'error:' "${rec%.execution.json}" 2> /dev/null | cut -c1-300)
                    [ -n "$err" ] || err=$(grep -m 1 -i 'error' "${rec%.execution.json}" 2> /dev/null | cut -c1-300) ;;
            *) verdict="" ;;
        esac
    else
        st="no-receipt rc=$rc"
        verdict=""
        # A failure before the compiler started (generator, planning) is an ERROR of the tree, not a retry case.
        if [ "$rc" != 0 ] && [ "$rc" != 130 ] && grep -q '^development_build:' "$OUT/cand/$id.log"; then
            verdict=ERROR
            err=$(grep -m 1 '^development_build:' "$OUT/cand/$id.log" | cut -c1-300)
        fi
    fi
    err=$(printf '%s' "${err:--}" | tr '\t' ' ')
    if [ -n "$verdict" ]; then
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$label" "$verdict" "$secs" "$err" "$st" "$id" > "$OUT/cand/$id.verdict"
        [ "$KEEP" = 1 ] || rm -rf "$wt" "$WORK/cache/$id"
    else
        printf '%s\tstatus=%s rc=%s seconds=%s (no verdict; retried on a re-run)\n' "$(date -u +%H:%M:%SZ)" "$st" "$rc" "$secs" \
            >> "$OUT/cand/$id.attempt"
    fi
    note "$id ${verdict:-NO-VERDICT} status=$st seconds=$secs $label"
    rm -f "$OUT/cand/$id.running"
}

rm -f "$OUT"/cand/*.running
running() { jobs -rp | grep -c -v -x "$SAMPLER"; }  # candidate jobs (the sampler is not one)
note "start: group $GROUP set $SET base $BASE from $BASE_SRC; $NCAND candidates, $PARALLEL at once ($REASON); stall ${STALL}s deadline ${DEADLINE}s"
T0=$(date +%s)
while IFS=$'\t' read -r id label revert; do  # driver-loop: over candidate trees
    while [ "$(running)" -ge "$PARALLEL" ]; do wait -n; done
    run_candidate "$id" "$label" "$revert" &
done < "$PLAN"
while [ "$(running)" -gt 0 ]; do wait -n; done

# results
{
    printf 'file\tverdict\tseconds\tfirst_error\tstatus\tid\n'
    cat "$OUT"/cand/*.verdict 2> /dev/null | awk -F'\t' 'BEGIN { OFS = "\t" }
        { r = ($2 == "ERROR" ? 0 : ($2 == "COMPILED" ? 1 : 2)); print r, $0 }' | sort -t$'\t' -k1,1n -k2,2 | cut -f2-
} > "$OUT/bisect.tsv"
column -t -s$'\t' "$OUT/bisect.tsv" 2> /dev/null || cat "$OUT/bisect.tsv"

busy=$(awk -F'\t' 'NR > 1 { s += $2; n++; if ($2 > m) m = $2 } END { if (n) printf "mean %.1f max %.1f of %d cores (%d samples)", s / n, m, $3, n; else print "no samples" }' \
    "$OUT/load.tsv" 2> /dev/null)
note "done: seconds=$(( $(date +%s) - T0 )) busy ${busy:-no samples}"
EXIT=0
head_v=$(cut -f2 "$OUT/cand/control-head.verdict" 2> /dev/null)
if [ "$head_v" != HANG ]; then
    note "control-head is ${head_v:-without a verdict}, not HANG: this tree does not reproduce the hang, the bisect is inconclusive"
    EXIT=4
fi
note "control-all: $(cut -f2,4 "$OUT/cand/control-all.verdict" 2> /dev/null || echo 'no verdict')"
culprits=$(awk -F'\t' '$6 ~ /^c[0-9]/ && ($2 == "ERROR" || $2 == "COMPILED") { print $1 " (" $2 ")" }' "$OUT/bisect.tsv" | paste -sd ";" -)
note "culprits (revert breaks the hang): ${culprits:-none}"
missing=$(while IFS=$'\t' read -r id _ _; do [ -f "$OUT/cand/$id.verdict" ] || printf '%s ' "$id"; done < "$PLAN")
if [ -n "$missing" ]; then
    note "no verdict (see cand/<id>.attempt; re-run to retry): $missing"
    [ "$EXIT" = 0 ] && EXIT=5
fi
exit "$EXIT"
