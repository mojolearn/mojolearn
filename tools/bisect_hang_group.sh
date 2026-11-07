#!/bin/bash
# tools/bisect_hang_group.sh -- find the file(s) whose change makes one development group hang in the Mojo compiler.
#
#   bash tools/bisect_hang_group.sh --group <stem> --set <host|identical> --base <commit> --out <dir>
#        [--suspects <file>] [--parallel N] [--stall 420] [--deadline 3600] [--base-archive <tgz>]
#        [--work <dir>] [--keep-trees] [--dry-run]
#   bash tools/bisect_hang_group.sh --export-base <tgz> --base <commit> [--suspects <file>]   # git only, no compile
#   bash tools/bisect_hang_group.sh --group <stem> --set <host|identical> --history <dir> --out <dir> [same options]
#   bash tools/bisect_hang_group.sh --export-history <dir> --base <commit> [--head HEAD] [--overlay-after <commit>]
#        [--history-paths '\.mojo$'] [--overlay-patches <dir> [--overlay-extra <sha,...>]]   # git only, no compile
#   run options for history mode: [--history-range FIRST:LAST] [--overlay-patches <dir>]
#   bash tools/bisect_hang_group.sh --group <stem> --set <host|identical> --root-ddmin <member root .mojo> --out <dir>
#        [--parallel N] [--stall 300] [--deadline 3600] [--work <dir>] [--keep-trees] [--dry-run]
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
#       --evidence <out>/cand/<id> --json <out>/cand/<id>.json --layout development
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
# HISTORY MODE (--history <dir>): one candidate per first-parent commit C from --base (index 0) to --head. The tree
# is this checkout (it must be --head's tree) with every path of `git diff --name-only <base> <head>` that matches
# --history-paths (default: the .mojo sources, the compiler's only inputs) set to C's content (absent at C: deleted),
# then an OVERLAY: each such path changed after --overlay-after (default d5d7463b99, the compile-fix batches and the
# hang fix) takes --head's content where it exists at C, so early commits are not masked by the parse errors the
# fixes removed. The build driver (tools/, bindings/*.sh) stays --head's in every candidate. --export-history writes,
# from git alone: <dir>/history.tsv (index, sha, subject), <dir>/manifest/hNN.tsv (path, blob sha or "-" for absent),
# <dir>/blobs.tgz (every distinct blob once, named by its sha: one archive per candidate would be ~2.5 MB x 62) and
# <dir>/meta.tsv. The run prints the table in first-parent order and the CULPRIT: the first commit whose verdict is HANG
# while its predecessor's is COMPILED or ERROR. Exit 4 when the --head candidate does not HANG (the last one run).
#
# PATCH OVERLAY (export with --overlay-patches <dir>; replaces the whole-file overlay, which broke early trees because
# head's files import modules that do not exist there yet): the manifests carry each commit's own content, and <dir>
# gets one patch per (fix commit, .mojo file): `git diff C^1 C -- <file>` for every --overlay-extra commit, then every
# first-parent commit after --overlay-after up to --head, in that order (<dir>/order.tsv). At build time each patch
# is tried on the candidate tree in order, all-or-nothing per file (patch -p1 --forward --fuzz=0 --dry-run, then for
# real; git apply when patch is missing); a patch that does not apply is skipped (that lane's code is not there yet). What
# applied is recorded in <out>/cand/<id>.patches.tsv and as [patches a/n] in the candidate's label. A patch dir inside
# the history dir is found from meta.tsv; --overlay-patches at run time overrides it. --history-range FIRST:LAST
# (indexes of history.tsv, inclusive) limits the candidates.
#
# ROOT MODE (--root-ddmin <member root>, e.g. bindings/_mojolearn_runtime_classical_host.mojo): every candidate is
# this checkout with only that root rewritten on the box by tools/bisect_root_rewrite.py (see its docstring) from its
# `from bindings./providers. import` lines, numbered 0.. in file order:
#   root-keep  the root re-emitted unchanged (must HANG)      root-none  every such import and its users dropped
#   loo-NN     LEAVE-ONE-OUT: import NN dropped, with every top-level function and builder statement that mentions
#              a name it binds (and, transitively, a dropped function)
#   loi-NN     LEAVE-ONE-IN: only import NN (plus the imports its kept users need, transitively) and its users
# ERROR means the rewrite broke the root: it is recorded with its first error and never counted as COMPILED.
# Each rewritten root is kept as <out>/cand/<id>.root.mojo with <id>.rewrite.txt (kept imports, item counts).
# <out>/root-ddmin.tsv: mode, import_line, verdict, seconds, first_error, id; summary lines "imports whose removal
# clears the hang" (loo COMPILED) and "imports that hang alone" (loi HANG). --stall defaults to 300 here.
# Exit 4 when root-keep does not HANG.
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
GROUP="" SET="" BASE="" SUSPECTS_FILE="" OUT="" PARALLEL="" STALL="" DEADLINE=3600 ROOT=""
BASE_ARCHIVE="" WORK="${BISECT_WORK:-$HOME/bisect-hang-work}" KEEP=0 DRY=0 EXPORT=""
HIST="" EXPORT_HIST="" HEAD_REF=HEAD OVERLAY_AFTER=d5d7463b99 HPATHS='\.mojo$'
PATCHES="" OVERLAY_EXTRA="" HRANGE=""
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
        --history) shift; HIST="${1:-}" ;;
        --export-history) shift; EXPORT_HIST="${1:-}" ;;
        --head) shift; HEAD_REF="${1:-}" ;;
        --overlay-after) shift; OVERLAY_AFTER="${1:-}" ;;
        --history-paths) shift; HPATHS="${1:-}" ;;
        --overlay-patches) shift; PATCHES="${1:-}" ;;
        --overlay-extra) shift; OVERLAY_EXTRA="${1:-}" ;;
        --history-range) shift; HRANGE="${1:-}" ;;
        --root-ddmin) shift; ROOT="${1:-}" ;;
        -h|--help) usage ;;
        *) die "unknown argument: $1 (see --help)" ;;
    esac
    shift
done

[ -n "$BASE" ] || [ -n "$HIST" ] || [ -n "$ROOT" ] || die "--base <commit> is required"
if [ -z "$STALL" ]; then if [ -n "$ROOT" ]; then STALL=300; else STALL=420; fi; fi

# --export-history: per first-parent commit, the content of every changed source path (+ the overlay), from git alone
if [ -n "$EXPORT_HIST" ]; then
    full() { git -C "$R" rev-parse --verify --quiet "$1^{commit}" || die "this checkout has no commit $1"; }
    B=$(full "$BASE") H=$(full "$HEAD_REF") OV=$(full "$OVERLAY_AFTER")
    git -C "$R" merge-base --is-ancestor "$B" "$H" || die "$BASE is not an ancestor of $HEAD_REF"
    git -C "$R" merge-base --is-ancestor "$OV" "$H" || die "$OVERLAY_AFTER is not an ancestor of $HEAD_REF"
    st=$(mktemp -d "${TMPDIR:-/tmp}/bisect-history.XXXXXX") || die "mktemp failed"
    D=$EXPORT_HIST
    rm -rf "$D/manifest" "$D/blobs.tgz" "$D/history.tsv" "$D/meta.tsv"; mkdir -p "$D/manifest" "$st/b"
    git -C "$R" diff --name-only "$B" "$H" | grep -E "$HPATHS" | sort > "$st/paths"
    if [ -n "$PATCHES" ]; then
        : > "$st/overlay"   # no whole-file overlay: the fixes go in as patches
        case "$PATCHES" in /*) ;; *) PATCHES="$(pwd)/$PATCHES" ;; esac
        rm -rf "$PATCHES"; mkdir -p "$PATCHES"
        printf 'seq\tcommit\tpath\tpatch\n' > "$PATCHES/order.tsv"
        { printf '%s\n' "$OVERLAY_EXTRA" | tr ',' '\n' | awk 'NF'
          git -C "$R" rev-list --first-parent --reverse "$OV..$H"; } > "$st/fixes"
        n=0
        while IFS= read -r c; do  # driver-loop: over fix commits
            c=$(full "$c"); cs=$(git -C "$R" rev-parse --short=10 "$c")
            git -C "$R" diff --name-only "$c^1" "$c" | grep -E "$HPATHS" > "$st/fixfiles"
            while IFS= read -r f; do
                n=$((n + 1)); pf=$(printf '%03d-%s-%s.patch' "$n" "$cs" "$(printf '%s' "$f" | tr '/' '_')")
                git -C "$R" diff --full-index "$c^1" "$c" -- "$f" > "$PATCHES/$pf" || die "git diff $cs -- $f failed"
                printf '%s\t%s\t%s\t%s\n' "$n" "$cs" "$f" "$pf" >> "$PATCHES/order.tsv"
            done < "$st/fixfiles"
        done < "$st/fixes"
    else
        git -C "$R" diff --name-only "$OV" "$H" | grep -E "$HPATHS" | sort > "$st/overlay"
    fi
    git -C "$R" ls-tree -r "$H" > "$st/head.tree"
    { echo "$B"; git -C "$R" rev-list --first-parent --reverse "$B..$H"; } > "$st/commits"
    printf 'index\tsha\tsubject\n' > "$D/history.tsv"
    i=0
    while IFS= read -r c; do  # driver-loop: over first-parent commits
        id=$(printf 'h%02d' "$i")
        git -C "$R" ls-tree -r "$c" > "$st/tree"
        # ls-tree lines: "<mode> <type> <sha>\t<path>"
        awk -F'\t' -v OFS='\t' '
            FILENAME == ARGV[1] { want[$1] = 1; order[++n] = $1; next }
            FILENAME == ARGV[2] { ov[$1] = 1; next }
            FILENAME == ARGV[3] { split($1, a, " "); if (a[2] == "blob") hb[$2] = a[3]; next }
            { split($1, a, " "); if (a[2] == "blob" && ($2 in want)) cb[$2] = a[3] }
            END { for (k = 1; k <= n; k++) { p = order[k]; b = (p in cb) ? cb[p] : "-"
                      if (b != "-" && (p in ov)) b = (p in hb) ? hb[p] : "-"
                      print p, b } }' "$st/paths" "$st/overlay" "$st/head.tree" "$st/tree" > "$D/manifest/$id.tsv"
        printf '%s\t%s\t%s\n' "$i" "$(git -C "$R" rev-parse --short=10 "$c")" \
            "$(git -C "$R" log -1 --format=%s "$c" | tr '\t' ' ')" >> "$D/history.tsv"
        i=$((i + 1))
    done < "$st/commits"
    cut -f2 "$D"/manifest/*.tsv | awk '$0 != "-"' | sort -u > "$st/blobs"
    while IFS= read -r b; do git -C "$R" cat-file blob "$b" > "$st/b/$b" || die "git cat-file $b failed"; done < "$st/blobs"
    COPYFILE_DISABLE=1 tar -C "$st/b" -czf "$D/blobs.tgz" . || die "tar failed"
    printf 'base\t%s\nhead\t%s\noverlay_after\t%s\nhistory_paths\t%s\npaths\t%s\noverlay_paths\t%s\nblobs\t%s\n' \
        "$B" "$H" "$OV" "$HPATHS" "$(wc -l < "$st/paths" | tr -d ' ')" "$(wc -l < "$st/overlay" | tr -d ' ')" \
        "$(wc -l < "$st/blobs" | tr -d ' ')" > "$D/meta.tsv"
    if [ -n "$PATCHES" ]; then
        Dabs=$(cd "$D" && pwd)
        case "$PATCHES/" in "$Dabs"/*) rel=${PATCHES#"$Dabs"/} ;; *) rel=$PATCHES ;; esac
        printf 'overlay\tpatches\npatches_dir\t%s\npatches\t%s\noverlay_extra\t%s\n' "$rel" \
            "$(( $(wc -l < "$PATCHES/order.tsv") - 1 ))" "$OVERLAY_EXTRA" >> "$D/meta.tsv"
        echo "patches: $(( $(wc -l < "$PATCHES/order.tsv") - 1 )) per-file patches -> $PATCHES"
    else
        printf 'overlay\tfiles\n' >> "$D/meta.tsv"
    fi
    echo "exported $i candidates, $(wc -l < "$st/paths" | tr -d ' ') paths ($(wc -l < "$st/overlay" | tr -d ' ') overlaid from head), $(wc -l < "$st/blobs" | tr -d ' ') blobs -> $D"
    rm -rf "$st"
    exit 0
fi
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

# candidates: id <TAB> label <TAB> revert set ("-" nothing, "ALL" every suspect, "H:<id>" a history manifest, else one file)
PLAN="$OUT/plan.tsv"
ROOT_IMPORT_RE='^from[[:space:]]+(bindings|providers)\.[A-Za-z0-9_.]+[[:space:]]+import[[:space:]]+'
if [ -n "$ROOT" ]; then
    case "$ROOT" in /*|*..*) die "--root-ddmin is a repo-relative path" ;; esac
    [ -f "$R/$ROOT" ] || die "--root-ddmin $ROOT does not exist in $R"
    [ -z "$HIST" ] || die "--root-ddmin and --history are separate modes"
    # The same lines, in the same order, that tools/bisect_root_rewrite.py numbers (single-line imports only).
    grep -E "$ROOT_IMPORT_RE" "$R/$ROOT" | tr '\t' ' ' > "$OUT/root-imports.txt"
    NIMP=$(wc -l < "$OUT/root-imports.txt" | tr -d ' ')
    [ "$NIMP" -gt 0 ] || die "$ROOT has no from bindings./providers. import lines"
    {
        printf 'root-keep\tkeep - (root unchanged)\tR:keep:-\n'
        printf 'root-none\tnone - (all %s imports dropped)\tR:none:-\n' "$NIMP"
        i=0
        while IFS= read -r line; do printf 'loo-%02d\tloo %02d %s\tR:loo:%s\n' "$i" "$i" "$line" "$i"; i=$((i + 1)); done < "$OUT/root-imports.txt"
        i=0
        while IFS= read -r line; do printf 'loi-%02d\tloi %02d %s\tR:loi:%s\n' "$i" "$i" "$line" "$i"; i=$((i + 1)); done < "$OUT/root-imports.txt"
    } > "$PLAN"
elif [ -n "$HIST" ]; then
    HIST=$(cd "$HIST" 2> /dev/null && pwd) || die "--history directory does not exist"
    { [ -f "$HIST/history.tsv" ] && [ -f "$HIST/blobs.tgz" ]; } || die "$HIST has no history.tsv or blobs.tgz (make it with --export-history)"
    LO=0 HI=999999
    if [ -n "$HRANGE" ]; then
        case "$HRANGE" in *[!0-9:]*|:*|*:|*:*:*) die "--history-range is FIRST:LAST (history.tsv indexes)" ;; *:*) ;; *) die "--history-range is FIRST:LAST" ;; esac
        LO=${HRANGE%:*} HI=${HRANGE#*:}
    fi
    awk -F'\t' -v lo="$LO" -v hi="$HI" 'NR > 1 && $1 >= lo && $1 <= hi { printf "h%02d\t%s %s %s\tH:h%02d\n", $1, $1, $2, $3, $1 }' \
        "$HIST/history.tsv" > "$PLAN"
    [ -s "$PLAN" ] || die "--history-range $HRANGE selects no candidate"
    if [ -z "$PATCHES" ]; then
        rel=$(awk -F'\t' '$1 == "patches_dir" { print $2 }' "$HIST/meta.tsv" 2> /dev/null)
        [ -n "$rel" ] && case "$rel" in /*) PATCHES=$rel ;; *) PATCHES="$HIST/$rel" ;; esac
    fi
    if [ -n "$PATCHES" ]; then
        PATCHES=$(cd "$PATCHES" 2> /dev/null && pwd) || die "--overlay-patches directory does not exist"
        [ -f "$PATCHES/order.tsv" ] || die "$PATCHES has no order.tsv"
    fi
    while IFS=$'\t' read -r id _ _; do [ -f "$HIST/manifest/$id.tsv" ] || die "$HIST/manifest/$id.tsv is missing"; done < "$PLAN"
else {
    printf 'control-head\t(none reverted)\t-\n'
    printf 'control-all\t(all suspects reverted)\tALL\n'
    i=0
    for f in "${FILES[@]}"; do printf 'c%02d\t%s\t%s\n' "$i" "$f" "$f"; i=$((i + 1)); done
} > "$PLAN"; fi
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
    printf 'cd %s && MOJOLEARN_COMPILE_JOBS=1 %s tools/development_build.py build --sets %s --group %s --parallel 1 --stall-seconds %s --group-seconds %s --cache %s --evidence %s --json %s --layout development' \
        "$2" "$PY" "$SET" "$GROUP" "$STALL" "$DEADLINE" "$WORK/cache/$1" "$OUT/cand/$1" "$OUT/cand/$1.json"
}

if [ -n "$ROOT" ]; then
    BASE_SRC="root $ROOT rewritten on the box ($NIMP candidate imports)"
elif [ -n "$HIST" ]; then
    BASE_SRC="history $HIST ($(awk -F'\t' '$1 == "base" || $1 == "head" || $1 == "overlay_after" || $1 == "overlay" { printf "%s=%.10s ", $1, $2 }' "$HIST/meta.tsv" 2> /dev/null))"
    [ -n "$PATCHES" ] && BASE_SRC="$BASE_SRC; patches $PATCHES ($(( $(wc -l < "$PATCHES/order.tsv") - 1 )))"
elif git_has_base; then
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

# base content (or the history blobs), once
BASED="$WORK/base"
BLOBS="$WORK/blobs"
if [ -n "$ROOT" ]; then
    [ -f "$R/tools/bisect_root_rewrite.py" ] || die "tools/bisect_root_rewrite.py is missing"
    "$PY" "$R/tools/bisect_root_rewrite.py" list "$R/$ROOT" > "$OUT/root-imports.tsv" || die "bisect_root_rewrite.py list failed"
    [ "$(wc -l < "$OUT/root-imports.tsv" | tr -d ' ')" = "$NIMP" ] \
        || die "the rewriter sees $(wc -l < "$OUT/root-imports.tsv" | tr -d ' ') imports, the plan $NIMP (multi-line import?)"
elif [ -n "$HIST" ]; then
    if [ ! -f "$BLOBS/.ready" ]; then
        rm -rf "$BLOBS"; mkdir -p "$BLOBS"
        tar -C "$BLOBS" -xzf "$HIST/blobs.tgz" || die "cannot unpack $HIST/blobs.tgz"
        touch "$BLOBS/.ready"
    fi
elif [ ! -f "$BASED/.ready" ]; then
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

apply_patches() {  # $1 id, $2 tree: each per-file patch all-or-nothing, in order; a patch that does not apply is skipped
    local seq c f pf res
    printf 'seq\tcommit\tpath\tresult\n' > "$OUT/cand/$1.patches.tsv"
    while IFS=$'\t' read -r seq c f pf; do  # driver-loop: over overlay patches
        [ "$seq" = seq ] && continue
        res=skipped
        if command -v patch > /dev/null; then
            if (cd "$2" && patch -p1 --forward --fuzz=0 --batch --silent --dry-run -r - < "$PATCHES/$pf") > /dev/null 2>&1; then
                (cd "$2" && patch -p1 --forward --fuzz=0 --batch --silent --no-backup-if-mismatch -r - < "$PATCHES/$pf") > /dev/null 2>&1 \
                    && res=applied || res=failed-after-check
            fi
        elif (cd "$2" && GIT_CEILING_DIRECTORIES=$(dirname "$2") git apply --check -p1 "$PATCHES/$pf") > /dev/null 2>&1; then
            (cd "$2" && GIT_CEILING_DIRECTORIES=$(dirname "$2") git apply -p1 "$PATCHES/$pf") > /dev/null 2>&1 \
                && res=applied || res=failed-after-check
        fi
        printf '%s\t%s\t%s\t%s\n' "$seq" "$c" "$f" "$res" >> "$OUT/cand/$1.patches.tsv"
    done < "$PATCHES/order.tsv"
}

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
        R:*) local cid=$1 m=${2#R:} ix
             ix=${m#*:} m=${m%%:*}
             if [ "$ix" = - ]; then set -- --mode "$m"; else set -- --mode "$m" --index "$ix"; fi
             "$PY" "$R/tools/bisect_root_rewrite.py" rewrite "$R/$ROOT" "$@" --out "$wt/$ROOT" > "$OUT/cand/$cid.rewrite.txt" 2>&1 \
                 || { cat "$OUT/cand/$cid.rewrite.txt"; return 1; }
             cp "$wt/$ROOT" "$OUT/cand/$cid.root.mojo"
             return 0 ;;
        H:*) local p b
             while IFS=$'\t' read -r p b; do
                 if [ "$b" = - ]; then rm -f "$wt/$p"
                 else mkdir -p "$wt/$(dirname "$p")" && cp "$BLOBS/$b" "$wt/$p" || return 1; fi
             done < "$HIST/manifest/${2#H:}.tsv"
             [ -n "$PATCHES" ] && apply_patches "$1" "$wt"
             return 0 ;;
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
        if [ "${revert#R:}" != "$revert" ] && [ -s "$OUT/cand/$id.rewrite.txt" ]; then
            # the rewriter refused this root: deterministic, so a verdict (ERROR), not a retry
            printf '%s\tERROR\t0\trewrite failed: %s\trewrite\t%s\n' "$label" \
                "$(head -n 1 "$OUT/cand/$id.rewrite.txt" | tr '\t' ' ' | cut -c1-300)" "$id" > "$OUT/cand/$id.verdict"
            note "$id ERROR rewrite failed $label"
        else
            printf '%s\ttree copy failed\n' "$(date -u +%H:%M:%SZ)" >> "$OUT/cand/$id.attempt"
        fi
        rm -f "$OUT/cand/$id.running"; return 0
    fi
    if [ -f "$OUT/cand/$id.patches.tsv" ] && [ "${revert#H:}" != "$revert" ]; then
        label="$label [patches $(awk -F'\t' 'NR > 1 && $4 == "applied" { a++ } NR > 1 { n++ } END { printf "%d/%d", a, n }' "$OUT/cand/$id.patches.tsv")]"
    fi
    t0=$(date +%s)
    ( cd "$wt" && MOJOLEARN_COMPILE_JOBS=1 PYTHONPATH="$wt/python" "$PY" tools/development_build.py build \
        --sets "$SET" --group "$GROUP" --parallel 1 --stall-seconds "$STALL" --group-seconds "$DEADLINE" \
        --cache "$WORK/cache/$id" --evidence "$ev" --json "$OUT/cand/$id.json" --layout development ) >> "$OUT/cand/$id.log" 2>&1
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
if [ -n "$ROOT" ]; then
    {
        printf 'mode\timport_line\tverdict\tseconds\tfirst_error\tid\n'
        while IFS=$'\t' read -r id label _; do
            if [ -f "$OUT/cand/$id.verdict" ]; then v=$(cut -f2-4 "$OUT/cand/$id.verdict"); else v=$'NO-VERDICT\t-\t-'; fi
            # label: "<mode> <NN|-> <import line or note>"
            printf '%s\t%s\t%s\t%s\n' "${label%% *}" "${label#* }" "$v" "$id"
        done < "$PLAN"
    } > "$OUT/root-ddmin.tsv"
    cp "$OUT/root-ddmin.tsv" "$OUT/bisect.tsv"
    column -t -s$'\t' "$OUT/root-ddmin.tsv" 2> /dev/null || cat "$OUT/root-ddmin.tsv"
    busy=$(awk -F'\t' 'NR > 1 { s += $2; n++; if ($2 > m) m = $2 } END { if (n) printf "mean %.1f max %.1f of %d cores (%d samples)", s / n, m, $3, n; else print "no samples" }' \
        "$OUT/load.tsv" 2> /dev/null)
    note "done: seconds=$(( $(date +%s) - T0 )) busy ${busy:-no samples}"
    EXIT=0
    keep_v=$(awk -F'\t' '$6 == "root-keep" { print $3 }' "$OUT/root-ddmin.tsv")
    [ "$keep_v" = HANG ] || { note "root-keep is $keep_v, not HANG: the rewrite path does not reproduce the hang"; EXIT=4; }
    note "root-none: $(awk -F'\t' '$6 == "root-none" { print $3 " " $5 }' "$OUT/root-ddmin.tsv")"
    pick() { awk -F'\t' -v m="$1" -v v="$2" '$1 == m && $3 == v { print $2 }' "$OUT/root-ddmin.tsv" | paste -sd ';' -; }
    note "imports whose removal clears the hang (loo COMPILED): $(pick loo COMPILED | sed 's/^$/none/')"
    note "imports whose removal breaks the root (loo ERROR, not counted): $(pick loo ERROR | sed 's/^$/none/')"
    note "imports that hang alone (loi HANG): $(pick loi HANG | sed 's/^$/none/')"
    note "imports that compile alone (loi COMPILED): $(pick loi COMPILED | sed 's/^$/none/')"
    note "loi ERROR (rewrite broke the root, not counted): $(pick loi ERROR | sed 's/^$/none/')"
    missing=$(awk -F'\t' '$3 == "NO-VERDICT" { printf "%s ", $6 }' "$OUT/root-ddmin.tsv")
    if [ -n "$missing" ]; then
        note "no verdict (see cand/<id>.attempt; re-run to retry): $missing"
        [ "$EXIT" = 0 ] && EXIT=5
    fi
    exit "$EXIT"
fi
if [ -n "$HIST" ]; then
    {
        printf 'commit\tverdict\tseconds\tfirst_error\tstatus\tid\n'
        while IFS=$'\t' read -r id label _; do
            if [ -f "$OUT/cand/$id.verdict" ]; then cat "$OUT/cand/$id.verdict"; else printf '%s\tNO-VERDICT\t-\t-\t-\t%s\n' "$label" "$id"; fi
        done < "$PLAN"
    } > "$OUT/bisect.tsv"
    column -t -s$'\t' "$OUT/bisect.tsv" 2> /dev/null || cat "$OUT/bisect.tsv"
    busy=$(awk -F'\t' 'NR > 1 { s += $2; n++; if ($2 > m) m = $2 } END { if (n) printf "mean %.1f max %.1f of %d cores (%d samples)", s / n, m, $3, n; else print "no samples" }' \
        "$OUT/load.tsv" 2> /dev/null)
    note "done: seconds=$(( $(date +%s) - T0 )) busy ${busy:-no samples}"
    EXIT=0
    last_v=$(tail -n 1 "$OUT/bisect.tsv" | cut -f2)
    [ "$last_v" = HANG ] || { note "the last candidate is $last_v, not HANG: the range does not end in the hang"; EXIT=4; }
    [ "$(sed -n 2p "$OUT/bisect.tsv" | cut -f2)" = HANG ] && note "the first candidate already HANGs: the culprit is at or before it"
    note "first candidate: $(sed -n 2p "$OUT/bisect.tsv" | cut -f1,2,4)"
    culprit=$(awk -F'\t' 'NR > 1 { if ($2 == "HANG" && (prev == "COMPILED" || prev == "ERROR")) { print $1 " (after " prevc ": " prev ")"; exit } prev = $2; prevc = $1 }' "$OUT/bisect.tsv")
    note "CULPRIT (first HANG whose predecessor is COMPILED or ERROR): ${culprit:-none found}"
    missing=$(awk -F'\t' '$2 == "NO-VERDICT" { printf "%s ", $6 }' "$OUT/bisect.tsv")
    if [ -n "$missing" ]; then
        note "no verdict (see cand/<id>.attempt; re-run to retry): $missing"
        [ "$EXIT" = 0 ] && EXIT=5
    fi
    exit "$EXIT"
fi
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
