#!/bin/bash
# tools/box_persist_run.sh -- RUNS ON A RENTED BOX: a job whose results reach R2 from the box itself, so they
# survive the Mac disconnecting, the box's dead-man firing, or both (Andrew, 2026-10-26).
#
#   box_persist_run.sh <persist.json> <results-dir> -- <command...>
#       runs the command (stdout+stderr also tee'd to <results-dir>/command.log), and every every_sec seconds
#       tars <results-dir> and PUTs it to the next of the job's slots (a ring: 00.tgz, 01.tgz, ...) with a
#       sha256 sidecar and a manifest.json. On ANY exit (normal, TERM, INT, the EXIT trap) it uploads
#       final.tgz. HUP and PIPE are ignored, so an ssh session that dies under `dev_pod.sh run` or
#       `do_amd_steward.sh run` leaves the job running and persisting. Exit status: the command's.
#   box_persist_run.sh --flush-all [seconds]
#       THE DEAD-MAN HOOK: uploads final.tgz for every active job on this box, bounded by <seconds> (default
#       120). The self-destruct paths (tools/runpod_guard.sh watchdog, the gpu_queue_box.sh lease delete,
#       tools/do_amd_steward.sh's mojolearn-selfkill unit) call it before they delete the box.
#   box_persist_run.sh --snapshot <persist.json> <results-dir> [periodic|final|flush]   (internal)
#
# <persist.json> is minted on the Mac by tools/box_persist.py (`mint`/`prepare`) and reaches the box on ssh
# stdin as a 0600 file. It holds presigned PUT/GET URLs for this job's keys only, which expire (at most 7
# days); no credential ever reaches the box. It is a flat JSON object, one "key": "value" pair per line, so
# this script reads it with sed: needs bash, curl, tar, gzip, sha256sum or shasum. Nothing else.
#
# Every upload writes one status line to <results-dir>/persist.log:
#   UPLOADED name=00.tgz seq=3 label=periodic bytes=.. sha256=..   FAILED name=.. http=..   REJECTED (403: expired
#   or wrong key)   EXPIRED (URLs past their expiry: nothing more is sent)   READBACK-OK|READBACK-MISMATCH
# Excluded from every tarball: venv, venv-*, .venv, .pixi, *.whl, .cache, __pycache__, .mojo_cache,
# node_modules, *.persist.json, persist.json, plus the json's "excludes". A tarball over max_bytes is rebuilt
# without its largest files, and PERSIST_TRUNCATED.txt in the results dir names what was left out.
#
# Box state: <root>/active/<tag>.env (one per running job; what --flush-all walks), <root>/state/<tag>/ (upload
# counter, entries, lock). <root> is $BOX_PERSIST_ROOT, else the parent of this script's bin/ directory, else
# /root/box-persist. Tests: tools/test_box_persist.py.
set -u
SELF="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
case "$(basename "$(cd "$(dirname "$0")" && pwd)")" in
    bin) _root_default=$(cd "$(dirname "$0")/.." && pwd) ;;
    *) _root_default=/root/box-persist ;;
esac
ROOT=${BOX_PERSIST_ROOT:-$_root_default}
POLL=${PERSIST_POLL_SEC:-2}
GRACE=${PERSIST_TERM_GRACE_SEC:-10}
PUT_MAX=${PERSIST_PUT_MAX_SEC:-1800}
FINAL_PUT_MAX=${PERSIST_FINAL_PUT_MAX_SEC:-300}
DEFAULT_EXCLUDES='venv venv-* .venv .pixi *.whl .cache __pycache__ .mojo_cache node_modules *.persist.json persist.json'

now_utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
sum256() { if command -v sha256sum > /dev/null 2>&1; then sha256sum "$1" | cut -c1-64; else shasum -a 256 "$1" | cut -c1-64; fi; }
fsize() { wc -c < "$1" | tr -d ' '; }
plog() { echo "$(now_utc) $*" >> "$RES/persist.log" 2> /dev/null; }
say() { echo "box_persist: $*" >&2 2> /dev/null; }
die() { echo "box_persist: $*" >&2; exit 2; }

# the flat json: "key": "value" on its own line (tools/box_persist.py writes every value as a string)
jget() {
    local k=${1//./\\.}
    sed -n "s/^[[:space:]]*\"$k\":[[:space:]]*\"\([^\"]*\)\",\{0,1\}[[:space:]]*\$/\1/p" "$JSON" | head -1
}

load_job() {  # JSON RES set
    [ -r "$JSON" ] || die "no persist json $JSON"
    TAG=$(jget tag); PREFIX=$(jget prefix); SLOTS=$(jget slots); EVERY=$(jget every_sec)
    EXP=$(jget expires_epoch); EXPU=$(jget expires_utc); MAXB=$(jget max_bytes); READBACK=$(jget readback)
    EXTRA_EXCL=$(jget excludes)
    case "$TAG" in ''|*[!A-Za-z0-9._-]*) die "persist json $JSON has no valid tag" ;; esac
    case "$SLOTS" in ''|*[!0-9]*) SLOTS=1 ;; esac; [ "$SLOTS" -ge 1 ] || SLOTS=1
    case "$EVERY" in ''|*[!0-9]*) EVERY=900 ;; esac; [ "$EVERY" -ge 1 ] || EVERY=900
    case "$EXP" in ''|*[!0-9]*) EXP=0 ;; esac
    case "$MAXB" in ''|*[!0-9]*) MAXB=4000000000 ;; esac
    READBACK=${READBACK:-final}
    ST="$ROOT/state/$TAG"
    ( umask 077; mkdir -p "$ST" "$ROOT/active" ) || die "cannot create $ST"
}

expired() { [ "$(date +%s)" -ge $(( EXP - 30 )) ]; }

lock() {  # waits up to $1 seconds for the job's upload lock
    local i=0 p
    while ! mkdir "$ST/lock" 2> /dev/null; do
        p=$(cat "$ST/lock/pid" 2> /dev/null || true)
        if [ -n "$p" ] && ! kill -0 "$p" 2> /dev/null; then rm -rf "$ST/lock"; continue; fi
        i=$((i + 1)); [ $i -lt "${1:-120}" ] || return 1
        sleep 1
    done
    echo "${BASHPID:-$$}" > "$ST/lock/pid"
}
unlock() { rm -rf "$ST/lock"; }

# PUT/GET through a 0600 curl config, so no URL is in any process list; prints the HTTP code (000: no answer)
http() {  # METHOD URL FILE MAXSEC
    local cfg="$ST/curl.$$.cfg" c
    ( umask 077; printf 'url = "%s"\n' "$2" > "$cfg" )
    if [ "$1" = PUT ]; then
        c=$(curl -sS -o /dev/null -w '%{http_code}' --max-time "$4" --connect-timeout 20 --retry 2 --retry-delay 3 \
            --speed-limit 1024 --speed-time 60 -K "$cfg" -T "$3" 2>> "$ST/curl.err") || true
    else
        c=$(curl -sS -o "$3" -w '%{http_code}' --max-time "$4" --connect-timeout 20 --retry 2 --retry-delay 3 \
            -K "$cfg" 2>> "$ST/curl.err") || true
    fi
    rm -f "$cfg"
    printf '%s' "${c:-000}"
}

tar_results() {  # <out> [exclude-file]: rc 0 or 1 (GNU: files changed while read) is a tarball
    local ex=() p jb rc
    for p in $DEFAULT_EXCLUDES $EXTRA_EXCL; do ex+=("--exclude=$p"); done
    jb=$(basename "$JSON"); ex+=("--exclude=$jb")
    [ -z "${2:-}" ] || ex+=(-X "$2")
    tar -czf "$1" "${ex[@]}" -C "$RES" . 2>> "$ST/tar.err"; rc=$?
    [ $rc -le 1 ] && [ -s "$1" ]
}

# the tarball within max_bytes: full; else without files over max/8; else without files over 1 MiB; else the logs
build_tarball() {  # <out>
    local out=$1 b lim
    tar_results "$out" || { plog "TAR-FAILED (see $ST/tar.err)"; return 1; }
    b=$(fsize "$out"); [ "$b" -gt "$MAXB" ] || { rm -f "$RES/PERSIST_TRUNCATED.txt"; return 0; }
    for lim in $(( MAXB / 8 )) 1048576; do
        ( cd "$RES" && find . -type f -size +"${lim}"c ) > "$ST/big.lst" 2> /dev/null
        {
            echo "box_persist: the results tarball was $b bytes, over max_bytes=$MAXB; files over $lim bytes were left out:"
            ( cd "$RES" && while IFS= read -r f; do echo "$(fsize "$f") $f"; done < "$ST/big.lst" )
        } > "$RES/PERSIST_TRUNCATED.txt"
        tar_results "$out" "$ST/big.lst" || return 1
        b=$(fsize "$out"); [ "$b" -gt "$MAXB" ] || { plog "TRUNCATED limit=$lim omitted=$(grep -c . "$ST/big.lst")"; return 0; }
    done
    echo "box_persist: even without files over 1 MiB the tarball was $b bytes; only the logs were uploaded" >> "$RES/PERSIST_TRUNCATED.txt"
    tar -czf "$out" -C "$RES" PERSIST_TRUNCATED.txt $( cd "$RES" && ls persist.log command.log 2> /dev/null ) 2>> "$ST/tar.err"
    plog "TRUNCATED to logs only"
}

write_manifest() {  # <state> <exit> <why>
    local first=1 seq name sha bytes utc label
    {
        printf '{\n  "tag": "%s",\n  "prefix": "%s",\n  "host": "%s",\n  "updated_utc": "%s",\n' \
            "$TAG" "$PREFIX" "$(hostname 2> /dev/null | tr -dc 'A-Za-z0-9._-')" "$(now_utc)"
        printf '  "state": "%s",\n  "exit": "%s",\n  "why": "%s",\n  "uploads": [' "$1" "$2" "$(printf '%s' "$3" | tr -dc 'A-Za-z0-9=._ -')"
        while IFS='|' read -r seq name sha bytes utc label; do
            [ -n "$seq" ] || continue
            [ $first = 1 ] || printf ','
            printf '\n    {"seq": %s, "name": "%s", "sha256": "%s", "bytes": %s, "utc": "%s", "label": "%s"}' \
                "$seq" "$name" "$sha" "$bytes" "$utc" "$label"
            first=0
        done < "$ST/entries"
        printf '\n  ]\n}\n'
    } > "$ST/manifest.json"
}

# one upload: label periodic -> the next ring slot; final|flush -> final.tgz
snapshot() {  # <label> [exit] [why]
    local label=$1 rcv=${2:-} why=${3:-} seq pseq slot name pk sk gk tgz side c c2 sha bytes back state maxs
    command -v curl > /dev/null 2>&1 || { plog "PERSIST-DISABLED no curl on this box"; return 1; }
    if expired; then
        [ -e "$ST/expired_logged" ] || { plog "EXPIRED the job's upload URLs expired at $EXPU; nothing more is sent"; : > "$ST/expired_logged"; }
        return 2
    fi
    lock "$([ "$label" = periodic ] && echo 5 || echo 90)" || { plog "LOCK-BUSY label=$label (another upload of this job holds $ST/lock)"; return 1; }
    seq=$(( $(cat "$ST/seq" 2> /dev/null || echo 0) + 1 ))
    if [ "$label" = periodic ]; then
        pseq=$(cat "$ST/pseq" 2> /dev/null || echo 0)
        slot=$(printf '%02d' $(( pseq % SLOTS ))); name=$slot.tgz
        pk=slot.$slot.put; sk=slot.$slot.sha.put; gk=slot.$slot.get; maxs=$PUT_MAX
    else
        name=final.tgz; pk=final.put; sk=final.sha.put; gk=final.get; maxs=$FINAL_PUT_MAX
    fi
    tgz="$ST/upload.tgz"; side="$ST/upload.sha256"; rm -f "$tgz"
    build_tarball "$tgz" || { unlock; return 1; }
    sha=$(sum256 "$tgz"); bytes=$(fsize "$tgz")
    printf '%s  %s\n' "$sha" "$name" > "$side"
    c=$(http PUT "$(jget "$pk")" "$tgz" "$maxs")
    case "$c" in
        2*) ;;
        403) plog "REJECTED name=$name http=403 (the URL expired or is not signed for this key)"; unlock; return 1 ;;
        *) plog "FAILED name=$name http=$c bytes=$bytes"; unlock; return 1 ;;
    esac
    c2=$(http PUT "$(jget "$sk")" "$side" 120)
    case "$c2" in 2*) ;; *) plog "FAILED name=$name.sha256 http=$c2" ;; esac
    if [ "$READBACK" = all ] || { [ "$READBACK" = final ] && [ "$label" != periodic ]; }; then
        back="$ST/readback.tgz"; rm -f "$back"
        c=$(http GET "$(jget "$gk")" "$back" "$maxs")
        if [ "${c#2}" != "$c" ] && [ -f "$back" ] && [ "$(sum256 "$back")" = "$sha" ]; then plog "READBACK-OK name=$name"
        else plog "READBACK-MISMATCH name=$name http=$c"; fi
        rm -f "$back"
    fi
    echo "$seq|$name|$sha|$bytes|$(now_utc)|$label" >> "$ST/entries"
    echo "$seq" > "$ST/seq"
    [ "$label" != periodic ] || echo $(( pseq + 1 )) > "$ST/pseq"
    case "$label" in periodic) state=running ;; flush) state=flushed ;; *) state=final ;; esac
    write_manifest "$state" "$rcv" "$why"
    c=$(http PUT "$(jget manifest.put)" "$ST/manifest.json" 120)
    case "$c" in 2*) ;; *) plog "FAILED name=manifest.json http=$c" ;; esac
    plog "UPLOADED name=$name seq=$seq label=$label bytes=$bytes sha256=$sha"
    rm -f "$tgz" "$side"
    unlock
    return 0
}

register() {
    { printf 'PID=%q\nJSON=%q\nRES=%q\nSTARTED=%q\n' "$$" "$JSON" "$RES" "$(now_utc)"; } > "$ROOT/active/$TAG.env.tmp" \
        && mv -f "$ROOT/active/$TAG.env.tmp" "$ROOT/active/$TAG.env"
}
unregister() { rm -f "$ROOT/active/$TAG.env"; }

# ---------------------------------------------------------------- --flush-all: the dead-man hook
flush_all() {  # <seconds>
    local budget=${1:-120} f pids='' p t0 left
    t0=$(date +%s)
    for f in "$ROOT"/active/*.env; do
        [ -f "$f" ] || continue
        ( JSON=''; RES=''; . "$f"; [ -n "$JSON" ] && [ -n "$RES" ] || exit 1
          PERSIST_FINAL_PUT_MAX_SEC=$budget exec bash "$SELF" --snapshot "$JSON" "$RES" flush ) &
        pids="$pids $!"
        echo "box_persist: flush $(basename "$f" .env) pid $!"
    done
    [ -n "$pids" ] || { echo "box_persist: flush-all: no active jobs"; return 0; }
    while :; do
        left=''
        for p in $pids; do kill -0 "$p" 2> /dev/null && left="$left $p"; done
        [ -n "$left" ] || break
        if [ $(( $(date +%s) - t0 )) -ge "$budget" ]; then
            echo "box_persist: flush-all: budget ${budget}s spent; stopping$left"
            kill $left 2> /dev/null; break
        fi
        sleep 1
    done
    echo "box_persist: flush-all done in $(( $(date +%s) - t0 ))s"
}

# ---------------------------------------------------------------- the run
CPID=''; FINAL_DONE=0; IN_FINAL=0
kill_job() {  # <signal>: the command's whole process group (it was started under set -m)
    [ -n "$CPID" ] || return 0
    kill -"$1" -- -"$CPID" 2> /dev/null || kill -"$1" "$CPID" 2> /dev/null || true
}
final_upload() {  # <exit> <why>
    [ "$FINAL_DONE" = 0 ] || return 0
    IN_FINAL=1
    trap '' TERM INT        # ignored dispositions are inherited: curl and tar finish (bounded by their own timeouts)
    snapshot final "$1" "$2"; local r=$?
    FINAL_DONE=1; unregister
    if [ $r = 0 ]; then say "final upload of $TAG done: $(tail -1 "$RES/persist.log" 2> /dev/null)"
    else say "final upload of $TAG NOT done (rc=$r): $(tail -1 "$RES/persist.log" 2> /dev/null)"; fi
}
on_sig() {  # <name> <exit code>
    [ "$IN_FINAL" = 0 ] || return 0
    plog "SIGNAL $1: the command gets TERM, then the final upload"
    kill_job TERM
    local i=0
    while [ $i -lt "$GRACE" ] && kill -0 "$CPID" 2> /dev/null; do sleep 1; i=$((i + 1)); done
    kill -0 "$CPID" 2> /dev/null && kill_job KILL
    final_upload "$2" "signal=$1"
    trap - EXIT
    exit "$2"
}
on_exit() { [ "$FINAL_DONE" = 1 ] || final_upload "${RC:-}" "exit-trap"; }

run_job() {
    JSON=$1; RES=$2; shift 2
    [ "${1:-}" = -- ] && shift
    [ $# -gt 0 ] || die "usage: box_persist_run.sh <persist.json> <results-dir> -- <command...>"
    mkdir -p "$RES" || die "cannot create $RES"; RES=$(cd "$RES" && pwd)
    JSON=$(cd "$(dirname "$JSON")" && pwd)/$(basename "$JSON")
    load_job
    register
    trap '' HUP PIPE
    trap 'on_sig TERM 143' TERM
    trap 'on_sig INT 130' INT
    trap 'on_exit' EXIT
    export MOJOLEARN_RESULTS_DIR="$RES" MOJOLEARN_PERSIST_TAG="$TAG"
    command -v curl > /dev/null 2>&1 || plog "PERSIST-DISABLED no curl on this box: the command runs, nothing reaches R2"
    plog "START tag=$TAG prefix=$PREFIX pid=$$ every=${EVERY}s slots=$SLOTS expires=$EXPU cmd=$*"
    set -m
    ( trap - TERM INT; "$@" 2>&1 | tee -a "$RES/command.log"; exit "${PIPESTATUS[0]}" ) &
    CPID=$!
    set +m
    local next=$(( $(date +%s) + EVERY ))
    while kill -0 "$CPID" 2> /dev/null; do
        sleep "$POLL" & wait $! 2> /dev/null
        if [ "$(date +%s)" -ge "$next" ] && kill -0 "$CPID" 2> /dev/null; then
            snapshot periodic || true
            next=$(( $(date +%s) + EVERY ))
        fi
    done
    wait "$CPID" 2> /dev/null; RC=$?
    plog "EXIT rc=$RC"
    final_upload "$RC" "exit=$RC"
    trap - EXIT
    exit "$RC"
}

case "${1:-}" in
    --flush-all) shift; flush_all "${1:-120}" ;;
    --snapshot)
        shift; JSON="${1:?--snapshot <json> <results> [label]}"; RES="${2:?results dir}"; label=${3:-flush}
        load_job; snapshot "$label" "" "$label"; exit $? ;;
    ''|-h|--help) sed -n 2,33p "$0"; exit 2 ;;
    *) run_job "$@" ;;
esac
