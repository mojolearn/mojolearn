#!/bin/bash
# tools/box_persist_run.sh -- RUNS ON A RENTED BOX: a job whose results reach R2 from the box itself, so they
# survive the Mac disconnecting, the box's dead-man firing, or both (Andrew, 2026-10-26).
#
#   box_persist_run.sh <persist.json> <results-dir> -- <command...>
#       runs the command (stdout+stderr also tee'd to <results-dir>/command.log), and every every_sec seconds
#       streams a tarball of <results-dir> to the next of the job's slots (a ring: 00.tgz, 01.tgz, ...) with a
#       sha256 sidecar and a manifest.json. On ANY exit (normal, TERM, INT, the EXIT trap) it uploads
#       final.tgz. Every upload is streamed `tar | gzip` cut into pieces of max_bytes (final.00.tgz,
#       final.01.tgz, ...; ring: 00.00.tgz, ...), each with a sha256 sidecar, at most one piece on the box's disk at
#       a time; <name>.sha256 holds the whole stream's sha256 and the manifest lists the pieces in order
#       (tools/box_persist.py fetch concatenates them back into <name>). HUP and PIPE are ignored, so an ssh session that dies under `dev_pod.sh run` or
#       `do_amd_steward.sh run` leaves the job running and persisting. Exit status: the command's.
#   box_persist_run.sh --flush-all [seconds]
#       THE DEAD-MAN HOOK: uploads final.tgz for every active job on this box, bounded by <seconds> (default
#       120). The self-destruct paths (tools/runpod_guard.sh watchdog, the gpu_queue_box.sh lease delete,
#       tools/do_amd_steward.sh's mojolearn-selfkill unit) call it before they delete the box.
#   box_persist_run.sh --snapshot <persist.json> <results-dir> [periodic|hard|final|flush]
#       one upload now (flush-all uses it per job; the gpu_queue_box.sh idle delete uses it for the pod's queue dir).
#       hard: a ring-slot upload like periodic, labelled `hard` in the manifest (the job stays running): a job that
#       has finished the part its caller waits for publishes it at once (tools/ops/release_gate_job.sh phase
#       hard-done; tools/ops/release_gate.py brings that slot home and returns while the job runs on).
#       A running job's command finds this script and its json in MOJOLEARN_PERSIST_RUNNER / MOJOLEARN_PERSIST_JSON.
#
# <persist.json> is minted on the Mac by tools/box_persist.py (`mint`/`prepare`) and reaches the box on ssh
# stdin as a 0600 file. It holds presigned PUT/GET URLs for this job's keys only, which expire (at most 7
# days); no credential ever reaches the box. It is a flat JSON object, one "key": "value" pair per line, so
# this script reads it with sed: needs bash, curl, tar, gzip (pigz when present), dd, mkfifo, sha256sum or shasum.
#
# Every upload writes one status line to <results-dir>/persist.log:
#   UPLOADED name=00.tgz seq=3 label=periodic bytes=.. sha256=..   FAILED name=.. http=..   REJECTED (403: expired
#   or wrong key)   EXPIRED (URLs past their expiry: nothing more is sent)   READBACK-OK|READBACK-MISMATCH
# Excluded from every tarball: venv, venv-*, .venv, .pixi, *.whl, .cache, __pycache__, .mojo_cache,
# node_modules, *.persist.json, persist.json, plus the json's "excludes". Nothing else is left out: the json's
# `pieces` (minted from --max-total-gb) bounds one upload; a results dir that needs more pieces logs OVER-BUDGET
# and that upload fails (re-mint with `box_persist.py remint --tag <tag> --max-total-gb N`). No whole tarball is
# ever staged on the box (2026-11-14: a 27 GB staging tarball could never leave the full 80 GB L40S pod).
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

# what `box_persist.py remint` may change under a running job (same tag and prefix): re-read before every upload
load_limits() {
    EXP=$(jget expires_epoch); EXPU=$(jget expires_utc); MAXB=$(jget max_bytes); READBACK=$(jget readback)
    PIECES=$(jget pieces)
    case "$EXP" in ''|*[!0-9]*) EXP=0 ;; esac
    case "$MAXB" in ''|*[!0-9]*) MAXB=4000000000 ;; esac
    case "$PIECES" in ''|*[!0-9]*) PIECES=1 ;; esac; [ "$PIECES" -ge 1 ] || PIECES=1
    # one piece = BCOUNT blocks of BS bytes, at most max_bytes (dd iflag=fullblock reads exactly that much)
    if [ "$MAXB" -ge 1048576 ]; then BS=1048576; else BS=4096; fi
    BCOUNT=$(( MAXB / BS )); [ "$BCOUNT" -ge 1 ] || BCOUNT=1
    PIECE_BYTES=$(( BS * BCOUNT ))
    READBACK=${READBACK:-final}
}

# the flat json: "key": "value" on its own line (tools/box_persist.py writes every value as a string)
jget() {
    local k=${1//./\\.}
    sed -n "s/^[[:space:]]*\"$k\":[[:space:]]*\"\([^\"]*\)\",\{0,1\}[[:space:]]*\$/\1/p" "$JSON" | head -1
}

load_job() {  # JSON RES set
    [ -r "$JSON" ] || die "no persist json $JSON"
    TAG=$(jget tag); PREFIX=$(jget prefix); SLOTS=$(jget slots); EVERY=$(jget every_sec)
    EXTRA_EXCL=$(jget excludes)
    case "$TAG" in ''|*[!A-Za-z0-9._-]*) die "persist json $JSON has no valid tag" ;; esac
    case "$SLOTS" in ''|*[!0-9]*) SLOTS=1 ;; esac; [ "$SLOTS" -ge 1 ] || SLOTS=1
    case "$EVERY" in ''|*[!0-9]*) EVERY=900 ;; esac; [ "$EVERY" -ge 1 ] || EVERY=900
    load_limits
    ST="$ROOT/state/$TAG"
    ( umask 077; mkdir -p "$ST" "$ROOT/active" ) || die "cannot create $ST"
}

expired() { [ "$(date +%s)" -ge $(( EXP - 30 )) ]; }

lock() {  # waits up to $1 seconds for the job's upload lock
    local i=0 p
    while ! mkdir "$ST/lock" 2> /dev/null; do
        p=$(cat "$ST/lock/pid" 2> /dev/null || true)
        if [ -n "$p" ] && ! kill -0 "$p" 2> /dev/null; then rm -rf "$ST/lock"; continue; fi
        # our own: a signal trap's final upload interrupted this process's periodic one, which never resumes
        if [ "$p" = "${BASHPID:-$$}" ]; then rm -rf "$ST/lock"; continue; fi
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

gz_stream() {  # gzip stdin to stdout on every core when pigz is there (HARD RULE: rented cores are all used)
    if command -v pigz > /dev/null 2>&1; then pigz -c; else gzip -c; fi
}
sha_stdin() { if command -v sha256sum > /dev/null 2>&1; then sha256sum | cut -c1-64; else shasum -a 256 | cut -c1-64; fi; }

tar_stream() {  # the results dir as one tar on stdout; rc 0 or 1 (GNU: files changed while read) is a tarball
    local ex=() p jb
    for p in $DEFAULT_EXCLUDES $EXTRA_EXCL; do ex+=("--exclude=$p"); done
    jb=$(basename "$JSON"); ex+=("--exclude=$jb")
    tar -cf - "${ex[@]}" -C "$RES" . 2>> "$ST/tar.err"
}

# reads the gzip stream on stdin and PUTs it in pieces of at most max_bytes: <base>.00.tgz, <base>.01.tgz, ... each
# with a sha256 sidecar. At most ONE piece is on the box's disk at a time: dd reads exactly one piece (fullblock)
# while tar waits on the pipe (2026-11-14: the whole-results staging tarball could never leave a nearly full pod).
# Each piece also goes to fd 7, the whole stream's sha256. Writes "name:sha:bytes" lines to $ST/pieces.lst.
# rc 0 done, 1 an upload failed, 3 the stream needs more than the minted `pieces`.
put_pieces() {  # <base> <url key prefix> <max secs> <readback 0|1>
    local base=$1 kp=$2 maxs=$3 rb=$4 i=0 pc="$ST/piece" pn name sha bytes c
    while :; do
        rm -f "$pc" "$pc.sha256"
        if [ "$i" -ge "$PIECES" ]; then
            dd bs=1 count=1 of="$pc" 2> /dev/null
            if [ -s "$pc" ]; then rm -f "$pc"; plog "OVER-BUDGET name=$base.tgz the results need more than pieces=$PIECES x max_bytes=$MAXB (mint with a larger --max-total-gb)"; return 3; fi
            rm -f "$pc"; return 0
        fi
        dd bs="$BS" count="$BCOUNT" iflag=fullblock of="$pc" 2>> "$ST/dd.err" || { plog "PIECE-FAILED name=$base (see $ST/dd.err)"; rm -f "$pc"; return 1; }
        bytes=$(fsize "$pc")
        if [ "$bytes" -eq 0 ]; then rm -f "$pc"; [ "$i" -gt 0 ] && return 0; plog "PIECE-FAILED name=$base empty stream"; return 1; fi
        cat "$pc" >&7
        pn=$(printf '%02d' "$i"); name="$base.$pn.tgz"
        sha=$(sum256 "$pc")
        printf '%s  %s\n' "$sha" "$name" > "$pc.sha256"
        c=$(http PUT "$(jget "$kp.p$pn.put")" "$pc" "$maxs")
        case "$c" in
            2*) ;;
            403) plog "REJECTED name=$name http=403 (the URL expired or is not signed for this key)"; rm -f "$pc" "$pc.sha256"; return 1 ;;
            *) plog "FAILED name=$name http=$c bytes=$bytes"; rm -f "$pc" "$pc.sha256"; return 1 ;;
        esac
        c=$(http PUT "$(jget "$kp.p$pn.sha.put")" "$pc.sha256" 120)
        case "$c" in 2*) ;; *) plog "FAILED name=$name.sha256 http=$c" ;; esac
        if [ "$rb" = 1 ]; then
            rm -f "$pc.back"
            c=$(http GET "$(jget "$kp.p$pn.get")" "$pc.back" "$maxs")
            if [ "${c#2}" != "$c" ] && [ -f "$pc.back" ] && [ "$(sum256 "$pc.back")" = "$sha" ]; then plog "READBACK-OK name=$name"
            else plog "READBACK-MISMATCH name=$name http=$c"; echo "$name" >> "$ST/readback.bad"; fi
            rm -f "$pc.back"
        fi
        echo "$name:$sha:$bytes" >> "$ST/pieces.lst"
        rm -f "$pc" "$pc.sha256"
        i=$((i + 1))
        [ "$bytes" -ge "$PIECE_BYTES" ] || return 0
    done
}

write_manifest() {  # <state> <exit> <why>
    local first=1 seq name sha bytes utc label pieces pf p pn ps pb
    {
        printf '{\n  "tag": "%s",\n  "prefix": "%s",\n  "host": "%s",\n  "updated_utc": "%s",\n' \
            "$TAG" "$PREFIX" "$(hostname 2> /dev/null | tr -dc 'A-Za-z0-9._-')" "$(now_utc)"
        printf '  "state": "%s",\n  "exit": "%s",\n  "why": "%s",\n  "uploads": [' "$1" "$2" "$(printf '%s' "$3" | tr -dc 'A-Za-z0-9=._ -')"
        while IFS='|' read -r seq name sha bytes utc label pieces; do
            [ -n "$seq" ] || continue
            [ $first = 1 ] || printf ','
            printf '\n    {"seq": %s, "name": "%s", "sha256": "%s", "bytes": %s, "utc": "%s", "label": "%s", "pieces": [' \
                "$seq" "$name" "$sha" "$bytes" "$utc" "$label"
            pf=1
            for p in $(printf '%s' "${pieces:-}" | tr ',' ' '); do
                pn=${p%%:*}; ps=${p#*:}; pb=${ps#*:}; ps=${ps%%:*}
                [ $pf = 1 ] || printf ', '
                printf '{"name": "%s", "sha256": "%s", "bytes": %s}' "$pn" "$ps" "$pb"
                pf=0
            done
            printf ']}'
            first=0
        done < "$ST/entries"
        printf '\n  ]\n}\n'
    } > "$ST/manifest.json"
}

# one upload: label periodic|hard -> the next ring slot (NN.tgz); final|flush -> final.tgz. Either is streamed in
# pieces (put_pieces); the manifest lists the pieces and the whole stream's sha256, which also goes to <name>.sha256.
snapshot() {  # <label> [exit] [why]
    local label=$1 rcv=${2:-} why=${3:-} seq pseq slot name base kp sk maxs ring rb side state hp fifo pst sha bytes pieces n c
    command -v curl > /dev/null 2>&1 || { plog "PERSIST-DISABLED no curl on this box"; return 1; }
    [ -r "$JSON" ] && load_limits
    if expired; then
        [ -e "$ST/expired_logged" ] || { plog "EXPIRED the job's upload URLs expired at $EXPU; nothing more is sent"; : > "$ST/expired_logged"; }
        return 2
    fi
    ring=0; case "$label" in periodic|hard) ring=1 ;; esac
    lock "$([ "$label" = periodic ] && echo 5 || echo 90)" || { plog "LOCK-BUSY label=$label (another upload of this job holds $ST/lock)"; return 1; }
    seq=$(( $(cat "$ST/seq" 2> /dev/null || echo 0) + 1 ))
    if [ "$ring" = 1 ]; then
        pseq=$(cat "$ST/pseq" 2> /dev/null || echo 0)
        slot=$(printf '%02d' $(( pseq % SLOTS ))); base=$slot
        kp=slot.$slot; sk=slot.$slot.sha.put; maxs=$PUT_MAX
    else
        base=final; kp=final; sk=final.sha.put; maxs=$FINAL_PUT_MAX
    fi
    name=$base.tgz
    rb=0; if [ "$READBACK" = all ] || { [ "$READBACK" = final ] && [ "$label" != periodic ]; }; then rb=1; fi
    side="$ST/upload.sha256"; fifo="$ST/whole.fifo"
    rm -f "$side" "$fifo" "$ST/whole.sha" "$ST/readback.bad"; : > "$ST/pieces.lst"
    mkfifo "$fifo" || { plog "FAILED name=$name (mkfifo $fifo)"; unlock; return 1; }
    sha_stdin < "$fifo" > "$ST/whole.sha" &
    hp=$!
    exec 7> "$fifo"
    tar_stream 7>&- | gz_stream 7>&- | put_pieces "$base" "$kp" "$maxs" "$rb"
    pst=("${PIPESTATUS[@]}")
    exec 7>&-
    wait "$hp" 2> /dev/null
    rm -f "$fifo" "$ST/piece" "$ST/piece.sha256"
    if [ "${pst[2]}" != 0 ]; then unlock; return 1; fi
    if [ "${pst[0]}" -gt 1 ] || [ "${pst[1]}" != 0 ]; then plog "TAR-FAILED name=$name tar=${pst[0]} gzip=${pst[1]} (see $ST/tar.err)"; unlock; return 1; fi
    sha=$(cat "$ST/whole.sha" 2> /dev/null); bytes=$(awk -F: '{ s += $3 } END { printf "%d", s }' "$ST/pieces.lst")
    pieces=$(paste -sd, "$ST/pieces.lst"); n=$(grep -c . "$ST/pieces.lst")
    printf '%s  %s\n' "$sha" "$name" > "$side"
    c=$(http PUT "$(jget "$sk")" "$side" 120)
    case "$c" in 2*) ;; *) plog "FAILED name=$name.sha256 http=$c" ;; esac
    if [ "$rb" = 1 ]; then
        if [ -s "$ST/readback.bad" ]; then plog "READBACK-MISMATCH name=$name pieces=$(paste -sd, "$ST/readback.bad")"
        else plog "READBACK-OK name=$name pieces=$n"; fi
    fi
    echo "$seq|$name|$sha|$bytes|$(now_utc)|$label|$pieces" >> "$ST/entries"
    echo "$seq" > "$ST/seq"
    [ "$ring" != 1 ] || echo $(( pseq + 1 )) > "$ST/pseq"
    case "$label" in periodic|hard) state=running ;; flush) state=flushed ;; *) state=final ;; esac
    write_manifest "$state" "$rcv" "$why"
    c=$(http PUT "$(jget manifest.put)" "$ST/manifest.json" 120)
    case "$c" in 2*) ;; *) plog "FAILED name=manifest.json http=$c" ;; esac
    plog "UPLOADED name=$name seq=$seq label=$label bytes=$bytes sha256=$sha pieces=$n"
    rm -f "$side"
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
        ( JSON=''; RES=''; PID=''; . "$f"; [ -n "$JSON" ] && [ -n "$RES" ] || exit 1
          PERSIST_FINAL_PUT_MAX_SEC=$budget bash "$SELF" --snapshot "$JSON" "$RES" flush || exit 1
          # a job that already ended (its final had failed): done once this flush landed
          [ -n "$PID" ] && kill -0 "$PID" 2> /dev/null || rm -f "$f" ) &
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
    FINAL_DONE=1
    # a final that did not land stays registered (its pid dead): the box's --flush-all retries it before any delete
    if [ $r = 0 ] || [ $r = 2 ]; then unregister; else plog "KEPT-ACTIVE the final upload is retried by --flush-all"; fi
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
    # MOJOLEARN_BOX_PERSIST=1: this command's results dir reaches R2 from the box (presigned PUTs minted on the Mac),
    # so tools/bench_board.py may keep its evidence archives local under it (--evidence-local-only) with no
    # credential on the box
    export MOJOLEARN_RESULTS_DIR="$RES" MOJOLEARN_PERSIST_TAG="$TAG" MOJOLEARN_BOX_PERSIST=1
    # the command may publish a finished part at once: bash "$MOJOLEARN_PERSIST_RUNNER" --snapshot "$MOJOLEARN_PERSIST_JSON" "$RES" hard
    export MOJOLEARN_PERSIST_RUNNER="$SELF" MOJOLEARN_PERSIST_JSON="$JSON"
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
    ''|-h|--help) sed -n 2,41p "$0"; exit 2 ;;
    *) run_job "$@" ;;
esac
