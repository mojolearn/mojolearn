#!/bin/bash
# tools/do_cpu_leg.sh -- THE ONLY WAY to rent DigitalOcean CPU build boxes
# (cheap fast-CPU cross-compile droplets). Every leg carries two dead-men and
# a hard lease, so a forgotten box ends by itself.
#
# Why (Andrew, 2026-10-11): four s-8vcpu-16gb-amd droplets
# (mojolearn-targeted-ab-{nvidia,amd,nvidia-02,amd-02}-cpu-6fe3cfce38, nyc2)
# were created 2026-10-07 by ad-hoc API calls from a Codex session (no repo
# script, no state file, no dead-man) and billed for 4.5 days (~$115).
#
#   tools/do_cpu_leg.sh up <name> [minutes]     RENTS one droplet (default 240 min, max 720)
#   tools/do_cpu_leg.sh extend <name> [minutes] both deadlines -> now + minutes (max 720); renames the droplet
#   tools/do_cpu_leg.sh status                  every leg: id, ip, deadline, dead-man pid
#   tools/do_cpu_leg.sh down <name>             DELETE, verify GET 404, then remove state
#   tools/do_cpu_leg.sh sweep [--delete]        droplets no live state file covers (ours or the
#                                               steward's), with a delete command for each;
#                                               --delete removes untracked ones tagged
#                                               mojolearn-cpu-leg or named mojolearn-*cpu* older than 6 h
#   --dry-run (any position)                    print the API bodies, call nothing
#
# CRON (not installed by this script; add it yourself with `crontab -e`):
#   */30 * * * * /bin/bash $HOME/CascadeProjects/mojolearn/tools/do_cpu_leg.sh sweep --delete >> $HOME/mojolearn-evidence/do-cpu-leg/sweep.log 2>&1
#
# THE LEASE (the mechanism of tools/do_amd_steward.sh, reused):
#  * Mac dead-man: a nohup'd loop armed BEFORE the create. It reads
#    $S/<name>/mac_deadline (droplet deadline + 10 min) and DELETEs the droplet
#    by id and by name under tag mojolearn-cpu-leg when it passes.
#  * On-droplet self-destruct: a systemd service (Restart=always) that DELETEs
#    its own droplet when /root/.mojolearn-cpu-guard/deadline passes or that
#    file is missing/unreadable (fail safe). Like the steward it needs a token
#    on the box to call DELETE. Put a DigitalOcean CUSTOM-SCOPED token
#    (droplet:read + droplet:delete only) in $MOJOLEARN_DO_SELFKILL_TOKEN_FILE
#    (default ~/.mojolearn_do_selfkill_token) and that is what goes on the box;
#    without it the full token is delivered as the steward does, with a warning.
#    A powered-off droplet still bills, so `shutdown` alone is not a dead-man.
#  * Name carries the deadline: mojolearn-cpu-<name>-until-<MMDDHHMM>z (UTC).
#  * sweep treats a state file whose Mac deadline is >30 min past as dead.
set -euo pipefail
S="${MOJOLEARN_DO_CPU_STATE:-$HOME/mojolearn-evidence/do-cpu-leg}"
STEWARD_STATE="${MOJOLEARN_STEWARD_DO_STATE:-$HOME/mojolearn-evidence/do-amd-steward}/state.env"
DO_API="${MOJOLEARN_DO_API:-https://api.digitalocean.com/v2}"
DO_TOKFILE="${MOJOLEARN_DO_TOKEN_FILE:-$HOME/.mojolearn_do_token}"
SK_TOKFILE="${MOJOLEARN_DO_SELFKILL_TOKEN_FILE:-$HOME/.mojolearn_do_selfkill_token}"
CURL="${MOJOLEARN_DO_CURL:-curl}"           # tests point this at a mock
DO_SIZE="${MOJOLEARN_DO_CPU_SIZE:-s-8vcpu-16gb-amd}"   # the targeted-ab droplets' size
DO_REGION="${MOJOLEARN_DO_CPU_REGION:-nyc2}"
DO_IMAGE="${MOJOLEARN_DO_CPU_IMAGE:-ubuntu-22-04-x64}"
DO_SSH_KEY_FP="df:f7:6b:0c:56:da:48:a5:6f:6d:ae:44:af:de:f3:0b"   # ~/.ssh/id_ed25519 (steward's)
DO_TAG=mojolearn-cpu-leg
DEFAULT_MIN=240; MAX_MIN=720; MAX_LEGS=4; SWEEP_AGE_H=6
G=/root/.mojolearn-cpu-guard
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -o BatchMode=yes -i $HOME/.ssh/id_ed25519 -o IdentitiesOnly=yes"

die() { echo "do-cpu-leg: $*" >&2; exit 1; }
say() { echo "do-cpu-leg: $*"; }
now() { date +%s; }
utc() { python3 -c 'import sys,time; print(time.strftime(sys.argv[2], time.gmtime(int(sys.argv[1]))))' "$1" "${2:-%Y-%m-%dT%H:%M:%SZ}"; }
TMPD="$(mktemp -d "${TMPDIR:-/tmp}/do-cpu-leg.XXXXXX")"
trap 'rm -rf "$TMPD"' EXIT

DRY=0; ARGS=()
for a in "$@"; do [ "$a" = --dry-run ] && DRY=1 || ARGS+=("$a"); done
set -- "${ARGS[@]+"${ARGS[@]}"}"

mk_curlrc() {  # tokenfile out
    [ -f "$1" ] || return 1
    local t=""; IFS= read -r t < "$1" || [ -n "$t" ] || return 1
    t="${t//[$'\t\r\n ']/}"; [ -n "$t" ] || return 1
    ( umask 077; printf 'header = "Authorization: Bearer %s"\nsilent\nshow-error\n' "$t" > "$2" )
}
load_token() { mk_curlrc "$DO_TOKFILE" "$TMPD/curlrc" || die "no DigitalOcean token ($DO_TOKFILE)"; }
do_call() {  # METHOD PATH [jsonfile]; sets DO_CODE, body in $TMPD/do.body
    if [ "$DRY" = 1 ]; then
        say "DRY-RUN $1 $DO_API/$2${3:+ body: $(cat "$3")}"; DO_CODE=dry; echo '{}' > "$TMPD/do.body"; return
    fi
    if [ -n "${3:-}" ]; then
        DO_CODE=$("$CURL" -K "$TMPD/curlrc" --max-time 120 -o "$TMPD/do.body" -w '%{http_code}' -X "$1" \
            -H 'Content-Type: application/json' --data-binary "@$3" "$DO_API/$2") || DO_CODE=000
    else
        DO_CODE=$("$CURL" -K "$TMPD/curlrc" --max-time 120 -o "$TMPD/do.body" -w '%{http_code}' -X "$1" "$DO_API/$2") || DO_CODE=000
    fi
}
jq_py() { python3 -c "import json,sys; d=json.load(open('$TMPD/do.body')); $1" 2>/dev/null || true; }
valid_name() { [[ "$1" =~ ^[a-z0-9][a-z0-9-]{0,23}$ ]] || die "name must be 1-24 chars of [a-z0-9-]: '$1'"; }
valid_min() {
    [[ "$1" =~ ^[0-9]+$ ]] || die "minutes must be a whole number: '$1'"
    [ "$1" -ge 10 ] || die "minutes must be >= 10"
    [ "$1" -le "$MAX_MIN" ] || die "minutes $1 exceeds the hard maximum lease ($MAX_MIN)"
}
dname() { echo "mojolearn-cpu-$1-until-$(utc "$2" %m%d%H%M)z"; }
set_file() { printf '%s\n' "$2" > "$1.tmp" && mv "$1.tmp" "$1"; }
live_state() {  # dir -> 0 if its Mac deadline is not >30 min past
    [ -f "$1/state.env" ] || return 1
    local d; d=$(cat "$1/mac_deadline" 2>/dev/null || echo 0); [[ "$d" =~ ^[0-9]+$ ]] || d=0
    [ "$(now)" -lt $(( d + 1800 )) ]
}
live_legs() { local n=0 d; for d in "$S"/*/; do [ -d "$d" ] && live_state "${d%/}" && n=$((n+1)); done; echo $n; }
bx() { perl -e 'alarm shift; exec @ARGV' "$1" ssh $SSH_OPTS "root@$IP" "$2"; }

arm_mac_deadman() {  # leg dir
    local L="$1"
    cp "$TMPD/curlrc" "$L/curlrc"; chmod 600 "$L/curlrc"
    cat > "$L/deadman.sh" <<DM
#!/bin/sh
# tools/do_cpu_leg.sh Mac dead-man for $L: DELETEs the droplet at mac_deadline.
set -u
trap '' HUP INT
while :; do
    [ -d '$L' ] || exit 0
    d=\$(cat '$L/mac_deadline' 2>/dev/null); case "\$d" in ''|*[!0-9]*) d=0 ;; esac
    [ "\$(date +%s)" -ge "\$d" ] && break
    sleep 30
done
LOG='$L/deadman.log'
echo "\$(date -u +%FT%TZ) dead-man firing" >> "\$LOG"
n=0
while [ \$n -lt 20 ]; do
    ids="\$(cat '$L/droplet_id' 2>/dev/null)"
    $CURL -K '$L/curlrc' --max-time 60 -o '$L/dm_list.json' '$DO_API/droplets?tag_name=$DO_TAG&per_page=200' >> "\$LOG" 2>&1
    ids="\$ids \$(python3 -c 'import json,sys; print(" ".join(str(x["id"]) for x in json.load(open(sys.argv[1])).get("droplets", []) if x.get("name") == open(sys.argv[2]).read().strip()))' '$L/dm_list.json' '$L/droplet_name' 2>/dev/null)"
    left=0
    for id in \$ids; do
        c=\$($CURL -K '$L/curlrc' --max-time 60 -o /dev/null -w '%{http_code}' -X DELETE "$DO_API/droplets/\$id" 2>> "\$LOG")
        echo "\$(date -u +%FT%TZ) DELETE \$id -> \$c" >> "\$LOG"
        case "\$c" in 2*|404) ;; *) left=1 ;; esac
    done
    [ \$left = 0 ] && exit 0
    sleep 60; n=\$((n + 1))
done
DM
    sh -n "$L/deadman.sh" || die "the Mac dead-man did not compose"
    nohup sh -c 'trap "" HUP INT; exec sh "$0"' "$L/deadman.sh" > /dev/null 2>&1 < /dev/null &
    echo $! > "$L/deadman.pid"; sleep 1
    kill -0 "$(cat "$L/deadman.pid")" 2>/dev/null || die "the Mac dead-man did not start"
}

selfkill_installer() {  # droplet id
    cat <<INST
set -e
umask 077; mkdir -p $G
cat > $G/selfkill.sh <<'SK'
#!/bin/sh
# tools/do_cpu_leg.sh ON-DROPLET self-destruct: DELETEs droplet $1 when $G/deadline
# passes, or when that file is missing or unreadable.
set -u
while :; do
    d=\$(cat $G/deadline 2>/dev/null); case "\$d" in ''|*[!0-9]*) d=0 ;; esac
    [ "\$(date +%s)" -ge "\$d" ] && break
    sleep 20
done
n=1
while [ \$n -le 40 ]; do
    c=\$(curl -K $G/curlrc --max-time 60 -o /dev/null -w '%{http_code}' -X DELETE '$DO_API/droplets/$1')
    echo "\$(date -u +%FT%TZ) DELETE $1 attempt \$n -> \$c" >> $G/selfkill.out
    case "\$c" in 2*|404) sleep 3600 ;; esac
    sleep 15; n=\$((n + 1))
done
SK
chmod 700 $G/selfkill.sh
cat > /etc/systemd/system/mojolearn-selfkill.service <<'UNIT'
[Unit]
Description=mojolearn CPU leg self-destruct (DELETE this droplet at $G/deadline)
After=network-online.target
[Service]
ExecStart=/bin/sh $G/selfkill.sh
Restart=always
RestartSec=10
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable --now mojolearn-selfkill.service
sleep 2; systemctl is-active mojolearn-selfkill.service
INST
}

delete_verify() {  # id -> 0 when GET is 404
    local i
    for i in 1 2 3 4 5 6; do do_call DELETE "droplets/$1"; case "$DO_CODE" in 2*|404|dry) break ;; esac; sleep 10; done
    [ "$DRY" = 1 ] && return 0
    for i in $(seq 1 30); do do_call GET "droplets/$1"; [ "$DO_CODE" = 404 ] && return 0; sleep 10; done
    return 1
}

cmd="${1:-}"; shift || true
case "$cmd" in
up)
    NAME="${1:-}"; [ -n "$NAME" ] || die "usage: $0 up <name> [minutes]"
    valid_name "$NAME"; MIN="${2:-$DEFAULT_MIN}"; valid_min "$MIN"
    L="$S/$NAME"
    [ ! -e "$L/state.env" ] || die "a state file exists for '$NAME' ($L/state.env); down it first"
    if [ "$DRY" = 1 ]; then : > "$TMPD/curlrc"; else load_token; fi
    n=$(live_legs); [ "$n" -lt "$MAX_LEGS" ] || die "$n CPU legs are live (max $MAX_LEGS)"
    if [ "$DRY" = 0 ]; then
        do_call GET "droplets?tag_name=$DO_TAG&per_page=200"
        [ "$DO_CODE" = 200 ] || die "droplet listing HTTP $DO_CODE"
        an=$(jq_py 'print(len(d["droplets"]))'); [ "${an:-99}" -lt "$MAX_LEGS" ] || die "$an droplets tagged $DO_TAG live in the account (max $MAX_LEGS)"
    fi
    DL=$(( $(now) + 900 + MIN * 60 ))   # 15 min of boot/bootstrap grace on top of the lease
    DN=$(dname "$NAME" "$DL")
    printf '{"name":"%s","region":"%s","size":"%s","image":"%s","ssh_keys":["%s"],"tags":["%s"]}\n' \
        "$DN" "$DO_REGION" "$DO_SIZE" "$DO_IMAGE" "$DO_SSH_KEY_FP" "$DO_TAG" > "$TMPD/create.json"
    if [ "$DRY" = 1 ]; then
        say "would arm the Mac dead-man for $(utc $((DL + 600))), then create. THE BILL STARTS HERE."
        do_call POST droplets "$TMPD/create.json"; say "would install the on-droplet self-destruct at $(utc "$DL")"; exit 0
    fi
    mkdir -p "$L"; chmod 700 "$L"
    set_file "$L/droplet_name" "$DN"; set_file "$L/mac_deadline" $(( DL + 600 ))
    arm_mac_deadman "$L"
    say "Mac dead-man armed pid=$(cat "$L/deadman.pid") at $(utc $((DL + 600)))"
    say "creating $DN ($DO_SIZE, $DO_REGION). THE BILL STARTS HERE."
    do_call POST droplets "$TMPD/create.json"
    ID=$(jq_py 'print(d["droplet"]["id"])')
    [ -n "$ID" ] || { rm -rf "$L"; die "create refused (HTTP $DO_CODE): $(head -c 300 "$TMPD/do.body")"; }
    set_file "$L/droplet_id" "$ID"
    printf 'DROPLET_ID=%q\nNAME=%q\nDNAME=%q\nIP=\nSIZE=%q\nREGION=%q\nCREATED=%q\n' "$ID" "$NAME" "$DN" "$DO_SIZE" "$DO_REGION" "$(utc "$(now)")" > "$L/state.env"
    IP=""
    for i in $(seq 1 40); do
        do_call GET "droplets/$ID"
        IP=$(jq_py 'print([n["ip_address"] for n in d["droplet"]["networks"]["v4"] if n["type"]=="public"][0])')
        [ -n "$IP" ] && [ "$(jq_py 'print(d["droplet"]["status"])')" = active ] && break; IP=""; sleep 10
    done
    fail() { say "FAIL: $*; deleting $ID now"; delete_verify "$ID" && rm -rf "$L"; die "$*"; }
    [ -n "$IP" ] || fail "droplet never became active"
    sed -i.bak "s/^IP=.*/IP=$IP/" "$L/state.env" && rm -f "$L/state.env.bak"
    for i in $(seq 1 30); do bx 30 true < /dev/null 2>/dev/null && break; sleep 10; done
    if mk_curlrc "$SK_TOKFILE" "$TMPD/sk_curlrc"; then say "on-droplet self-destruct uses the scoped token $SK_TOKFILE"
    else say "WARNING: no scoped token ($SK_TOKFILE); the on-droplet self-destruct gets the full token"; cp "$TMPD/curlrc" "$TMPD/sk_curlrc"; fi
    bx 60 "umask 077; mkdir -p $G && cat > $G/curlrc && printf '%s\n' $DL > $G/deadline" < "$TMPD/sk_curlrc" || fail "could not deliver the self-destruct token"
    selfkill_installer "$ID" | bx 120 "bash -s" | tail -n 1 | grep -qx active || fail "the on-droplet self-destruct is not active"
    say "LIVE: $DN id=$ID ip=$IP; on-droplet deadline $(utc "$DL"), Mac dead-man $(utc $((DL + 600)))"
    ;;
extend)
    NAME="${1:-}"; [ -n "$NAME" ] || die "usage: $0 extend <name> [minutes]"
    valid_name "$NAME"; MIN="${2:-$DEFAULT_MIN}"; valid_min "$MIN"
    L="$S/$NAME"; [ -f "$L/state.env" ] || die "no leg '$NAME' ($L/state.env)"
    . "$L/state.env"
    DL=$(( $(now) + MIN * 60 )); DN=$(dname "$NAME" "$DL")
    if [ "$DRY" = 1 ]; then
        printf '{"type":"rename","name":"%s"}\n' "$DN" > "$TMPD/rn.json"; do_call POST "droplets/$DROPLET_ID/actions" "$TMPD/rn.json"; exit 0
    fi
    load_token
    bx 60 "printf '%s\n' $DL > $G/deadline.tmp && mv $G/deadline.tmp $G/deadline && systemctl is-active mojolearn-selfkill.service" < /dev/null | grep -qx active \
        || die "could not move the on-droplet deadline (the old one stands)"
    set_file "$L/mac_deadline" $(( DL + 600 ))
    kill -0 "$(cat "$L/deadman.pid" 2>/dev/null)" 2>/dev/null || { arm_mac_deadman "$L"; say "re-armed the Mac dead-man"; }
    printf '{"type":"rename","name":"%s"}\n' "$DN" > "$TMPD/rn.json"; do_call POST "droplets/$DROPLET_ID/actions" "$TMPD/rn.json"
    case "$DO_CODE" in 2*) set_file "$L/droplet_name" "$DN"; sed -i.bak "s/^DNAME=.*/DNAME=$DN/" "$L/state.env"; rm -f "$L/state.env.bak" ;;
        *) say "rename HTTP $DO_CODE (deadlines moved anyway)" ;; esac
    say "extended $NAME: droplet deadline $(utc "$DL"), Mac $(utc $((DL + 600)))"
    ;;
status)
    found=0
    for d in "$S"/*/; do
        [ -f "${d}state.env" ] || continue; found=1; . "${d}state.env"
        md=$(cat "${d}mac_deadline" 2>/dev/null || echo 0); pid=$(cat "${d}deadman.pid" 2>/dev/null || echo -)
        alive=dead; kill -0 "$pid" 2>/dev/null && alive=alive
        lv=STALE; live_state "${d%/}" && lv=live
        echo "$(basename "$d") id=$DROPLET_ID ip=${IP:-?} name=$(cat "${d}droplet_name" 2>/dev/null) mac_deadline=$(utc "$md") deadman=$pid($alive) $lv"
    done
    [ $found = 1 ] || say "no CPU legs"
    ;;
down)
    NAME="${1:-}"; [ -n "$NAME" ] || die "usage: $0 down <name>"
    valid_name "$NAME"; L="$S/$NAME"; [ -f "$L/state.env" ] || die "no leg '$NAME' ($L/state.env)"
    . "$L/state.env"
    [ "$DRY" = 1 ] || load_token
    say "deleting $DROPLET_ID ($NAME)"
    delete_verify "$DROPLET_ID" || die "droplet $DROPLET_ID is NOT verified gone; state kept, dead-man still armed"
    [ "$DRY" = 1 ] && exit 0
    say "droplet $DROPLET_ID verified gone (GET 404)"
    kill "$(cat "$L/deadman.pid" 2>/dev/null)" 2>/dev/null || true
    mkdir -p "$S/released"; mv "$L" "$S/released/$NAME-$(date -u +%Y%m%dT%H%M%SZ)"
    rm -f "$S/released/$NAME-"*/curlrc
    ;;
sweep)
    DEL=0; [ "${1:-}" = --delete ] && DEL=1
    if [ "$DRY" = 1 ] && [ ! -f "$DO_TOKFILE" ]; then die "sweep needs a listing; no token ($DO_TOKFILE)"; fi
    load_token
    DRY_SAVE=$DRY; DRY=0; do_call GET "droplets?per_page=200"; DRY=$DRY_SAVE
    [ "$DO_CODE" = 200 ] || die "droplet listing HTTP $DO_CODE"
    tracked=""
    for d in "$S"/*/; do live_state "${d%/}" && tracked="$tracked $(cat "${d}droplet_id" 2>/dev/null)"; done
    steward=""; [ -f "$STEWARD_STATE" ] && steward=$(sed -n 's/^DROPLET_ID=//p' "$STEWARD_STATE" | tr -d "'\"")
    tracked="$tracked $steward"
    python3 - "$TMPD/do.body" "$tracked" "$DO_TAG" "$SWEEP_AGE_H" > "$TMPD/sweep.tsv" <<'PY'
import json, sys, time, calendar, fnmatch
d = json.load(open(sys.argv[1])); tracked = set(sys.argv[2].split()); tag = sys.argv[3]; age_h = float(sys.argv[4])
for x in d.get("droplets", []):
    i = str(x["id"]); name = x.get("name", ""); tags = x.get("tags") or []
    if i in tracked: continue
    age = (time.time() - calendar.timegm(time.strptime(x["created_at"], "%Y-%m-%dT%H:%M:%SZ"))) / 3600
    steward = "mojolearn-steward" in tags or name.startswith("mojolearn-steward")
    ours = (tag in tags or fnmatch.fnmatch(name, "mojolearn-*cpu*")) and not steward
    kind = "steward" if steward else ("ours" if ours else "other")
    print(i, name, x.get("size_slug", "?"), "%.1f" % age, kind, "yes" if ours and age > age_h else "no", sep="\t")
PY
    [ -s "$TMPD/sweep.tsv" ] || { say "sweep: every droplet is covered by a live state file"; exit 0; }
    while IFS=$'\t' read -r id name size age kind auto; do
        echo "UNTRACKED $id $name $size age=${age}h kind=$kind"
        if [ "$kind" = steward ]; then echo "  GPU steward droplet: never auto-deleted; use tools/do_amd_steward.sh down"; continue; fi
        echo "  delete: curl -K <curlrc> -X DELETE $DO_API/droplets/$id"
        if [ "$DEL" = 1 ] && [ "$auto" = yes ]; then
            if delete_verify "$id"; then say "sweep deleted $id $name (verified 404)"; else say "sweep: $id NOT verified gone"; fi
        fi
    done < "$TMPD/sweep.tsv"
    ;;
*) sed -n '2,20p' "$0"; [ -z "$cmd" ] || die "unknown command '$cmd'"; exit 2 ;;
esac
