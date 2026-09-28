#!/bin/bash
# tools/do_amd_steward.sh -- THE SHARED AMD STEWARD on one DigitalOcean GPU
# droplet ("treat AMD like Apple", 2026-09-27). Lanes that hold no AMD box of
# their own submit AMD identity checks to it with tools/apple_steward.py
# (steward `do-amd`), exactly as they submit to the cloud Macs; it works its
# queue one request at a time on its one GPU (tools/algos_lane_check.sh,
# CPU == AMD, the sabotage DISAGREE, reversed AGREE).
#
#   tools/do_amd_steward.sh up [minutes]    RENTS the droplet (one GPU droplet per
#                                           account: takes /tmp/mojolearn-do-gpu.lock and
#                                           holds it for the droplet's life), bootstraps
#                                           git, pixi, the repo at origin/main, and starts
#                                           the steward as a systemd service
#   tools/do_amd_steward.sh extend [minutes]   the heartbeat: moves BOTH deadlines (on-droplet
#                                           self-destruct and Mac dead-man) to now + minutes
#   tools/do_amd_steward.sh update [sha]    the droplet's tree to origin/main (or a pushed sha), steward restarted
#                                           (drains: claims nothing new, waits for the running work; the queue never moves)
#   tools/do_amd_steward.sh ssh <command>   one command on the droplet, as root
#   tools/do_amd_steward.sh status          droplet, deadlines, service, queue
#   tools/do_amd_steward.sh down            DELETE, verify 404, disarm, release the lock
#
# THE LEASE. Both dead-men read a DEADLINE FILE instead of sleeping a fixed
# time, so `extend` is one atomic write on each side and two extends can
# never leave a stale timer behind. On the droplet the self-destruct is a
# systemd service (Restart=always, so it survives a reboot) that DELETEs its
# own droplet when the deadline passes or its deadline file is missing or
# unreadable (fail safe). On the Mac a nohup'd loop DELETEs the droplet by
# id and by name when the Mac deadline (droplet deadline + 10 min) passes.
# The allocator (~/mojolearn-evidence/devpods/amd_allocator.sh) calls
# `extend 120` hourly; a steward nobody renews ends by itself.
#
# SIZE. gpu-mi300x1-192gb when DigitalOcean has it in a region, else
# gpu-mi325x1-256gb (also gfx942, CDNA3: the same codegen target). Override:
# MOJOLEARN_STEWARD_DO_SIZES (comma list, in order), _DO_REGIONS.
# On AMD compare NUMBERS, never .so digests: gfx942 codegen from a cold Mojo
# cache varies run to run (tools/dev_pod.sh header).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
S="${MOJOLEARN_STEWARD_DO_STATE:-$HOME/mojolearn-evidence/do-amd-steward}"
DO_API=https://api.digitalocean.com/v2
DO_TOKFILE="${MOJOLEARN_DO_TOKEN_FILE:-$HOME/.mojolearn_do_token}"
DO_GPU_LOCK="${MOJOLEARN_DO_GPU_LOCK:-/tmp/mojolearn-do-gpu.lock}"
DO_SIZES="${MOJOLEARN_STEWARD_DO_SIZES:-gpu-mi300x1-192gb,gpu-mi325x1-256gb}"
DO_REGIONS="${MOJOLEARN_STEWARD_DO_REGIONS:-}"      # empty: any region the size lists
DO_IMAGE="${MOJOLEARN_STEWARD_DO_IMAGE:-188571990}"  # ROCm 6.4.1 on Ubuntu 24.04
DO_SSH_KEY_FP="df:f7:6b:0c:56:da:48:a5:6f:6d:ae:44:af:de:f3:0b"   # ~/.ssh/id_ed25519
DO_TAG=mojolearn-steward
NAME=mojolearn-steward-do-amd
REPO_URL="${MOJOLEARN_DEVPOD_REPO_URL:-https://github.com/mojolearn/mojolearn.git}"
G=/root/.mojolearn-steward-guard
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o BatchMode=yes -i $HOME/.ssh/id_ed25519 -o IdentitiesOnly=yes"

die() { echo "do-amd-steward: $*" >&2; exit 1; }
say() { echo "do-amd-steward: $*"; }
now() { date +%s; }
TMPD="$(mktemp -d "${TMPDIR:-/tmp}/do-steward.XXXXXX")"
trap 'rm -rf "$TMPD"' EXIT

load_token() {
    [ -f "$DO_TOKFILE" ] || die "no DigitalOcean token ($DO_TOKFILE)"
    _t=""; IFS= read -r _t < "$DO_TOKFILE" || [ -n "$_t" ] || die "empty token file"
    _t="${_t//[$'\t\r\n ']/}"
    ( umask 077; printf 'header = "Authorization: Bearer %s"\nsilent\nshow-error\n' "$_t" > "$TMPD/curlrc" )
    _t=""
}
do_call() {  # METHOD PATH [json]; DO_CODE, body in $TMPD/do.body
    if [ -n "${3:-}" ]; then
        DO_CODE=$(curl -K "$TMPD/curlrc" --max-time 120 -o "$TMPD/do.body" -w '%{http_code}' -X "$1" \
            -H 'Content-Type: application/json' --data-binary "@$3" "$DO_API/$2") || DO_CODE=000
    else
        DO_CODE=$(curl -K "$TMPD/curlrc" --max-time 120 -o "$TMPD/do.body" -w '%{http_code}' -X "$1" "$DO_API/$2") || DO_CODE=000
    fi
}
jq_py() { python3 -c "import json,sys; d=json.load(open('$TMPD/do.body')); $1" 2>/dev/null || true; }
bx() {  # seconds command; as root on the droplet
    # shellcheck disable=SC2086
    perl -e 'alarm shift; exec @ARGV' "$1" ssh $SSH_OPTS "root@$IP" "$2"
}
load_state() { [ -f "$S/state.env" ] || die "no steward droplet ($S/state.env); run: $0 up"; . "$S/state.env"; }

arm_mac_deadman() {  # writes the Mac deadman once; it reads $S/mac_deadline
    ( umask 077; mkdir -p "$S/deadman"; cp "$TMPD/curlrc" "$S/deadman/curlrc" )
    cat > "$S/deadman/deadman.sh" <<DM
#!/bin/sh
# tools/do_amd_steward.sh's Mac dead-man: DELETEs the steward droplet (by id
# and by name under tag $DO_TAG) once $S/mac_deadline passes.
set -u
trap '' HUP INT
while :; do
    d=\$(cat '$S/mac_deadline' 2>/dev/null); case "\$d" in ''|*[!0-9]*) d=0 ;; esac
    [ "\$(date +%s)" -ge "\$d" ] && break
    [ -d '$S' ] || exit 0
    sleep 30
done
L='$S/deadman/deadman.log'
echo "\$(date -u +%FT%TZ) dead-man firing" >> "\$L"
ids="\$(cat '$S/droplet_id' 2>/dev/null)"
curl -K '$S/deadman/curlrc' --max-time 60 -o '$S/deadman/list.json' '$DO_API/droplets?tag_name=$DO_TAG&per_page=200' >> "\$L" 2>&1
ids="\$ids \$(python3 -c 'import json,sys; print(" ".join(str(x["id"]) for x in json.load(open(sys.argv[1])).get("droplets", []) if x.get("name") == sys.argv[2]))' '$S/deadman/list.json' '$NAME' 2>/dev/null)"
for id in \$ids; do
    c=\$(curl -K '$S/deadman/curlrc' --max-time 60 -o /dev/null -w '%{http_code}' -X DELETE "$DO_API/droplets/\$id" 2>> "\$L")
    echo "\$(date -u +%FT%TZ) DELETE \$id -> \$c" >> "\$L"
done
DM
    sh -n "$S/deadman/deadman.sh" || die "the Mac dead-man did not compose"
    nohup sh -c 'trap "" HUP INT; exec sh "$0"' "$S/deadman/deadman.sh" > /dev/null 2>&1 < /dev/null &
    echo $! > "$S/deadman.pid"; sleep 1
    kill -0 "$(cat "$S/deadman.pid")" 2>/dev/null || die "the Mac dead-man did not start"
}
set_mac_deadline() { printf '%s\n' "$1" > "$S/mac_deadline.tmp" && mv "$S/mac_deadline.tmp" "$S/mac_deadline"; }

selfkill_unit() {  # droplet id -> the on-droplet script + unit, on stdout as a shell installer
    cat <<INST
set -e
umask 077; mkdir -p $G
cat > $G/selfkill.sh <<'SK'
#!/bin/sh
# tools/do_amd_steward.sh's ON-DROPLET self-destruct: DELETEs droplet $1 when
# $G/deadline passes, or when that file is missing or unreadable.
set -u
while :; do
    d=\$(cat $G/deadline 2>/dev/null); case "\$d" in ''|*[!0-9]*) d=0 ;; esac
    [ "\$(date +%s)" -ge "\$d" ] && break
    sleep 20
done
n=1
while [ \$n -le 40 ]; do
    c=\$(curl -K $G/curlrc --max-time 60 -o $G/selfkill.body -w '%{http_code}' -X DELETE '$DO_API/droplets/$1')
    echo "\$(date -u +%FT%TZ) DELETE $1 attempt \$n -> \$c" >> $G/selfkill.out
    case "\$c" in 2*|404) sleep 3600 ;; esac
    sleep 15; n=\$((n + 1))
done
SK
chmod 700 $G/selfkill.sh
cat > /etc/systemd/system/mojolearn-selfkill.service <<'UNIT'
[Unit]
Description=mojolearn steward self-destruct (DELETE this droplet at $G/deadline)
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
set_box_deadline() {  # epoch
    bx 60 "printf '%s\n' $1 > $G/deadline.tmp && mv $G/deadline.tmp $G/deadline && cat $G/deadline && systemctl is-active mojolearn-selfkill.service" < /dev/null
}

steward_unit() {
    cat <<'INST'
set -e
cat > /etc/systemd/system/mojolearn-steward.service <<'UNIT'
[Unit]
Description=mojolearn AMD steward (tools/apple_steward.py work --steward do-amd)
After=network-online.target
[Service]
WorkingDirectory=/root/mojolearn
Environment=HOME=/root
Environment=PATH=/root/.pixi/bin:/opt/rocm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
Environment=MOJOLEARN_STEWARD_REPO=/root/mojolearn
Environment=PYTHONUNBUFFERED=1
ExecStart=/usr/bin/python3 /root/mojolearn/tools/apple_steward.py work --steward do-amd
Restart=always
RestartSec=30
StandardOutput=append:/root/mojolearn-steward.log
StandardError=append:/root/mojolearn-steward.log
[Install]
WantedBy=multi-user.target
UNIT
systemctl daemon-reload
systemctl enable mojolearn-steward.service
systemctl restart mojolearn-steward.service
sleep 3; systemctl is-active mojolearn-steward.service
INST
}
tree_to() {  # sha: /root/mojolearn (a shallow git tree) checked out at sha
    bx 1800 "set -e; mkdir -p /root/mojolearn && cd /root/mojolearn
[ -d .git ] || { git init -q . && git remote add origin $REPO_URL; }
git config --global --add safe.directory '*' 2>/dev/null || true
git fetch -q --depth=1 origin $1
git checkout -q --detach -f $1
echo HEAD=\$(git rev-parse HEAD)" < /dev/null | tee "$TMPD/tree.out"
    grep -qx "HEAD=$1" "$TMPD/tree.out" || die "the droplet's tree is not at $1"
}

cmd="${1:-}"; shift || true
case "$cmd" in
up)
    minutes="${1:-240}"
    [ ! -f "$S/state.env" ] || die "a steward droplet exists ($S/state.env); down it first"
    mkdir -p "$S"; load_token
    mkdir "$DO_GPU_LOCK" 2>/dev/null || die "$DO_GPU_LOCK is held (one GPU droplet per account): $(tr '\n' ' ' < "$DO_GPU_LOCK/owner" 2>/dev/null)"
    { echo "lane=do-amd-steward"; echo "script=tools/do_amd_steward.sh"; echo "pid=$$"; echo "utc=$(date -u +%FT%TZ)"; } > "$DO_GPU_LOCK/owner"
    do_call GET "droplets?per_page=200"
    [ "$DO_CODE" = 200 ] || die "droplet listing HTTP $DO_CODE"
    [ "$(jq_py 'print("yes" if any(str(x.get("size_slug","")).startswith("gpu-") for x in d["droplets"]) else "no")')" = no ] \
        || die "a GPU droplet already runs on the account (one allowed)"
    do_call GET "sizes?per_page=200"
    pick=$(jq_py "
want=[x for x in '$DO_SIZES'.split(',') if x]; regs=[x for x in '$DO_REGIONS'.split(',') if x]
sz={s['slug']:s for s in d['sizes']}
for w in want:
    s=sz.get(w)
    if not s or not s.get('available'): continue
    r=[x for x in s.get('regions') or [] if not regs or x in regs]
    if r: print(w, ','.join(r), s['price_hourly']); break")
    [ -n "$pick" ] || { rm -rf "$DO_GPU_LOCK"; die "none of $DO_SIZES is in stock in a region"; }
    read -r SIZE REGS PRICE <<< "$pick"
    say "size $SIZE at \$$PRICE/h, regions $REGS"
    set_mac_deadline $(( $(now) + 1800 + minutes * 60 + 600 ))
    arm_mac_deadman
    DROPLET_ID=""
    for REGION in ${REGS//,/ }; do
        printf '{"name":"%s","region":"%s","size":"%s","image":%s,"ssh_keys":["%s"],"tags":["%s"]}\n' \
            "$NAME" "$REGION" "$SIZE" "$DO_IMAGE" "$DO_SSH_KEY_FP" "$DO_TAG" > "$TMPD/create.json"
        say "creating $NAME ($SIZE) in $REGION. THE BILL STARTS HERE."
        do_call POST droplets "$TMPD/create.json"
        DROPLET_ID=$(jq_py 'print(d["droplet"]["id"])')
        [ -n "$DROPLET_ID" ] && break
        say "create in $REGION refused (HTTP $DO_CODE): $(head -c 300 "$TMPD/do.body")"
    done
    if [ -z "$DROPLET_ID" ]; then
        sleep 10; do_call GET "droplets?tag_name=$DO_TAG&per_page=200"
        [ -z "$(jq_py 'print(" ".join(str(x["id"]) for x in d["droplets"]))')" ] || die "no id parsed but a tagged droplet exists; the Mac dead-man stays armed"
        kill "$(cat "$S/deadman.pid")" 2>/dev/null || true; rm -rf "$S" "$DO_GPU_LOCK"
        die "nothing was created"
    fi
    echo "$DROPLET_ID" > "$S/droplet_id"
    IP=""; t0=$(now)
    while [ $(( $(now) - t0 )) -lt 900 ]; do
        do_call GET "droplets/$DROPLET_ID"
        IP=$(jq_py 'x=d["droplet"]; print(next((n["ip_address"] for n in x["networks"]["v4"] if n["type"]=="public"),"") if x["status"]=="active" else "")')
        if [ -n "$IP" ] && bx 30 'echo SSH-OK' < /dev/null 2>/dev/null | grep -q SSH-OK; then break; fi
        IP=""; sleep 10
    done
    printf 'DROPLET_ID=%q\nIP=%q\nSIZE=%q\nREGION=%q\nPRICE=%q\nCREATED=%q\n' "$DROPLET_ID" "$IP" "$SIZE" "$REGION" "$PRICE" "$(date -u +%FT%TZ)" > "$S/state.env"
    [ -n "$IP" ] || die "no ssh after 900s; the Mac dead-man stays armed (run: $0 down)"
    say "droplet $DROPLET_ID at $IP"
    # the self-destruct BEFORE any work
    bx 60 "umask 077; mkdir -p $G && cat > $G/curlrc" < "$TMPD/curlrc" || die "could not deliver the token"
    bx 60 "printf '%s\n' $(( $(now) + minutes * 60 )) > $G/deadline" < /dev/null
    selfkill_unit "$DROPLET_ID" > "$TMPD/sk.sh"
    bx 120 'bash -s' < "$TMPD/sk.sh" | tee "$TMPD/sk.out"
    grep -qx active "$TMPD/sk.out" || die "the on-droplet self-destruct is not active; run: $0 down"
    set_mac_deadline $(( $(now) + minutes * 60 + 600 ))
    say "self-destruct active on the droplet; lease ${minutes} min"
    bx 900 'export DEBIAN_FRONTEND=noninteractive; for i in $(seq 1 60); do fuser /var/lib/dpkg/lock-frontend > /dev/null 2>&1 || break; sleep 5; done
command -v git > /dev/null || { apt-get -o DPkg::Lock::Timeout=300 update -qq && apt-get -o DPkg::Lock::Timeout=300 install -y -qq git; } > /dev/null
[ -x /root/.pixi/bin/pixi ] || curl -fsSL https://pixi.sh/install.sh | bash > /dev/null 2>&1
/root/.pixi/bin/pixi --version; rocminfo | grep -m1 -o "gfx[0-9a-f]*"' < /dev/null | tee "$TMPD/boot.out"
    grep -q '^gfx942' "$TMPD/boot.out" || die "rocminfo does not read gfx942"
    tree_to "$(git -C "$ROOT" rev-parse origin/main)"
    steward_unit > "$TMPD/st.sh"; bx 120 'bash -s' < "$TMPD/st.sh" | tail -1 | grep -qx active || die "the steward service is not active"
    say "LIVE: steward do-amd on $IP ($SIZE, \$$PRICE/h); submit with tools/apple_steward.py submit (it targets do-amd while $S/state.env exists)"
    ;;
extend)
    load_state; minutes="${1:-120}"; load_token
    out=$(set_box_deadline $(( $(now) + minutes * 60 ))) || die "could not reach the droplet; its old deadline stands"
    printf '%s\n' "$out" | grep -qx active || die "the on-droplet self-destruct is not active: $out"
    set_mac_deadline $(( $(now) + minutes * 60 + 600 ))
    if ! kill -0 "$(cat "$S/deadman.pid" 2>/dev/null || echo 0)" 2>/dev/null; then arm_mac_deadman; say "Mac dead-man re-armed"; fi
    say "extended droplet $DROPLET_ID by ${minutes} min (on-droplet self-destruct and Mac dead-man)"
    ;;
update)
    # Never restart under a running request (a killed check leaves its
    # request stranded in working/ and possibly a sabotage applied), and never
    # move the queue: queued requests that left queue/ vanished from `status`
    # and from coalescing, lanes resubmitted them, and they came back as
    # duplicates (a hold loop left running on the box by an interrupted update
    # kept hiding every new submission for hours). The steward DRAINS instead:
    # $Qd/drain makes it claim nothing new, and it writes $Qd/drained once
    # nothing of its own runs; the wait is a laptop-side poll, and an
    # interrupted update removes the drain file, so nothing is left behind.
    load_state
    Qd=/root/mojolearn-evidence/apple-steward
    sha="$(git -C "$ROOT" rev-parse "${1:-origin/main}^{commit}")"
    # an earlier update's leftovers: its on-box hold loop, and requests it held
    # (the pattern and the path are spelled so this command's own line matches neither)
    bx 60 "pkill -f 'queue/hel[d]/' || true; H=$Qd/queue/hel
for f in \${H}d/[0-9]*.json; do [ -f \"\$f\" ] && mv \"\$f\" $Qd/queue/ && echo \"released \$f\"; done
rmdir \${H}d 2>/dev/null; true" < /dev/null
    if bx 60 "grep -q DRAIN_FILE /root/mojolearn/tools/apple_steward.py" < /dev/null; then
        trap 'bx 60 "rm -f $Qd/drain" < /dev/null || true; rm -rf "$TMPD"' EXIT
        bx 60 "rm -f $Qd/drained; touch $Qd/drain" < /dev/null
        say "draining (the queue stays visible); waiting for the running work to finish"
        until bx 60 "test -f $Qd/drained" < /dev/null 2> /dev/null; do sleep 20; done
    else
        # a steward from before the drain file: STOP it between requests. It
        # claims only from queue/, so the laptop polls for a moment with no
        # request in working/ and stops the service then; the queue never moves.
        say "the running steward predates the drain file: stopping it the first moment nothing runs"
        # A request claimed in the instant before the stop goes back to queue/
        # under its own name (it had only begun its checkout).
        until bx 60 "ls $Qd/working/[0-9]*.json > /dev/null 2>&1 || { systemctl stop mojolearn-steward.service
  for f in $Qd/working/[0-9]*.json; do [ -f \"\$f\" ] || continue; b=\$(basename \"\$f\"); mv \"\$f\" $Qd/queue/\${b%%.*}.json; echo \"requeued \$b\"; done
  echo STOPPED; }" < /dev/null 2> /dev/null | tee /dev/stderr | grep -qx STOPPED; do sleep 5; done
    fi
    tree_to "$sha"
    bx 60 "systemctl restart mojolearn-steward.service; rm -f $Qd/drain; sleep 3; systemctl is-active mojolearn-steward.service" < /dev/null
    trap 'rm -rf "$TMPD"' EXIT
    ;;
ssh)
    load_state; [ $# -gt 0 ] || die "ssh needs a command"
    # shellcheck disable=SC2086
    ssh $SSH_OPTS "root@$IP" "export PATH=/root/.pixi/bin:/opt/rocm/bin:\$PATH; $*"
    ;;
status)
    load_state
    echo "droplet $DROPLET_ID $IP $SIZE \$$PRICE/h created $CREATED"
    echo "mac deadline in $(( $(cat "$S/mac_deadline") - $(now) ))s; Mac dead-man pid $(cat "$S/deadman.pid") $(kill -0 "$(cat "$S/deadman.pid")" 2>/dev/null && echo alive || echo DEAD)"
    bx 60 "echo box deadline in \$(( \$(cat $G/deadline) - \$(date +%s) ))s; systemctl is-active mojolearn-selfkill.service mojolearn-steward.service; cd ~/mojolearn && echo \"tree at \$(git log -1 --format=%h)\"; [ -f ~/mojolearn-evidence/apple-steward/drain ] && echo DRAINING; ls ~/mojolearn-evidence/apple-steward/queue ~/mojolearn-evidence/apple-steward/working 2>/dev/null; tail -5 /root/mojolearn-steward.log" < /dev/null
    ;;
down)
    load_state; load_token
    for i in 1 2 3 4; do
        do_call DELETE "droplets/$DROPLET_ID"; say "DELETE $DROPLET_ID -> HTTP $DO_CODE"
        case "$DO_CODE" in 2*|404) break ;; esac; sleep 10
    done
    gone=0
    for i in $(seq 1 30); do do_call GET "droplets/$DROPLET_ID"; [ "$DO_CODE" = 404 ] && { gone=1; break; }; sleep 10; done
    [ "$gone" = 1 ] || die "droplet $DROPLET_ID NOT CONFIRMED GONE (GET $DO_CODE); the dead-men stay armed"
    _p=$(cat "$S/deadman.pid" 2>/dev/null || true); [ -z "$_p" ] || kill "$_p" 2>/dev/null || true
    grep -q 'do-amd-steward' "$DO_GPU_LOCK/owner" 2>/dev/null && rm -rf "$DO_GPU_LOCK"
    mv "$S" "$S.down-$(date -u +%Y%m%dT%H%M%SZ)"
    say "droplet $DROPLET_ID verified gone (GET 404)"
    ;;
*) sed -n 2,20p "$0"; exit 2 ;;
esac
