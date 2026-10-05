#!/bin/bash
# amd_up.sh (box-run-2, 2026-10-05; from identical-all/amd-box): rent ONE DigitalOcean gpu-mi325x1-256gb droplet for the lq amd queue.
# Order: Mac dead-man (by tag) -> create -> ssh -> on-box deadman (idle 60 min, cap 360 min) -> state file.
# Never prints the token. State: ~/mojolearn-evidence/box-bringup/amd_state.env
set -u
B=$HOME/mojolearn-evidence/box-run-2/amd; TOK=$HOME/.mojolearn_do_token; API=https://api.digitalocean.com/v2
NAME=mojolearn-br2-amd; TAG=mojolearn-br2-amd; SIZE=gpu-mi325x1-256gb; IMAGE=188571990
FP="df:f7:6b:0c:56:da:48:a5:6f:6d:ae:44:af:de:f3:0b"
SO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -o BatchMode=yes"
say() { echo "$(date -u +%FT%TZ) $*"; }
die() { say "FAIL: $*"; exit 1; }
umask 077; RC=$B/.do_curlrc
printf 'header = "Authorization: Bearer %s"\nsilent\nshow-error\n' "$(tr -d ' \t\r\n' < $TOK)" > $RC
call() { curl -K $RC --max-time 120 -o $B/do.body -w '%{http_code}' -X "$1" -H 'Content-Type: application/json' ${3:+--data-binary @$3} "$API/$2"; }
py() { python3 -c "import json,sys; d=json.load(open('$B/do.body')); $1" 2>/dev/null; }

[ ! -f $B/amd_state.env ] || die "amd_state.env exists; a box is already recorded"
code=$(call GET "droplets?per_page=200"); [ "$code" = 200 ] || die "listing HTTP $code"
[ "$(py 'print(len([x for x in d["droplets"] if str(x.get("size_slug","")).startswith("gpu-")]))')" = 0 ] || die "a GPU droplet already exists"

# 1. Mac dead-man: deletes every droplet tagged $TAG at the deadline unless $B/amd_mac_deadman.disarm exists
DEADLINE=$(( $(date +%s) + 400*60 )); echo $DEADLINE > $B/amd_mac_deadline
cat > $B/amd_mac_deadman.sh <<EOF
#!/bin/sh
while :; do
  [ -f $B/amd_mac_deadman.disarm ] && exit 0
  if [ \$(date +%s) -ge \$(cat $B/amd_mac_deadline) ]; then
    echo "\$(date -u +%FT%TZ) deadline: deleting tag $TAG \$(curl -K $RC --max-time 60 -o /dev/null -w '%{http_code}' -X DELETE '$API/droplets?tag_name=$TAG')" >> $B/amd_mac_deadman.log
    sleep 600
  fi
  sleep 60
done
EOF
rm -f $B/amd_mac_deadman.disarm
nohup sh -c 'trap "" HUP INT; exec sh "$0"' $B/amd_mac_deadman.sh > /dev/null 2>&1 < /dev/null &
echo $! > $B/amd_mac_deadman.pid; sleep 1; kill -0 $(cat $B/amd_mac_deadman.pid) || die "Mac dead-man did not start"
say "Mac dead-man armed pid=$(cat $B/amd_mac_deadman.pid) deadline=$(date -u -r $DEADLINE +%FT%TZ)"

# 2. create
ID=""
for REGION in nyc2 tor1; do
  printf '{"name":"%s","region":"%s","size":"%s","image":%s,"ssh_keys":["%s"],"tags":["%s"]}\n' $NAME $REGION $SIZE $IMAGE "$FP" $TAG > $B/create.json
  say "creating $NAME ($SIZE) in $REGION"
  code=$(call POST droplets $B/create.json); ID=$(py 'print(d["droplet"]["id"])')
  [ -n "$ID" ] && break
  say "create in $REGION refused HTTP $code: $(head -c 300 $B/do.body)"
done
if [ -z "$ID" ]; then
  sleep 10; code=$(call GET "droplets?tag_name=$TAG&per_page=200")
  [ "$(py 'print(len(d["droplets"]))')" = 0 ] || die "no id parsed but a tagged droplet exists; Mac dead-man stays armed"
  touch $B/amd_mac_deadman.disarm; die "nothing was created"
fi
echo $ID > $B/amd_droplet_id; say "droplet $ID created in $REGION"

# 3. ssh
IP=""; t0=$(date +%s)
while [ $(( $(date +%s) - t0 )) -lt 900 ]; do
  call GET "droplets/$ID" > /dev/null
  IP=$(py 'x=d["droplet"]; print(next((n["ip_address"] for n in x["networks"]["v4"] if n["type"]=="public"),"") if x["status"]=="active" else "")')
  if [ -n "$IP" ] && ssh $SO root@$IP 'echo SSH-OK' < /dev/null 2>/dev/null | grep -q SSH-OK; then break; fi
  IP=""; sleep 10
done
printf 'DROPLET_ID=%s\nIP=%s\nSIZE=%s\nREGION=%s\nCREATED=%s\n' "$ID" "$IP" $SIZE $REGION "$(date -u +%FT%TZ)" > $B/amd_state.env
[ -n "$IP" ] || die "no ssh after 900 s; Mac dead-man stays armed; droplet $ID must be deleted"
say "ssh up at $IP"

# 4. on-box deadman BEFORE anything else: token over stdin, script, armed flag, setsid nohup
tr -d ' \t\r\n' < $TOK | ssh $SO root@$IP 'umask 077; cat > /root/.do_token; test -s /root/.do_token' || die "token delivery failed"
ssh $SO root@$IP 'cat > /root/lq_deadman.sh; chmod 700 /root/lq_deadman.sh' < $B/lq_deadman.sh || die "deadman copy failed"
ssh $SO root@$IP "touch /root/deadman.armed; IDLE_MIN=90 CAP_MIN=480 setsid nohup bash /root/lq_deadman.sh $ID > /dev/null 2>&1 < /dev/null & sleep 3; pgrep -f 'lq_deadman.sh $ID' > /dev/null && echo DEADMAN-ALIVE; tail -n 1 /root/deadman.log" < /dev/null > $B/amd_deadman_arm.out 2>&1
grep -q DEADMAN-ALIVE $B/amd_deadman_arm.out || die "on-box deadman not alive; droplet $ID must be deleted"
say "on-box deadman alive: $(tail -n 1 $B/amd_deadman_arm.out)"
say "OK id=$ID ip=$IP"
