#!/bin/bash
# pull_and_release.sh <runner out dir> <slug> <dest dir>
#
# Runs on the operator's machine beside tools/gemm_remote_leg.sh. The runner
# fetches and terminates the moment the body returns, so the body holds the
# box (bounded, 420 s) until /root/par_pull_ok exists. This script pulls the
# slug directory over ssh every minute (partial results come home as each
# command finishes), and when the body has written BODY_DONE it pulls once
# more, CHECKS the pull, prints the file count, and only then releases the
# hold. A pull that fails the check releases nothing: the hold times out by
# itself and the runner's own fetch is the second copy.
#
# The API key is read from MOJOLEARN_RUNPOD_KEY_FILE into a 0600 curl config
# and never appears in an argv.
set -u
LEGOUT=$1; SLUG=$2; DEST=$3
mkdir -p "$DEST"
KEYFILE="${MOJOLEARN_RUNPOD_KEY_FILE:-$HOME/.mojolearn_runpod_key}"
CFG=$(mktemp); chmod 600 "$CFG"
printf 'header = "Authorization: Bearer %s"\n' "$(cat "$KEYFILE")" > "$CFG"
trap 'rm -f "$CFG"' EXIT

for _ in $(seq 1 120); do [ -s "$LEGOUT/pod_id.txt" ] && break; sleep 5; done
POD=$(tr -d ' \r\n' < "$LEGOUT/pod_id.txt" 2>/dev/null)
[ -n "$POD" ] || { echo "no pod id appeared in $LEGOUT"; exit 2; }
echo "pod=$POD"

target() {
    curl -s -K "$CFG" "https://rest.runpod.io/v1/pods/$POD" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    ip, port = d.get("publicIp") or "", (d.get("portMappings") or {}).get("22")
    if ip and port: print(port, ip)
except Exception:
    pass'
}
rssh() { ssh -o StrictHostKeyChecking=accept-new -o ConnectTimeout=25 -o BatchMode=yes -p "$PORT" "root@$IP" "$@"; }
pull() { rssh "cd /root/gemm_leg_out && [ -d '$SLUG' ] && tar czf - '$SLUG'" 2>/dev/null | ( cd "$DEST" && tar xzf - ) 2>/dev/null; }

check() {   # the pull must be non-empty and every finished JSON step must parse
    python3 - "$DEST/$SLUG" <<'PY'
import json, os, sys
d = sys.argv[1]
ok = True
files = [f for f in os.listdir(d)]
print("pulled files:", len(files))
st = os.path.join(d, "status.tsv")
if not os.path.exists(st) or os.path.getsize(st) == 0:
    print("CHECK FAILED: status.tsv missing or empty"); sys.exit(1)
for line in open(st):
    name, code, secs = line.rstrip("\n").split("\t")
    # par_self_test_json is a JSON document FOLLOWED by the SELF-TEST line, so it is
    # not parsed here (lease 2: parsing it failed this check on a complete pull)
    if not name.startswith(("par_quick", "par_all_", "par_lanes_")) or code == "NOT-STARTED":
        continue
    out = os.path.join(d, name + ".out")
    log = os.path.join(d, name + ".log")
    try:
        doc = json.load(open(out))
        print(f"  {name}: exit {code}, {secs}s, state={doc.get('state')}, lanes={len(doc.get('lanes', []))}, "
              f"parts={doc.get('compared')}, counts={doc.get('counts')}")
    except Exception as exc:
        size = os.path.getsize(log) if os.path.exists(log) else 0
        print(f"  {name}: exit {code}, {secs}s, NO JSON DOCUMENT ({type(exc).__name__}); log bytes={size}")
        if code not in ("124", "137") or size == 0:
            ok = False
sys.exit(0 if ok else 1)
PY
}

DEADLINE=$(( $(date +%s) + 4200 ))
IP=""; PORT=""
while [ "$(date +%s)" -lt "$DEADLINE" ]; do
    if [ -z "$IP" ]; then read -r PORT IP < <(target) || true; fi
    if [ -n "${IP:-}" ]; then
        pull
        if [ -f "$DEST/$SLUG/status.tsv" ]; then
            echo "[$(date +%T)] $(tr '\t' ' ' < "$DEST/$SLUG/status.tsv" | tail -3 | tr '\n' ';')"
        else
            echo "[$(date +%T)] reachable, body not started yet"
        fi
        if [ -f "$DEST/$SLUG/BODY_DONE" ]; then
            echo "BODY_DONE seen; final pull and check"
            pull
            echo "file count: $(find "$DEST/$SLUG" -type f | wc -l)"
            if check; then
                rssh 'touch /root/par_pull_ok' && echo "PULL VERIFIED; hold released"
                exit 0
            fi
            echo "PULL CHECK FAILED; hold NOT released (it times out by itself; the runner fetch is the second copy)"
            exit 1
        fi
    else
        echo "[$(date +%T)] no public ip/port yet"
    fi
    # the runner finishing (teardown.txt) means the pod is gone
    [ -f "$LEGOUT/teardown.txt" ] && { echo "runner tore down before BODY_DONE was seen"; exit 3; }
    sleep 60
done
echo "gave up waiting"; exit 4
