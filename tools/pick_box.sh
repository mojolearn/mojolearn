#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
#
# pick_box.sh: which rented box can take a leg RIGHT NOW, in the order
# ENGINEERING_RULES 10 sets (Andrew, 2026-09-11: "use hot aisle first if we
# can, digital ocean second if we can, runpod 3rd.. DO NOT WAIT").
#
#   tools/pick_box.sh [--need amd|any] [--spec 13core|8core] [--gpu "NVIDIA H100 80GB HBM3"]
#
# Prints exactly one word on stdout and the reasons on stderr:
#   hotaisle      a Hot Aisle slot is free, the spec is in stock, balance >= $5
#                 -> tools/hotaisle_leg.sh amd --rent
#   do            the DigitalOcean GPU lock is free and no GPU droplet is live
#                 -> tools/do_extra_leg.sh amd (it takes the lock itself)
#   runpod-amd    RunPod has AMD MI300X stock and fewer than 3 non-samba pods
#   runpod-nvidia RunPod has stock of --gpu and fewer than 3 non-samba pods
#                 -> tools/trees_leg.sh / tools/gemm_remote_leg.sh (NVIDIA row)
#   none          nothing can take it now; exit 3
#
# --need amd (default): only AMD boxes qualify (Hot Aisle, DigitalOcean,
# RunPod AMD). A leg whose numbers must be AMD rows asks this. When it prints
# none, DO NOT WAIT: run the NVIDIA version of the work that is also owed
# (`--need any`), and come back for the AMD row when a box frees.
# --need any: RunPod NVIDIA also qualifies, last.
#
# Read-only: GETs only, creates nothing, takes no lock. The answer is a
# snapshot, so the runner's own slot, lock and stock checks still decide.
# Keys are read from their files into curl config on a pipe, never an argv.
set -u
NEED=amd; SPEC=13core; GPU="NVIDIA H100 80GB HBM3"
while [ $# -gt 0 ]; do
  case "$1" in
    --need) NEED=$2; shift 2 ;;
    --spec) SPEC=$2; shift 2 ;;
    --gpu) GPU=$2; shift 2 ;;
    -h|--help) sed -n '5,31p' "$0"; exit 0 ;;
    *) echo "unknown argument $1" >&2; exit 2 ;;
  esac
done
case "$NEED" in amd|any) ;; *) echo "--need must be amd or any" >&2; exit 2 ;; esac
case "$SPEC" in 13core) CORES=13 ;; 8core) CORES=8 ;; *) echo "--spec must be 13core or 8core" >&2; exit 2 ;; esac

HA_KEY=${MOJOLEARN_HOTAISLE_KEY_FILE:-$HOME/.mojolearn_hotaisle_key}
DO_KEY=${MOJOLEARN_DO_TOKEN_FILE:-$HOME/.mojolearn_do_token}
RP_KEY=${MOJOLEARN_RUNPOD_KEY_FILE:-$HOME/.mojolearn_runpod_key}
HA_TEAM=${MOJOLEARN_HOTAISLE_TEAM:-andrews-team}
say() { printf 'pick_box: %s\n' "$*" >&2; }

hotaisle_ok() {
  [ -s "$HA_KEY" ] || { say "hotaisle: no key file"; return 1; }
  ha() { curl -s -m 20 -K <(printf 'header = "Authorization: Token %s"\n' "$(tr -d '\n' < "$HA_KEY")") "https://admin.hotaisle.app/api$1"; }
  local verdict
  verdict=$( { ha "/teams/$HA_TEAM/"; printf '\n'; ha "/teams/$HA_TEAM/balance/"; printf '\n'; ha "/teams/$HA_TEAM/virtual_machines/available/"; printf '\n'; } | python3 -c "
import json,sys
lines=[l for l in sys.stdin.read().split('\n') if l.strip()]
try:
    team,bal,avail=(json.loads(l) for l in lines[:3])
except Exception as e:
    print('no api:', e); sys.exit()
cap=int(team.get('maximum_virtual_machines') or 0)
cents=int(bal.get('available_balance') or 0)
qty=sum(int(v.get('Quantity') or 0) for v in avail
        if v.get('Specs',{}).get('cpu_cores')==$CORES
        and sum(g.get('count',0) for g in v.get('Specs',{}).get('gpus',[]))==1)
print(cap, cents, qty)
")
  set -- $verdict
  case "${1:-}" in ''|no) say "hotaisle: $verdict"; return 1 ;; esac
  local cap=$1 cents=$2 qty=$3 used
  used=$(ls -d /tmp/mojolearn-hotaisle-slot.* 2>/dev/null | wc -l | tr -d ' ')
  say "hotaisle: slots $used/$cap, $SPEC stock $qty, balance \$$((cents / 100))"
  [ "$used" -lt "$cap" ] && [ "$qty" -gt 0 ] && [ "$cents" -ge 500 ]
}

do_ok() {
  [ -s "$DO_KEY" ] || { say "do: no token file"; return 1; }
  if [ -d /tmp/mojolearn-do-gpu.lock ]; then
    say "do: lock held by $(head -1 /tmp/mojolearn-do-gpu.lock/owner 2>/dev/null)"; return 1
  fi
  local n
  n=$(curl -s -m 20 -K <(printf 'header = "Authorization: Bearer %s"\n' "$(tr -d '\n' < "$DO_KEY")") \
        "https://api.digitalocean.com/v2/droplets?per_page=200" \
      | python3 -c "import json,sys; print(sum(1 for d in json.load(sys.stdin).get('droplets',[]) if d.get('size_slug','').startswith('gpu-')))" 2>/dev/null)
  say "do: lock free, live GPU droplets ${n:-unknown}"
  [ "${n:-1}" = 0 ]
}

runpod_pick() {  # prints runpod-amd or runpod-nvidia when one qualifies
  [ -s "$RP_KEY" ] || { say "runpod: no key file"; return 1; }
  local out
  out=$(curl -s -m 30 -H "Content-Type: application/json" \
        -K <(printf 'header = "Authorization: Bearer %s"\n' "$(tr -d '\n' < "$RP_KEY")") \
        https://api.runpod.io/graphql \
        -d '{"query":"query { myself { pods { name desiredStatus } } gpuTypes { id displayName lowestPrice(input:{gpuCount:1}) { stockStatus } } }"}' \
      | NEED="$NEED" GPU="$GPU" python3 -c "
import json,os,sys
d=json.load(sys.stdin)['data']
running=sum(1 for p in d['myself']['pods'] if p.get('desiredStatus')=='RUNNING' and not (p.get('name') or '').startswith('samba'))
stock={g['id']:((g.get('lowestPrice') or {}).get('stockStatus')) for g in d['gpuTypes']}
amd=stock.get('AMD Instinct MI300X OAM'); nv=stock.get(os.environ['GPU'])
print(running, amd, nv)
" 2>/dev/null)
  set -- $out
  local running=${1:-9} amd=${2:-None} nv=${3:-None}
  say "runpod: running non-samba pods $running, MI300X stock $amd, $GPU stock $nv"
  [ "$running" -lt 3 ] 2>/dev/null || return 1
  if [ "$amd" != None ]; then echo runpod-amd; return 0; fi
  if [ "$NEED" = any ] && [ "$nv" != None ]; then echo runpod-nvidia; return 0; fi
  return 1
}

if hotaisle_ok; then echo hotaisle; exit 0; fi
if do_ok; then echo "do"; exit 0; fi
if pick=$(runpod_pick); then echo "$pick"; exit 0; fi
echo none
exit 3
