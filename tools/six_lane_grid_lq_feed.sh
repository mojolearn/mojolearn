#!/bin/bash
# six_lane_grid_lq_feed.sh <nv|nv2|amd|amd2> <lines file> [max queue depth, default 4] [poll seconds, default 300]
# Feeds the `lq add` lines that tools/six_lane_grid_lq.py render wrote, in order, to one box through lq
# (orchestrator side; lq is the only way to a box). Before each line it waits while the box queue holds
# max-depth or more unfinished jobs (`lq status <box>`: nv/nv2 = queued+starting+running gpu-queue jobs, amd/amd2 = queue
# lines minus done-through). Progress is kept in <lines file>.fed-<box> (lines already added), so a rerun
# resumes after the last line lq accepted. A line lq refuses stops the loop (rc 1) without advancing.
# LQ=<path to lq> overrides ~/mojolearn-evidence/lq/lq.
set -u
box=${1:?usage: six_lane_grid_lq_feed.sh <nv|nv2|amd|amd2> <lines> [max depth] [poll s]}; lines=${2:?lines file}
max=${3:-4}; poll=${4:-300}; LQ=${LQ:-$HOME/mojolearn-evidence/lq/lq}
# nv2 (2026-10-08) is a second L40S: it takes the nvidia lines (`lq add nv ...`) and shares nv's tag ledger, so the two
# NVIDIA feeders never queue one tag twice (the ledger is claimed under a lock right before each lq add)
# amd2 (2026-10-10) is a Hot Aisle MI300X VM (gfx942, as the amd MI325X): it takes the amd lines (`lq add amd ...`) and
# shares amd's tag ledger the same way
case $box in nv|nv2) src=nv;; amd|amd2) src=amd;; *) echo "box must be nv, nv2, amd or amd2" >&2; exit 2;; esac
state=$lines.fed-$box; done_n=$(cat "$state" 2>/dev/null || echo 0); total=$(grep -c '^lq add ' "$lines")
depth() {
  local s; s=$("$LQ" status "$box" 2>/dev/null </dev/null | grep "^$box:")
  if [ "$box" = amd ] || [ "$box" = amd2 ]; then
    echo "$s" | awk '{n=$2; p=$NF} END {if (n ~ /^[0-9]+$/ && p ~ /^[0-9]+$/) print n-p; else print -1}'
  else
    echo "$s" | awk '{t=0; ok=0; for (i=2;i<NF;i++) if ($i ~ /^[0-9]+$/) {ok=1; if ($(i+1) ~ /^(queued|starting|running)$/) t+=$i} print (ok ? t : (NF<=1 ? 0 : -1))}'
  fi
}
# the ledger lock is a directory (macOS has no flock); a lock older than 60 s is a dead feeder's and is broken
lk() { local w=0; until mkdir "$ledger.lockd" 2>/dev/null; do w=$((w+1)); [ $w -ge 60 ] && rmdir "$ledger.lockd" 2>/dev/null; sleep 1; done; }
ulk() { rmdir "$ledger.lockd" 2>/dev/null; }
n=0
while IFS= read -r line; do
  case $line in "lq add $src "*) ;; *) continue;; esac
  [ $box = $src ] || line="lq add $box ${line#lq add $src }"
  n=$((n+1)); [ $n -le "$done_n" ] && continue
  # tag ledger (per box, shared by every feeder of that box): a line whose grid tag was already queued by another
  # feeder (e.g. an incumbent line fed early for the noise floor) is skipped, not queued twice
  ledger=$(dirname "$lines")/fed-tags-$src.txt; touch "$ledger"
  tag=$(printf '%s' "$line" | grep -o 'MOJOLEARN_GRID_TAG=[^ ]*' | head -1)
  fed() { grep -qxF -e "$1" -e "${1#MOJOLEARN_GRID_TAG=}" "$ledger"; }   # the ledger holds both spellings
  if [ -n "$tag" ] && fed "$tag"; then
    echo "$(date -u +%FT%TZ) line $n/$total already queued ($tag); skipping"; done_n=$n; echo "$done_n" > "$state"; continue
  fi
  while :; do
    d=$(depth)
    if [ "$d" -ge 0 ] 2>/dev/null && [ "$d" -lt "$max" ]; then break; fi
    echo "$(date -u +%FT%TZ) $box depth=${d} (max $max); line $n/$total waits ${poll}s"; sleep "$poll"
  done
  # claim the tag under the ledger lock (another feeder of this vendor may have queued it while this one waited)
  if [ -n "$tag" ]; then
    lk; if fed "$tag"; then claimed=no; else echo "$tag" >> "$ledger"; claimed=yes; fi; ulk
    if [ "$claimed" != yes ]; then echo "$(date -u +%FT%TZ) line $n/$total queued by another feeder ($tag); skipping"; done_n=$n; echo "$done_n" > "$state"; continue; fi
  fi
  # shellcheck disable=SC2086
  out=$("$LQ" ${line#lq } 2>&1 </dev/null); rc=$?
  unclaim() { [ -n "$tag" ] || return 0; lk; grep -vxF "$tag" "$ledger" > "$ledger.tmp"; mv "$ledger.tmp" "$ledger"; ulk; }   # stdin detached: lq ssh-es, and would otherwise eat the lines file
  echo "$(date -u +%FT%TZ) line $n/$total rc=$rc $(echo "$out" | tail -1 | cut -c1-160)"
  case "$out" in *queued*) ;; *) unclaim; echo "lq did not queue line $n; stopping (state $state = $done_n)" >&2; exit 1;; esac
  [ $rc -eq 0 ] || { echo "lq rc=$rc on line $n; stopping" >&2; exit 1; }
  done_n=$n; echo "$done_n" > "$state"
done < "$lines"
echo "fed $done_n/$total lines to $box"
