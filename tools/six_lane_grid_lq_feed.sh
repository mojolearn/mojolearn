#!/bin/bash
# six_lane_grid_lq_feed.sh <nv|amd> <lines file> [max queue depth, default 4] [poll seconds, default 300]
# Feeds the `lq add` lines that tools/six_lane_grid_lq.py render wrote, in order, to one box through lq
# (orchestrator side; lq is the only way to a box). Before each line it waits while the box queue holds
# max-depth or more unfinished jobs (`lq status`: nv = queued+starting+running gpu-queue jobs, amd = queue
# lines minus done-through). Progress is kept in <lines file>.fed-<box> (lines already added), so a rerun
# resumes after the last line lq accepted. A line lq refuses stops the loop (rc 1) without advancing.
# LQ=<path to lq> overrides ~/mojolearn-evidence/lq/lq.
set -u
box=${1:?usage: six_lane_grid_lq_feed.sh <nv|amd> <lines> [max depth] [poll s]}; lines=${2:?lines file}
max=${3:-4}; poll=${4:-300}; LQ=${LQ:-$HOME/mojolearn-evidence/lq/lq}
case $box in nv|amd) ;; *) echo "box must be nv or amd" >&2; exit 2;; esac
state=$lines.fed-$box; done_n=$(cat "$state" 2>/dev/null || echo 0); total=$(grep -c '^lq add ' "$lines")
depth() {
  local s; s=$("$LQ" status 2>/dev/null </dev/null | grep "^$box:")
  if [ "$box" = amd ]; then
    echo "$s" | awk '{n=$2; p=$NF} END {if (n ~ /^[0-9]+$/ && p ~ /^[0-9]+$/) print n-p; else print -1}'
  else
    echo "$s" | awk '{t=0; ok=0; for (i=2;i<NF;i++) if ($i ~ /^[0-9]+$/) {ok=1; if ($(i+1) ~ /^(queued|starting|running)$/) t+=$i} print (ok ? t : (NF<=1 ? 0 : -1))}'
  fi
}
n=0
while IFS= read -r line; do
  case $line in "lq add $box "*) ;; *) continue;; esac
  n=$((n+1)); [ $n -le "$done_n" ] && continue
  # tag ledger (per box, shared by every feeder of that box): a line whose grid tag was already queued by another
  # feeder (e.g. an incumbent line fed early for the noise floor) is skipped, not queued twice
  ledger=$(dirname "$lines")/fed-tags-$box.txt; touch "$ledger"
  tag=$(printf '%s' "$line" | grep -o 'MOJOLEARN_GRID_TAG=[^ ]*' | head -1)
  if [ -n "$tag" ] && grep -qxF "$tag" "$ledger"; then
    echo "$(date -u +%FT%TZ) line $n/$total already queued ($tag); skipping"; done_n=$n; echo "$done_n" > "$state"; continue
  fi
  while :; do
    d=$(depth)
    if [ "$d" -ge 0 ] 2>/dev/null && [ "$d" -lt "$max" ]; then break; fi
    echo "$(date -u +%FT%TZ) $box depth=${d} (max $max); line $n/$total waits ${poll}s"; sleep "$poll"
  done
  # shellcheck disable=SC2086
  out=$("$LQ" ${line#lq } 2>&1 </dev/null); rc=$?   # stdin detached: lq ssh-es, and would otherwise eat the lines file
  echo "$(date -u +%FT%TZ) line $n/$total rc=$rc $(echo "$out" | tail -1 | cut -c1-160)"
  case "$out" in *queued*) ;; *) echo "lq did not queue line $n; stopping (state $state = $done_n)" >&2; exit 1;; esac
  [ $rc -eq 0 ] || { echo "lq rc=$rc on line $n; stopping" >&2; exit 1; }
  [ -n "$tag" ] && echo "$tag" >> "$ledger"
  done_n=$n; echo "$done_n" > "$state"
done < "$lines"
echo "fed $done_n/$total lines to $box"
