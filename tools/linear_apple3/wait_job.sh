#!/bin/sh
# lane linear-apple3: wait for one steward request to leave PENDING, then
# print its verdict line and its stdout (read from the Mac that ran it).
#   sh tools/linear_apple3/wait_job.sh <request name> <mac> [out file]
set -u
cd "$(dirname "$0")/../.."
name=$1; mac=$2; out=${3:-}
while :; do
    line=$(python3 tools/apple_steward.py status 2>/dev/null | grep "$name" | head -n 1)
    case "$line" in PENDING*|"") sleep 90 ;; *) break ;; esac
done
echo "$line"
txt=$(tools/cloudmac.sh ssh "$mac" "cat ~/mojolearn-evidence/apple-steward/done/$name/speed.stdout; echo ---STDERR---; tail -n 40 ~/mojolearn-evidence/apple-steward/done/$name/speed.stderr" 2>&1)
if [ -n "$out" ]; then mkdir -p "$(dirname "$out")"; printf '%s\n' "$txt" > "$out"; fi
printf '%s\n' "$txt" | tail -n 150
