#!/bin/bash
# waitany.sh <id>... : returns when any id is no longer PENDING (cap 570 s)
cd ~/mojolearn-wt/trees-apple3
end=$(( $(date +%s) + 570 ))
while [ $(date +%s) -lt $end ]; do
  st=$(python3 tools/apple_steward.py status 2>/dev/null)
  for id in "$@"; do
    line=$(echo "$st" | grep "$id")
    case "$line" in PASS*|FAIL*) echo "$line"; exit 0;; esac
  done
  sleep 30
done
echo "$st" | grep -E "$(IFS='|'; echo "$*")"
echo "still pending $(date -u +%H:%M)"
