#!/bin/bash
# peek.sh <mac> <request>: progress lines of a working or done job
cd ~/mojolearn-wt/trees-apple3
tools/cloudmac.sh ssh $1 "for d in ~/mojolearn-evidence/apple-steward/working/$2* ~/mojolearn-evidence/apple-steward/done/$2*; do [ -d \$d ] && { echo \$d; wc -l \$d/speed.stdout; grep -E '^##### ARM|wall_s|FAILED|Traceback|Error' \$d/speed.stdout | tail -n 12 | cut -c1-200; }; done" 2>/dev/null
