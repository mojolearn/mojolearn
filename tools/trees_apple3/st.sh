#!/bin/bash
# st.sh <id>...: status lines of the ids plus the queue table
cd ~/mojolearn-wt/trees-apple3
st=$(python3 tools/apple_steward.py status 2>/dev/null)
for id in "$@"; do echo "$st" | grep "^[A-Z]* *$id" | cut -c1-260; done
echo "$st" | sed -n '/^steward/,$p'
date -u +%H:%MZ
