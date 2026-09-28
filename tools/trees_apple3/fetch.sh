#!/bin/bash
# usage: fetch.sh <mac> <request>  copies speed.stdout into runs/
cd ~/mojolearn-wt/trees-apple3
tools/cloudmac.sh ssh $1 "cat ~/mojolearn-evidence/apple-steward/done/$2/speed.stdout" > ~/mojolearn-evidence/trees-apple3/runs/$2.stdout 2>/dev/null
wc -l ~/mojolearn-evidence/trees-apple3/runs/$2.stdout
