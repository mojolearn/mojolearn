#!/bin/bash
# cf_run.sh <sha> <jobs.tsv>: compile matrix at <sha> (box-run-2). Output /root/lq/br2-compile-<sha12>/.
SHA=$1; J=$2; PY=/root/mojolearn/.pixi/envs/default/bin/python; cd /root/lq/br2-cf
git -C /root/mojolearn fetch -q origin lane/box-run-2
$PY cf_driver.py setup $SHA > setup-${SHA:0:9}.log 2>&1; echo $? > setup-${SHA:0:9}.rc
$PY cf_driver.py run $SHA $J 4 > run-${SHA:0:9}.log 2>&1; echo $? > run-${SHA:0:9}.rc
