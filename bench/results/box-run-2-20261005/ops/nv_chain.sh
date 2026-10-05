#!/bin/bash
# nv_chain.sh (box-run-2): wave prepare/quality/identity on the 4090, then the compile matrix once prepare is done.
SHA=$(git -C /root/mojolearn rev-parse HEAD); PY=/root/mojolearn/.pixi/envs/default/bin/python
bash /root/lq/br2-wave_launch.sh nvidia sm_89 prepare
(bash /root/lq/br2-wave_launch.sh nvidia sm_89 "quality identity") &
cd /root/lq/br2-cf && $PY cf_driver.py setup $SHA > setup.log 2>&1; echo $? > setup.rc
$PY cf_driver.py run $SHA /root/lq/br2-cf/cf_jobs.tsv 4 > run.log 2>&1; echo $? > run.rc
wait
