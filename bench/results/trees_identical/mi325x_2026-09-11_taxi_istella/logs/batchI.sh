#!/bin/sh
# Leg 1: the setup's Istella-S fetch stopped at 454,285,384 of 472,129,615
# bytes (ContentTooShortError); the tar is completed by a curl byte-range
# resume (logs/istella_resume.log). No Istella-S cache exists, so batchT's
# PHASE_I cells would fall back to the synthetic fixture: this watcher stops
# batchT the moment its taxi cells end (PHASE_T2_DONE), then extracts and
# decodes Istella-S onto the volume with nothing timed running, so the next
# leg starts from the decoded cache. The Istella-S cells move to that leg.
OUT=/root/trees_out
cd /root/mojolearn || exit 9
export GBM_BENCH_DATA=/mnt/mojolearn-data/gbm-bench
while [ ! -f $OUT/phase.PHASE_T2_DONE ]; do sleep 1; done
pkill -f "[s]h /root/batchT.sh"
pkill -f "[f]orest_speed_arm.py --lane .* --dataset istella"
echo "batchT stopped after PHASE_T2_DONE $(date -u +%T)" | tee -a $OUT/ab.txt
while [ ! -f $OUT/istella_tar.ok ]; do sleep 5; done
timeout -k 30 1500 python3 tools/speed_gbdt_arm.py --download istella > $OUT/logs/download_istella.decode.log 2>&1
rc=$?
echo "download_istella_decode=$rc $(date -u +%T)" | tee -a $OUT/ab.txt $OUT/setup.txt
[ "$rc" = 0 ] && : > $OUT/istella.ok
ls -la $GBM_BENCH_DATA/istella >> $OUT/logs/download_istella.decode.log 2>&1
sync
