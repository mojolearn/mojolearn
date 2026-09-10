#!/bin/bash
set -euo pipefail
export PATH=/root/.pixi/bin:$PATH
cd /root/knnbatch
while [ ! -f /root/jobs/mamba-stage-retry.rc ]; do sleep 3; done
if [ "${KNN_BATCH_ACTIVE:-0}" != 1 ]; then
 export KNN_BATCH_ACTIVE=1
 exec pixi run bash /root/jobs/knn-batch.sh
fi
bash tools/knn_query_batch_probe.sh /root/evidence/knn-batch
