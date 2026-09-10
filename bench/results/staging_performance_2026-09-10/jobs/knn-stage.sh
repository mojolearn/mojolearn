#!/bin/bash
set -euo pipefail
export PATH=/root/.pixi/bin:$PATH
cd /root/knnperf
while [ ! -f /root/jobs/gemm-stage.rc ]; do sleep 3; done
if [ "${KNN_STAGE_ACTIVE:-0}" != 1 ]; then
 export KNN_STAGE_ACTIVE=1
 exec pixi run bash /root/jobs/knn-stage.sh
fi
bash tools/knn_coalesced_columns_probe.sh /root/evidence/knn-stage
