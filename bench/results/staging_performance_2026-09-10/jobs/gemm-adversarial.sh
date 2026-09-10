#!/bin/bash
set -euo pipefail
export PATH=/root/.pixi/bin:$PATH
cd /root/stageperf
while [ ! -f /root/jobs/mamba-stage.rc ]; do sleep 3; done
if [ "${GEMM_ADVERSARIAL_ACTIVE:-0}" != 1 ]; then
 export GEMM_ADVERSARIAL_ACTIVE=1
 exec pixi run bash /root/jobs/gemm-adversarial.sh
fi
out=/root/evidence/gemm-stage
mojo run -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_GEMM_STAGE_FTZ=1 -I . gemm/checks/gemm_stage_ftz_check.mojo > "$out/adversarial.log" 2>&1
