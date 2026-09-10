#!/bin/bash
set -euo pipefail
export PATH=/root/.pixi/bin:$PATH
cd /root/mambaperf
tar xzf /root/stage-mamba-fixtures.tar.gz
tar xzf /root/stage-grid.tar.gz
while [ ! -f /root/jobs/knn-stage.rc ]; do sleep 3; done
export MOJOLEARN_MAMBA3_REPO=/root/mambaperf MOJOLEARN_MAMBA3_RESULTS=/root/evidence/mamba-stage MOJOLEARN_MAMBA3_PYTHON=/usr/bin/python3
bash tools/mamba3_scratch_leg.sh
