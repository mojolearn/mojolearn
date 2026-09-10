#!/bin/bash
set -euo pipefail
cd /root/continuation
tar -xzf /root/mamba-fixtures.tar.gz
export PATH=/root/.pixi/bin:$PATH
export MOJOLEARN_MAMBA3_REPO=/root/continuation MOJOLEARN_MAMBA3_RESULTS=/root/evidence/mamba-yintra MOJOLEARN_MAMBA3_PYTHON=/usr/bin/python3
bash tools/mamba3_yintra_tile_leg.sh
