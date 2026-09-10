#!/bin/bash
set -euo pipefail
export MOJOLEARN_MAMBA3_REPO=/root/mamba3-next
export MOJOLEARN_MAMBA3_RESULTS=/root/jobs/m3-increment-tile
export MOJOLEARN_MAMBA3_PYTHON=/usr/bin/python3
export MOJOLEARN_PIXI=/root/.pixi/bin/pixi
exec bash /root/mamba3-next/tools/mamba3_increment_tile_leg.sh
