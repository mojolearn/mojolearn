#!/bin/bash
set -euo pipefail
cd /Users/andrewhendel/CascadeProjects/mojolearn
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_MAMBA3_TILED_YSTATE=1 -D MOJOLEARN_MAMBA3_TILED_QKS=1 -I . mamba/checks/mamba3_check.mojo -o /tmp/mojolearn-mamba3-both-tiles-apple > /tmp/mojolearn-mamba3-both-tiles-apple-build.log 2>&1
bash /tmp/mojolearn-mamba3-both-tiles-apple-check.sh
