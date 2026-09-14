#!/bin/bash
set -euo pipefail
export RUNPOD_POD_ID=4ra98lfm0pqum0 MOJOLEARN_NUMERIC_MODE=identical
export PATH="$HOME/.pixi/bin:$PATH"
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
pixi run mojo build --target-accelerator sm_90a -D MOJOLEARN_COLUMN_NVIDIA -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . training/checks/dbscan_parallel_check.mojo -o /root/dbscan-out/native > /root/dbscan-out/native-build.log 2>&1
sha256sum /root/dbscan-out/native > /root/dbscan-out/native.sha256
MOJOLEARN_DBSCAN_DEVICE_COUNT=1 /root/dbscan-out/native > /root/dbscan-out/native-one.log 2>&1
MOJOLEARN_DBSCAN_DEVICE_COUNT=2 /root/dbscan-out/native > /root/dbscan-out/native-two.log 2>&1
pixi run python tools/parallel_dbscan_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/dbscan-out/report.json > /root/dbscan-out/public.log 2>&1
