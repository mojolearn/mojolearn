#!/bin/bash
set -euo pipefail
export RUNPOD_POD_ID=4ra98lfm0pqum0 MOJOLEARN_NUMERIC_MODE=identical
export PATH="$HOME/.pixi/bin:$PATH"
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
mv /root/neighbors-out/gate.log /root/neighbors-out/gate-zero-weight-refusal.log
sha256sum python/mojolearn/parallel_neighbors.py python/mojolearn/_parallel_worker.py tools/parallel_neighbors_check.py > /root/neighbors-out/final-source.sha256
pixi run python tools/parallel_neighbors_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/neighbors-out/report.json > /root/neighbors-out/gate.log 2>&1
