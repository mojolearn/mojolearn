#!/bin/bash
set -euo pipefail
export RUNPOD_POD_ID=4ra98lfm0pqum0 MOJOLEARN_NUMERIC_MODE=identical
export PATH="$HOME/.pixi/bin:$PATH"
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
mv /root/reference-out/report.json /root/reference-out/report-initial-56.json
mv /root/reference-out/public.log /root/reference-out/public-initial-56.log
sha256sum python/mojolearn/parallel_neighbors_reference.py tools/parallel_reference_neighbors_check.py > /root/reference-out/final-source.sha256
pixi run python tools/parallel_reference_neighbors_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/reference-out/report.json > /root/reference-out/public.log 2>&1
pixi run python tools/parallel_reference_capacity_check.py --cloud --corpus training/corpus/enwik8/input.txt --reference-gib 2 --shard-gib 1 --report /root/reference-out/capacity-small-final.json > /root/reference-out/capacity-small-final.log 2>&1
pixi run python tools/parallel_graph_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/graph-out/report.json > /root/graph-out/public.log 2>&1
