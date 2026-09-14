#!/bin/bash
set -euo pipefail
export RUNPOD_POD_ID=4ra98lfm0pqum0 MOJOLEARN_NUMERIC_MODE=identical
export PATH="$HOME/.pixi/bin:$PATH"
cd /root/mojolearn
export PYTHONPATH="$PWD/python"
sha256sum python/mojolearn/parallel_neighbors_reference.py tools/parallel_reference_capacity_check.py > /root/reference-out/capacity-source.sha256
nvidia-smi --query-gpu=timestamp,uuid,memory.total,memory.used --format=csv -l 2 > /root/reference-out/capacity-gpu-memory.csv &
monitor_pid=$!
trap 'kill "$monitor_pid" 2>/dev/null || true' EXIT
pixi run python tools/parallel_reference_capacity_check.py --cloud --corpus training/corpus/enwik8/input.txt --reference-gib 2 --shard-gib 1 --report /root/reference-out/capacity-small.json > /root/reference-out/capacity-small.log 2>&1
pixi run python tools/parallel_reference_capacity_check.py --cloud --corpus training/corpus/enwik8/input.txt --report /root/reference-out/capacity-96gib.json > /root/reference-out/capacity-96gib.log 2>&1
cat /sys/fs/cgroup/memory.peak > /root/reference-out/container-memory-peak.txt
