#!/bin/bash
set -eu
export PATH=/root/.pixi/bin:$PATH
cd /root/mojolearn
MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_transformer.sh
cp python/mojolearn/_mojolearn_transformer.so /root/transformer_baseline.so
cd python
PYTHONPATH=/root/mojolearn/python MOJOLEARN_NUMERIC_MODE=identical python3 ../transformer/bench_window_timing.py --skip-torch --log /root/jobs/baseline_timing.json
PYTHONPATH=/root/mojolearn/python MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TRANSFORMER_TIMING=1 python3 ../transformer/bench_window_timing.py --skip-torch --rounds 1 --log /root/jobs/baseline_breakdown.json
