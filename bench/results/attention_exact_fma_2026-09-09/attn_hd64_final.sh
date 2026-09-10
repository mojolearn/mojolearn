#!/bin/bash
set -eu
export PATH=/root/.pixi/bin:$PATH
cd /root/attn/python
PYTHONPATH=/root/attn/python MOJOLEARN_NUMERIC_MODE=identical python3 ../transformer/bench_window_timing.py --skip-torch > /root/jobs/attn-public/final_hd64_timing.log 2>&1
