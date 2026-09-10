#!/bin/bash
set -eu
export PATH=/root/.pixi/bin:$PATH
cd /root/mojolearn
./tools/with_identical_mode.sh pixi run mojo build -I . transformer/checks/transformer_fused_check.mojo -o /root/bin/fused_check_bk64
/root/bin/fused_check_bk64 > /root/jobs/gate_bk64.log 2>&1
MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_transformer.sh
cd python
PYTHONPATH=/root/mojolearn/python MOJOLEARN_NUMERIC_MODE=identical python3 ../transformer/bench_window_timing.py --skip-torch --log /root/jobs/bk64_timing.log
PYTHONPATH=/root/mojolearn/python MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TRANSFORMER_TIMING=1 python3 ../transformer/bench_window_timing.py --skip-torch --rounds 1 --log /root/jobs/bk64_breakdown.log
