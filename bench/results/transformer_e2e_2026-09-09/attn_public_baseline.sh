#!/bin/bash
set -eu
export PATH=/root/.pixi/bin:$PATH
cd /root/attn
mkdir -p /root/jobs/attn-public/base_dump
MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_transformer.sh
cp python/mojolearn/identical/_mojolearn_transformer.so /root/jobs/attn-public/baseline.so
PYTHONPATH=/root/attn/python MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SPEED_DUMP_DIR=/root/jobs/attn-public/base_dump python3 bench/speed/seq_py_speed_arm.py --lane transformer --rows large --rounds 5 > /root/jobs/attn-public/baseline.log 2>&1
PYTHONPATH=/root/attn/python MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TRANSFORMER_TIMING=1 python3 bench/speed/seq_py_speed_arm.py --lane transformer --rows large --rounds 1 > /root/jobs/attn-public/baseline_profile.log 2>&1
sha256sum /root/jobs/attn-public/base_dump/*.bin > /root/jobs/attn-public/baseline_full_sha256.log
