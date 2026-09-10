#!/bin/bash
set -eu
export PATH=/root/.pixi/bin:$PATH
cd /root/attn
./tools/with_identical_mode.sh pixi run mojo build -I . transformer/checks/attention_ftz_guarded_check.mojo -o /root/attn/bin/attention_ftz_guarded_check
/root/attn/bin/attention_ftz_guarded_check > /root/jobs/attn-public/guarded_boundary.log 2>&1
./tools/with_identical_mode.sh pixi run mojo build -I . transformer/checks/transformer_fused_check.mojo -o /root/attn/bin/fused_check_rn
/root/attn/bin/fused_check_rn > /root/jobs/attn-public/guarded_fused_check.log 2>&1
MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_transformer.sh
PYTHONPATH=/root/attn/python MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SPEED_DUMP_DIR=/root/jobs/attn-public/guarded_dump python3 bench/speed/seq_py_speed_arm.py --lane transformer --rows large --rounds 5 > /root/jobs/attn-public/guarded.log 2>&1
sha256sum /root/jobs/attn-public/guarded_dump/*.bin > /root/jobs/attn-public/guarded_full_sha256.log
for name in narrow.b8_l4096_d512 wide.b8_l1024_d2048; do
    cmp /root/jobs/attn-public/base_dump/seq.transformer.$name.f32.bin /root/jobs/attn-public/guarded_dump/seq.transformer.$name.f32.bin
done
PYTHONPATH=/root/attn/python MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TRANSFORMER_TIMING=1 python3 bench/speed/seq_py_speed_arm.py --lane transformer --rows large --rounds 1 > /root/jobs/attn-public/guarded_profile.log 2>&1
PYTHONPATH=/root/attn/python MOJOLEARN_NUMERIC_MODE=identical python3 -m mojolearn.tests.test_transformer_hd128 > /root/jobs/attn-public/guarded_hd128_surface.log 2>&1
