#!/bin/bash
set -eu
export PATH=/root/.pixi/bin:$PATH
cd /root/attn
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . transformer/checks/attention_ftz_guarded_check.mojo -o /root/attn/bin/mul_production_boundary
/root/attn/bin/mul_production_boundary > /root/jobs/attn-public/mul_production_boundary.log 2>&1
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/gemm_device_check.mojo -o /root/attn-mul-gemm-device
/root/attn-mul-gemm-device > /root/jobs/attn-public/mul_gemm_device.log 2>&1
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/gemm_wide_split_probe.mojo -o /root/attn-mul-gemm-wide
/root/attn-mul-gemm-wide > /root/jobs/attn-public/mul_gemm_wide.log 2>&1
./tools/with_identical_mode.sh pixi run mojo build -I . transformer/checks/transformer_check.mojo -o /root/attn/bin/transformer_check_final
./tools/with_identical_mode.sh pixi run mojo build -I . transformer/checks/transformer_backward_check.mojo -o /root/attn/bin/transformer_backward_check_final
MOJOLEARN_IDENTITY_TRACE=/root/jobs/attn-public/mul_final.transformer.identical.card MOJOLEARN_TRANSFORMER_CHECK_CLAUSE_D=1 /root/attn/bin/transformer_check_final > /root/jobs/attn-public/mul_final_transformer_check.log 2>&1
cmp /root/jobs/attn-public/mul_final.transformer.identical.card /root/attn/cards/transformer.identical.card
MOJOLEARN_IDENTITY_TRACE=/root/jobs/attn-public/mul_final.transformer-backward.identical.card /root/attn/bin/transformer_backward_check_final > /root/jobs/attn-public/mul_final_transformer_backward_check.log 2>&1
cmp /root/jobs/attn-public/mul_final.transformer-backward.identical.card /root/attn/cards/transformer-backward.identical.card
cd python
MOJOLEARN_NUMERIC_MODE=identical python3 -m mojolearn.tests.test_transformer_surface > /root/jobs/attn-public/mul_final_surface.log 2>&1
cd /root/attn
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . gemm/checks/gemm_tuned_probe.mojo -o /root/attn-mul-gemm-tuned
MOJOLEARN_GEMM_BASELINE_PLAN=-2 MOJOLEARN_SPEED_SHAPES=gram.32x32x1M,gram.128sq.x100003,llama8b.qkv.t512,llama8b.mlp_up.t512,llama8b.mlp_down.t512 MOJOLEARN_SPEED_ROUNDS=7 /root/attn-mul-gemm-tuned > /root/jobs/attn-public/mul_gemm_tuned.log 2>&1
