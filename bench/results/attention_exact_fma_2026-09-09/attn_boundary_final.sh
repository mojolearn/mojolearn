#!/bin/bash
set -eu
export PATH=/root/.pixi/bin:$PATH
cd /root/attn
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . transformer/checks/attention_fma_boundary_check.mojo -o /root/attn/bin/attention_fma_boundary_final
while [ ! -f /root/jobs/attn-mul-final.done ]; do sleep 2; done
/root/attn/bin/attention_fma_boundary_final > /root/jobs/attn-public/mul_boundary_final.log 2>&1
