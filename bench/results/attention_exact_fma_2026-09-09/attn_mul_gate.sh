#!/bin/bash
set -eu
export PATH=/root/.pixi/bin:$PATH
cd /root/attn
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . transformer/checks/attention_ftz_mul_check.mojo -o /root/attn/bin/attention_ftz_mul_check
while [ ! -f /root/jobs/attn-guarded-final.done ]; do sleep 2; done
/root/attn/bin/attention_ftz_mul_check > /root/jobs/attn-public/mul_boundary.log 2>&1
