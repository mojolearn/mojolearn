#!/bin/bash
set -eu
export PATH=/root/.pixi/bin:$PATH
cd /root/attn
./tools/with_identical_mode.sh pixi run mojo build -I . transformer/checks/attention_ftz_boundary_check.mojo -o /root/attn/bin/attention_ftz_boundary_check
/root/attn/bin/attention_ftz_boundary_check > /root/jobs/attn-public/boundary.log 2>&1
