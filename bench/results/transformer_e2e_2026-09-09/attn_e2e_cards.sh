#!/bin/bash
set -eu
export PATH=/root/.pixi/bin:$PATH
cd /root/attn
./tools/with_identical_mode.sh pixi run mojo build -I . transformer/checks/transformer_check.mojo -o /root/attn/bin/transformer_check_final
./tools/with_identical_mode.sh pixi run mojo build -I . transformer/checks/transformer_backward_check.mojo -o /root/attn/bin/transformer_backward_check_final
MOJOLEARN_IDENTITY_TRACE=/root/jobs/attn-public/final.transformer.identical.card MOJOLEARN_TRANSFORMER_CHECK_CLAUSE_D=1 /root/attn/bin/transformer_check_final > /root/jobs/attn-public/final_transformer_check.log 2>&1
cmp /root/jobs/attn-public/final.transformer.identical.card /root/attn/cards/transformer.identical.card
MOJOLEARN_IDENTITY_TRACE=/root/jobs/attn-public/final.transformer-backward.identical.card /root/attn/bin/transformer_backward_check_final > /root/jobs/attn-public/final_transformer_backward_check.log 2>&1
cmp /root/jobs/attn-public/final.transformer-backward.identical.card /root/attn/cards/transformer-backward.identical.card
cd python
MOJOLEARN_NUMERIC_MODE=identical python3 -m mojolearn.tests.test_transformer_surface > /root/jobs/attn-public/final_surface.log 2>&1
