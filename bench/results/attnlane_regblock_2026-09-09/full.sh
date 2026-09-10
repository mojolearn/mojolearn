#!/bin/bash
set -eu
export PATH=/root/.pixi/bin:$PATH
cd /root/mojolearn
MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_transformer.sh
./tools/with_identical_mode.sh pixi run mojo build -I . transformer/checks/transformer_check.mojo -o /root/bin/transformer_check
./tools/with_identical_mode.sh pixi run mojo build -I . transformer/checks/transformer_backward_check.mojo -o /root/bin/transformer_backward_check
MOJOLEARN_IDENTITY_TRACE=/root/jobs/transformer.identical.card MOJOLEARN_TRANSFORMER_CHECK_CLAUSE_D=1 /root/bin/transformer_check > /root/jobs/transformer_check.log 2>&1
cmp /root/jobs/transformer.identical.card /root/cards/transformer.identical.card
MOJOLEARN_IDENTITY_TRACE=/root/jobs/transformer-backward.identical.card /root/bin/transformer_backward_check > /root/jobs/transformer_backward_check.log 2>&1
cmp /root/jobs/transformer-backward.identical.card /root/cards/transformer-backward.identical.card
cd python
MOJOLEARN_NUMERIC_MODE=identical python3 -m mojolearn.tests.test_transformer_surface > /root/jobs/surface.log 2>&1
PYTHONPATH=/root/mojolearn/python MOJOLEARN_NUMERIC_MODE=identical python3 ../transformer/bench_window_timing.py --skip-torch --log /root/jobs/candidate_timing.log
PYTHONPATH=/root/mojolearn/python MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TRANSFORMER_TIMING=1 python3 ../transformer/bench_window_timing.py --skip-torch --rounds 1 --log /root/jobs/candidate_breakdown.log
PYTHONPATH=/root/mojolearn/python MOJOLEARN_NUMERIC_MODE=identical python3 ../transformer/bench_window_timing.py --skip-ours --skip-compiled --log /root/jobs/torch_reference.log
python3 -c 'import torch; print(torch.__version__, torch.version.cuda)' > /root/jobs/torch_version.log
