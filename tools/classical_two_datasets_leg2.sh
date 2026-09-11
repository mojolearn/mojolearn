#!/bin/sh
# tools/classical_two_datasets_leg2.sh -- the second lease of DEVIATION 2570's
# classical leg: kde and svc (no torch arm) by default. tools/do_extra_leg.sh
# passes no environment to a body, so the lane list lives here; everything
# else is tools/classical_two_datasets_leg.sh.
#
#   MOJOLEARN_DO_TOKEN_FILE=$HOME/.mojolearn_do_token \
#   MOJOLEARN_GPU_ARCHS=gfx942 \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/classical_two_datasets_leg2.sh \
#   MOJOLEARN_GEMM_LEG_OUT=$HOME/mojolearn-evidence/classical-two-datasets-$(date -u +%Y-%m-%d_%H%M%S)-amd-leg2 \
#   bash tools/do_extra_leg.sh amd --minutes 60 --skip-gates
MOJOLEARN_CTD_LANES=${MOJOLEARN_CTD_LANES:-kde,svc}
export MOJOLEARN_CTD_LANES
exec sh /root/mojolearn/tools/classical_two_datasets_leg.sh
