#!/bin/bash
# lane/linfit-speed: the board shapes, this tree (new arm only; the base arm's
# times and words on this box are job nvc1-0005's), then identity.
set -uo pipefail
SKIP_BASE=1 STAGE=full bash "$(dirname "$0")/job_nv.sh"
echo "base (nvc1-0005, same L40S): poisson taxi 48.974s 4d7dede9ad8f88ca istella 920.563s 6afc53f88796e11c; sgd-reg taxi 128.001s 378e0dda34958a93 istella 630.0s 077f8d3a2175f52d; sgd-clf taxi 128.139s 1f3a018cb3d17cb5 istella 614.807s e7be9f3686458fdf"
bash "$(dirname "$0")/nv_identity.sh"
