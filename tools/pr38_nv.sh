#!/bin/bash
bash /root/mojolearn-pass34/tools/pr38_opt.sh nvidia
bash /root/mojolearn-pass34/tools/cell_ab_job2.sh nvidia opt38 "training" "sgd adam adamw" "" MOJOLEARN_OPT_POOL=1 "synthetic"
