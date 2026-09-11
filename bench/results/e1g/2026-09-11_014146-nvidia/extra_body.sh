#!/bin/sh
export MOJOLEARN_KNN_SELECTION_ARMS=uniform,capk,capk_selp
export MOJOLEARN_KNN_SELECTION_PHASE_TIMERS=1
export MOJOLEARN_KNN_SELECTION_SKIP_PROFILE=1
cd /root/mojolearn && PATH="$HOME/.pixi/bin:$PATH" pixi run mojo run \
    -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_KNN_SELECT_TRIAL=1 \
    -I . neighbors/checks/knn_selector_arms_check.mojo \
    > /root/gemm_leg_out/knn-selector-arms-check.log 2>&1
echo "arms_check_exit=$?" >> /root/gemm_leg_out/leg.txt
exec sh /root/mojolearn/tools/knn_selection_gate.sh
