#!/bin/sh
export MOJOLEARN_KNN_SELECTION_ARMS=uniform,warpbound_guard
export MOJOLEARN_KNN_SELECTION_SKIP_PROFILE=1
export MOJOLEARN_KNN_SELECTION_OPPONENT=1
exec sh /root/mojolearn/tools/knn_selection_gate.sh
