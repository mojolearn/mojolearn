#!/bin/bash
bash /root/mojolearn-pass40/tools/cell_ab_job2.sh nvidia seq45 "x_sequence" "layernorm cross-entropy maxpool2d avgpool2d batchnorm2d" "" MOJOLEARN_SEQ_POOL=0 "synthetic"
