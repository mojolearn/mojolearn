#!/bin/bash
bash /root/mojolearn-pass28/tools/cell_ab_job.sh nvidia emb "embedding" "embedding" "check-embedding" -
bash /root/mojolearn-pass29/tools/cell_ab_job.sh nvidia moe "x_sequence" "moe" "" MOJOLEARN_SEQ_MOE_TILED=0
