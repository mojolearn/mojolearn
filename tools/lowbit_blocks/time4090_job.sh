#!/bin/bash
# lane/lowbit-blocks on the 4090: the checked time at this commit (bindings built by the last job at the same code).
cd "$(dirname "$0")/../.." || exit 9
LB_BOX=rtx4090-nvc2 bash tools/lowbit_blocks/time_job.sh
