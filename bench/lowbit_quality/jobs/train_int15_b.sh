#!/bin/bash
# Part b of the split 15-bit training job; see train_int15_split.sh.
export LOWBIT_QUALITY_PART=1
. "$(dirname "${BASH_SOURCE[0]}")/train_int15_split.sh"
