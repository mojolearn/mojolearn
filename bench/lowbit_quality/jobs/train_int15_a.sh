#!/bin/bash
# Part a of the split 15-bit training job; see train_int15_split.sh.
export LOWBIT_QUALITY_PART=0
. "$(dirname "${BASH_SOURCE[0]}")/train_int15_split.sh"
