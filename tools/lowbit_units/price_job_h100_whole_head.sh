#!/bin/bash
# The head at t512 WHOLE (no cap: m=512 n=128256 k=4096) on the H100, filed
# beside the capped run of record under price_whole_head. Its digests are
# comparable only with another box's run at the same extents.
export MOJOLEARN_LOWBIT_BOX=h100
export MOJOLEARN_LOWBIT_PRICE_DIR=price_whole_head
export MOJOLEARN_LOWBIT_PRICE_MAC_BUDGET=0
export MOJOLEARN_LOWBIT_PRICE_ONLY=lm_head.t512
exec bash "$(dirname "$0")/price_job.sh"
