#!/bin/bash
# the 46 smoke-passed paper races nvc1-0001 did not time: its board stopped after race 1 because the lane tree was
# patch-synced (the rest job's commit) while the timed board ran ("source changed during measurement"). Never sync a
# lane tree while one of its jobs is running.
export PC_IDS_FILE="$PWD/tools/ops/paper_compare_nvidia_0866_rest2_ids.txt"
export PC_SMOKE_DIR="$PWD/box-out/paper-compare-nv"
exec bash tools/ops/paper_compare_job.sh nvidia 0.8.66
