#!/bin/bash
# the 15 paper races the first NVIDIA run (nvc1-0001) left untimed: "no smoke result" (smoke-id mapping), timed directly
export PC_IDS_FILE="$PWD/tools/ops/paper_compare_nvidia_0866_rest_ids.txt"
exec bash tools/ops/paper_compare_job.sh nvidia 0.8.66
