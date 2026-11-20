#!/bin/bash
# all 61 paper races still untimed on NVIDIA (62 minus dbscan/istella, timed by nvc1-0001): smoke first (smoke ids),
# then the timed board, in ONE job. Reusing nvc1-0001's smoke.json was refused: the smoke gate keys on the board and
# driver files' sha256 and the tree was synced since.
export PC_IDS_FILE="$PWD/tools/ops/paper_compare_nvidia_0866_all61_ids.txt"
exec bash tools/ops/paper_compare_job.sh nvidia 0.8.66
