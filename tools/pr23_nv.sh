#!/bin/bash
cd "$(dirname "$0")/.."
exec bash tools/cell_ab_job2.sh nvidia svd23 "x_decomp linalg" "svd" "" MOJOLEARN_LINALG_SVD_U=householder "taxi istella"
