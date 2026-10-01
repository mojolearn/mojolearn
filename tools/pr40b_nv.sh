#!/bin/bash
cd "$(dirname "$0")/.."
exec bash tools/cell_ab_job2.sh nvidia lu40b "x_decomp linalg" "lu-factor lu-solve" "" MOJOLEARN_XD_LU_FUSED=1 "synthetic"
