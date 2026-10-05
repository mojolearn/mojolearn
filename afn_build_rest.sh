#!/bin/bash
cd ~/mojolearn-wt/afn-gemm
for d in SIMDGROUP SPLITK TILESHAPE BF16_MMA INT8_MMA EPILOGUE; do
  ./afn_build.sh "$(echo $d | tr A-Z a-z)" fast "-D MOJOLEARN_AFN_GEMM_$d"
done
./afn_build.sh fastoff fast ""
./afn_build.sh identical identical ""
echo DONE >> afn_build_status.txt
