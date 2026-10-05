#!/bin/bash
# afn-samba compile matrix: one build at a time, through the slot semaphore.
cd ~/mojolearn-wt/afn-samba || exit 2
: > scratch-afn/SUMMARY.txt
run() {  # tag binding mode defines
  local tag=$1 b=$2 mode=$3 defs=$4
  MOJOLEARN_COMPILE_JOBS=1 MOJOLEARN_NUMERIC_MODE=$mode MOJOLEARN_MOJO_BUILD_FLAGS="$defs" \
    bash ~/mojolearn-evidence/compile_slot.sh bash bindings/build_$b.sh > scratch-afn/build-$tag.log 2>&1
  echo "$tag rc=$?" >> scratch-afn/SUMMARY.txt
}
run training-ALL      training fast "-D MOJOLEARN_AFN_SAMBA_ALL"
run mamba-ALL         mamba    fast "-D MOJOLEARN_AFN_SAMBA_ALL"
run training-FUSE     training fast "-D MOJOLEARN_AFN_SAMBA_FUSE"
run training-ARENA    training fast "-D MOJOLEARN_AFN_SAMBA_ARENA"
run training-ADMIT    training fast "-D MOJOLEARN_AFN_SAMBA_DEVICE_ADMIT"
run training-ATOMIC   training fast "-D MOJOLEARN_AFN_SAMBA_EMB_ATOMIC"
run mamba-CHUNK       mamba    fast "-D MOJOLEARN_AFN_MAMBA3_BWD_CHUNK"
run mamba-ARENA       mamba    fast "-D MOJOLEARN_AFN_MAMBA3_BWD_ARENA"
run training-off      training fast ""
run mamba-off         mamba    fast ""
run training-IDENT    training identical ""
run mamba-IDENT       mamba    identical ""
echo DONE >> scratch-afn/SUMMARY.txt
