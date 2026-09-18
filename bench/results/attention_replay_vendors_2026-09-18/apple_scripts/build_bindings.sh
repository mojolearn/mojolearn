#!/bin/bash
# CPU slot: Apple byte-LM bindings for the reduced witness (bswz = NVIDIA
# schedule define so Metal training reaches the replay kernels; legacy = pre-flip).
set -u
cd /Users/andrewhendel/mojolearn-wt/attention-replay-vendors
EV=/Users/andrewhendel/mojolearn-evidence/attention-replay-vendors
for arm in bswz legacy; do
  case $arm in
    bswz) d="-D MOJOLEARN_ATTN_DEFAULT_BSWZ_EVERY_COLUMN=1" ;;
    legacy) d="-D MOJOLEARN_ATTN_LEGACY_CORNER=1 -D MOJOLEARN_BYTE_LM_RETAIN_EAGER=1" ;;
  esac
  rm -rf "$EV/build-apple-$arm"
  MOJOLEARN_COMPILE_JOBS=1 MOJOLEARN_BYTE_LM_OUTDIR=$EV/build-apple-$arm MOJOLEARN_BUILD_EXTRA_DEFINES="$d" \
    sh bindings/build_byte_lm.sh > "$EV/apple-native/build_binding_$arm.log" 2>&1
  echo "$arm exit=$?" >> "$EV/apple-native/build_bindings_status.txt"
done
