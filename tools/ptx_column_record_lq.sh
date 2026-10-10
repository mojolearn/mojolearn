#!/bin/bash
# ptx_column_record_lq.sh <ptx-set-dir> <outdir>
# The PTX identity column, run ONCE through `lq add <nv box> CMD` (2026-10-10).
# Copies the cuda-sm_80 leg's set (scp'd to the box, outside the tree) into
# python/mojolearn/cuda/sm_80 of this lq branch tree, then records the column
# with REPEATS=1. The record lands in <outdir> (outside the tree).
set -u
# <ptx-set-dir> may instead be a presigned R2 GET URL of the leg's sets/cuda.tar.gz
# (minted on the Mac; no credential reaches the box) with its sha256 as the third
# argument: the box fetches, verifies and unpacks it to /root/ptx-set first.
SET=${1:?ptx set dir (holding the sm_80 bindings) or presigned URL}; OUT=${2:?outdir}
case "$SET" in https://*)
    SHA=${3:?sha256 of the set tarball}; D=/root/ptx-set; mkdir -p "$D"
    curl -fsS --retry 3 -o "$D/cuda.tar.gz" "$SET" || { echo "PTXREC fetch failed"; exit 2; }
    got=$(sha256sum "$D/cuda.tar.gz" | cut -c1-64)
    [ "$got" = "$SHA" ] || { echo "PTXREC sha256 $got != $SHA"; exit 2; }
    rm -rf "$D/cuda" && tar xzf "$D/cuda.tar.gz" -C "$D" || { echo "PTXREC untar failed"; exit 2; }
    SET="$D/cuda/sm_80" ;;
esac
[ -f "$SET/PTX_BASELINE.json" ] || { echo "PTXREC no PTX_BASELINE.json in $SET"; exit 2; }
rm -rf python/mojolearn/cuda/sm_80
mkdir -p python/mojolearn/cuda
cp -a "$SET" python/mojolearn/cuda/sm_80
echo "PTXREC set=$(ls python/mojolearn/cuda/sm_80 | wc -l) files"
REPEATS=1 bash tools/record_identity_column.sh nvidia-ptx-l40s-sm80 "$OUT"
rc=$?
echo "PTXREC rc=$rc out=$OUT $(ls "$OUT" 2>/dev/null | grep -m 3 identical | tr '\n' ' ')"
exit $rc
