#!/bin/sh
# Written by tools/do_extra_leg.sh. The integrity check that replaces a
# clone's: the box recomputes the archive's sha256 and refuses a mismatch.
set -eu
cd /root
got=$(sha256sum extra_src.tgz | awk '{print $1}')
if [ "$got" != "797b2f26c4ced56051b385eb6959a4b52c7251fa2e5ca66734b7a2e9db771a18" ]; then echo "ARCHIVE SHA MISMATCH: sent 797b2f26c4ced56051b385eb6959a4b52c7251fa2e5ca66734b7a2e9db771a18 got $got"; exit 9; fi
echo ARCHIVE-SHA-OK
rm -rf /root/mojolearn /root/gemm_leg_out /root/gemm_leg.done /root/gemm_leg_extra.sh
mkdir -p /root/mojolearn
tar -xzf extra_src.tgz -C /root/mojolearn
rm -f extra_src.tgz
echo "UNPACKED $(find /root/mojolearn -type f | wc -l) files"
