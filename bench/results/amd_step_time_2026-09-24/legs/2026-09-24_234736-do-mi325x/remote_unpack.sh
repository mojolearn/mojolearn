#!/bin/sh
# Written by tools/do_extra_leg.sh. The integrity check that replaces a
# clone's: the box recomputes the archive's sha256 and refuses a mismatch.
set -eu
cd /root
got=$(sha256sum extra_src.tgz | awk '{print $1}')
if [ "$got" != "18ec85cea362e6f4694862b2880ef9a94c0a6a447cb0c56c3361ef1eae2e2ae4" ]; then echo "ARCHIVE SHA MISMATCH: sent 18ec85cea362e6f4694862b2880ef9a94c0a6a447cb0c56c3361ef1eae2e2ae4 got $got"; exit 9; fi
echo ARCHIVE-SHA-OK
rm -rf /root/mojolearn /root/gemm_leg_out /root/gemm_leg.done /root/gemm_leg_extra.sh
mkdir -p /root/mojolearn
tar -xzf extra_src.tgz -C /root/mojolearn
rm -f extra_src.tgz
echo "UNPACKED $(find /root/mojolearn -type f | wc -l) files"
