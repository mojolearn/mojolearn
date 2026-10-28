#!/bin/bash
# Self-merge re-smoke (lane decode-resident-20261028): the host families through the cache, each build declaring
# its own output, then import mojolearn + the tier_dir probe. No compile roots here: the roots were green on
# pod 8jy6vhu4bfpioh and no file they import changed since.
set -u
OUT=${LEG_OUT:?}; mkdir -p "$OUT"
rm -f python/mojolearn/host/*.so
LEG_OUT=$OUT/smoke bash experiments/decode_resident_20261028/pod_smoke.sh > "$OUT/pod_smoke.out" 2>&1; rc=$?
echo "pod_smoke rc=$rc" > "$OUT/selfmerge.txt"
cat "$OUT/smoke/summary.txt" >> "$OUT/selfmerge.txt" 2>/dev/null
grep -v 'rc=0' "$OUT/smoke/results.tsv" >> "$OUT/selfmerge.txt" 2>/dev/null
cat "$OUT/selfmerge.txt"
exit $rc
