#!/bin/sh
# lane linear-apple3 on the M2 Pro: the arms with Metal's API validation on.
# The M2 Pro has no Dynamic Caching: a pipeline's thread limit falls with its
# register use and a dispatch above it is dropped with no error. With the
# validation layer every such dispatch prints
# "... must be <= N. (kernel threadgroup size limit)"; the last lines count them.
set -u
export MTL_DEBUG_LAYER=1 MTL_DEBUG_LAYER_ERROR_MODE=nslog
pixi run -e default python tools/linear_apple3/ab.py "${1:-tools/linear_apple3/m2.json}" > /tmp/l3_m2.out 2>&1
grep -v "threadgroup size limit" /tmp/l3_m2.out
echo "=== Metal validation: dispatches over a pipeline's thread limit"
grep -c "threadgroup size limit" /tmp/l3_m2.out
grep "threadgroup size limit" /tmp/l3_m2.out | sed 's/^.*\] //' | sort | uniq -c | head -40
echo M2DONE
