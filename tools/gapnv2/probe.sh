#!/bin/bash
# lane/gap-nv-classical2: run the stage probes on a GPU box (lq CMD); prints PROBE / ANN-STAGE lines.
cd "$(dirname "$0")/../.."
for spec in ${PROBES:-rsvd:istella rsvd:taxi ivf-sq:istella ivf-refine:istella lasso:istella lasso:taxi}; do
  a=${spec%%:*}; d=${spec#*:}
  timeout 900 python tools/gapnv2/probe.py $a $d 2>&1 | grep -E '^PROBE|^ANN-STAGE|Error|error' | head -60
done
echo "probe done"
