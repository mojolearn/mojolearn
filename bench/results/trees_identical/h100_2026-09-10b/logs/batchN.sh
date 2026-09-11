#!/bin/sh
# After batch M: the DEVIATION 2510 gbdt host-phase stamps, one stage replicate each.
cd /root/mojolearn || exit 9
AB="sh tools/trees_identical_ab.sh"
OUT=/root/trees_out
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical
while [ ! -f $OUT/phase.PHASE_D_DONE ]; do sleep 10; done
$AB build stampsg gbdt
$AB speed stampsg gbdt-depthwise higgs 1000000 1 stage
$AB speed stampsg gbdt-lossguide higgs 1000000 1 stage
echo "PHASE_N_DONE $(date -u +%T)" | tee -a $OUT/ab.txt; : > $OUT/phase.PHASE_N_DONE
