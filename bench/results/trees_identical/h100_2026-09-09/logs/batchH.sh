#!/bin/sh
# H100 leg, one batch: baseline snapshot, candidate builds, fingerprints, reach gates,
# the reference table (ours IDENTICAL vs the opponents' fast arms), the A/B cells.
cd /root/mojolearn || exit 9
AB="sh tools/trees_identical_ab.sh"
while [ ! -f /root/trees_out/setup.done ]; do sleep 20; done
echo "batchH start $(date -u +%T)"; cat /root/trees_out/setup.txt
mkdir -p /root/bins/baseline && cp python/mojolearn/identical/*.so /root/bins/baseline/
$AB ib baseline
# --- candidate builds ---
cp /root/kernel_matrix.patched.mojo /root/kernel_matrix.patched.keep 2>/dev/null
$AB build fold base
$AB build fold gbdt
mkdir -p /root/bins/fused && cp /root/bins/fold/*.so /root/bins/fused/
$AB build fused gbdt "-D MOJOLEARN_2030_FUSED_EST_MOVE=1"
$AB build rf2010 rf "-D MOJOLEARN_2010_ROWS_SORTED=1"
$AB build rf2011 rf "-D MOJOLEARN_2011_HIST_ITEMS4=1"
$AB build rf2012 rf "-D MOJOLEARN_2012_SMEM_COPIES4=1"
cp checks/kernel_matrix.mojo /root/kernel_matrix.orig.mojo
cp /root/kernel_matrix.patched.mojo checks/kernel_matrix.mojo
mkdir -p /root/bins/sp /root/bins/spnf && cp /root/bins/fold/*.so /root/bins/sp/ && cp /root/bins/fold/*.so /root/bins/spnf/
$AB build spnf gbdt
$AB build sp gbdt "-D MOJOLEARN_2030_FUSED_EST_MOVE=1"
cp /root/kernel_matrix.orig.mojo checks/kernel_matrix.mojo
# --- fingerprints and reach gates ---
$AB ib fold
$AB ib fused gbdt-symmetric,gbdt-depthwise,gbdt-lossguide,gbdt-rmse
$AB ib sp gbdt-symmetric,gbdt-depthwise,gbdt-lossguide,gbdt-rmse
for s in rf2010 rf2011 rf2012; do $AB ib $s rf-clf,rf-reg; done
for s in fold fused sp rf2010 rf2011 rf2012; do $AB diff baseline $s; done
$AB rfgate src
$AB rfgate rf2010 "-D MOJOLEARN_2010_ROWS_SORTED=1"
$AB rfgate rf2011 "-D MOJOLEARN_2011_HIST_ITEMS4=1"
$AB rfgate rf2012 "-D MOJOLEARN_2012_SMEM_COPIES4=1"
echo "PHASE_BUILDS_DONE $(date -u +%T)"
# --- task 1: the reference table, baseline identical vs the opponents ---
$AB speed baseline gbdt-symmetric higgs 1000000 7 full
$AB speed baseline gbdt-symmetric higgsreg 1000000 7 full
$AB speed baseline gbdt-symmetric higgs 1000000 1 stage
$AB speed baseline gbdt-symmetric higgsreg 1000000 1 stage
$AB speed baseline rf higgs 1000000 5 full
$AB speed baseline et higgs 1000000 5 full
$AB speed baseline rf higgs 2000000 5 full
$AB speed baseline et higgs 2000000 5 full
echo "PHASE_TABLE_1M2M_DONE $(date -u +%T)"
# --- tasks 2, 3, 5: symmetric A/B, ours only, 1M and 2M ---
for s in baseline fold fused spnf sp; do $AB speed $s gbdt-symmetric higgs 1000000 7 ours; done
for s in baseline fold fused spnf sp; do $AB speed $s gbdt-symmetric higgs 2000000 5 ours; done
for s in baseline sp; do $AB speed $s gbdt-symmetric higgsreg 1000000 7 ours; done
$AB speed sp gbdt-symmetric higgs 1000000 1 stage
echo "PHASE_SYM_AB_DONE $(date -u +%T)"
# --- task 4: RF candidates, ours only, 1M and 2M ---
for s in baseline rf2010 rf2011 rf2012; do $AB speed $s rf higgs 1000000 5 ours; done
for s in baseline rf2010 rf2011 rf2012; do $AB speed $s rf higgs 2000000 5 ours; done
echo "PHASE_RF_AB_DONE $(date -u +%T)"
# --- the 5M rungs and the final stack against CatBoost ---
$AB speed baseline rf higgs 5000000 5 full
$AB speed baseline et higgs 5000000 5 full
$AB speed sp gbdt-symmetric higgs 1000000 7 full
$AB speed sp gbdt-symmetric higgsreg 1000000 7 full
$AB speed sp gbdt-symmetric higgs 2000000 5 full
$AB speed sp gbdt-symmetric higgs 5000000 5 full
echo "BATCHH_DONE $(date -u +%T)"
