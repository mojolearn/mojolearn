#!/bin/sh
# batch 2 (L40S iteration): the symmetric profile cells, then the A/B sets ours-only
cd /root/mojolearn || exit 9
AB="sh tools/trees_identical_ab.sh"
while ! grep -q HIGGS_DONE /root/trees_out/higgs_fast.log 2>/dev/null; do sleep 20; done
while ! grep -q BATCH1B_DONE /root/trees_out/batch1b.log 2>/dev/null; do sleep 20; done
echo "batch2 start $(date -u +%T)"
$AB speed baseline gbdt-symmetric higgs 1000000 7 full
$AB speed baseline gbdt-symmetric higgsreg 1000000 7 full
$AB speed baseline gbdt-symmetric higgs 1000000 1 stage
$AB speed baseline gbdt-symmetric higgsreg 1000000 1 stage
$AB speed baseline gbdt-symmetric higgs 1000000 7 ours
$AB speed fold gbdt-symmetric higgs 1000000 7 ours
$AB speed fused gbdt-symmetric higgs 1000000 7 ours
$AB speed baseline gbdt-symmetric higgs 2000000 5 ours
$AB speed fold gbdt-symmetric higgs 2000000 5 ours
$AB speed fused gbdt-symmetric higgs 2000000 5 ours
$AB speed baseline rf higgs 1000000 5 ours
$AB speed rf2010 rf higgs 1000000 5 ours
$AB speed rf2011 rf higgs 1000000 5 ours
$AB speed rf2012 rf higgs 1000000 5 ours
$AB speed baseline rf higgs 2000000 5 ours
$AB speed rf2010 rf higgs 2000000 5 ours
$AB speed rf2011 rf higgs 2000000 5 ours
$AB speed rf2012 rf higgs 2000000 5 ours
$AB speed baseline rf higgs 1000000 5 full
$AB speed baseline et higgs 1000000 5 full
echo BATCH2_DONE
