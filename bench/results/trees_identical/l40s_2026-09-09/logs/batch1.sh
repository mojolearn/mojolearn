#!/bin/sh
# batch 1: baseline fingerprints, the fold/fused/RF candidate builds, their fingerprints, diffs
cd /root/mojolearn || exit 9
AB="sh tools/trees_identical_ab.sh"
$AB ib baseline
$AB build fold base
$AB build fold gbdt
mkdir -p /root/bins/fused && cp /root/bins/fold/*.so /root/bins/fused/
$AB build fused gbdt "-D MOJOLEARN_2030_FUSED_EST_MOVE=1"
$AB build rf2010 rf "-D MOJOLEARN_2010_ROWS_SORTED=1"
$AB build rf2011 rf "-D MOJOLEARN_2011_HIST_ITEMS4=1"
$AB build rf2012 rf "-D MOJOLEARN_2012_SMEM_COPIES4=1"
$AB ib fold
$AB ib fused gbdt-symmetric,gbdt-depthwise,gbdt-lossguide,gbdt-rmse
$AB ib rf2010 rf-clf,rf-reg
$AB ib rf2011 rf-clf,rf-reg
$AB ib rf2012 rf-clf,rf-reg
for s in fold fused rf2010 rf2011 rf2012; do $AB diff baseline $s; done
$AB rfgate baseline
$AB rfgate rf2010 "-D MOJOLEARN_2010_ROWS_SORTED=1"
$AB rfgate rf2011 "-D MOJOLEARN_2011_HIST_ITEMS4=1"
$AB rfgate rf2012 "-D MOJOLEARN_2012_SMEM_COPIES4=1"
echo BATCH1_DONE
