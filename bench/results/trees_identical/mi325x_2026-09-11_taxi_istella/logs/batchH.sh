#!/bin/sh
# Leg 1, while the Istella-S tar downloads (network only, CPU quiet): the
# taxi pointwise A/B showed our IDENTICAL symmetric arm with a DIFFERENT
# prediction hash every round (the pointwise arm held one hash). Repeat-round
# hash checks, ours only, 5 rounds, every lane on taxi; then identity_break
# fingerprints of this box diffed against the H100 set of the same default
# forest (h100_2026-09-11_istella ib/pureleaf.json). Each cell starts only
# while setup.done is absent, so none overlaps the opponent cells.
cd /root/mojolearn || exit 9
AB="sh tools/trees_identical_ab.sh"
OUT=/root/trees_out
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical
export GBM_BENCH_DATA=/mnt/mojolearn-data/gbm-bench
quiet() { [ ! -f $OUT/setup.done ]; }
echo "batchH start $(date -u +%T)"
quiet && MOJOLEARN_SPEED_TAG=hashcheck $AB speed baseline gbdt-symmetric taxi 1000000 5 ours
quiet && MOJOLEARN_SPEED_TAG=hashcheck $AB speed baseline gbdt-depthwise taxi 1000000 5 ours
quiet && MOJOLEARN_SPEED_TAG=hashcheck $AB speed baseline gbdt-lossguide taxi 1000000 5 ours
quiet && $AB ib baseline
quiet && $AB diff h100_pureleaf baseline
quiet && MOJOLEARN_SPEED_TAG=hashcheck $AB speed baseline rf taxi 1000000 5 ours
quiet && MOJOLEARN_SPEED_TAG=hashcheck $AB speed baseline et taxi 1000000 5 ours
echo "batchH end $(date -u +%T) setup_done=$([ -f $OUT/setup.done ] && echo yes || echo no)" | tee -a $OUT/ab.txt
