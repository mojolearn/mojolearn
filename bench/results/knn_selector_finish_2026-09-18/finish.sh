#!/bin/bash
# The kNN selector tip (defaults flipped, no defines) against a base from origin/main: the doc's finish sequence.
export PATH="$HOME/.pixi/bin:$PATH" PYTHONUNBUFFERED=1
cd /root/mojolearn; B=tools/knn_selector_body.sh; O=/root/kss_out
say() { echo "[$(date +%T)] $*"; }
say start; sh $B setup; say "setup rc=$?"
sh $B arm final ""; say "arm final rc=$?"; sh $B hostafter; say "hostafter rc=$?"
sh $B arm final_allk "-D MOJOLEARN_KNN_SELECTOR_BOUND_ALL_K=1 -D MOJOLEARN_KNN_IDENTICAL_MATRIX_SELECT=1"; say "arm final_allk rc=$?"
sh $B arm final_allk_sabo "-D MOJOLEARN_KNN_SELECTOR_BOUND_ALL_K=1 -D MOJOLEARN_KNN_IDENTICAL_MATRIX_SELECT=1 -D MOJOLEARN_KNN_SELECTOR_BOUND_SABOTAGE=1"; say "arm final_allk_sabo rc=$?"
sh $B arm final_selsabo "-D MOJOLEARN_KNN_SELECTOR_BOUND_SABOTAGE=1"; say "arm final_selsabo rc=$?"
sh $B arm final_cachesabo "-D MOJOLEARN_KNN_RESIDENT_CACHE_SABOTAGE=1"; say "arm final_cachesabo rc=$?"
for a in base final final_allk final_allk_sabo final_cachesabo; do sh $B identity identity2 $a; say "identity $a rc=$?"; done
sh $B cpuidentity identity2 base; say "cpuidentity base rc=$?"; sh $B cpuidentity identity2 after; say "cpuidentity after rc=$?"
ls $O/identity2/
I=$O/identity2
sh $B diff identity2 base-vs-final $I/base.json $I/final.json
sh $B diff identity2 base-vs-final-allk $I/base.json $I/final_allk.json
sh $B diff identity2 final-allk-vs-sabo $I/final_allk.json $I/final_allk_sabo.json
sh $B diff identity2 final-vs-cachesabo $I/final.json $I/final_cachesabo.json
sh $B diff identity2 cpu-base-vs-cpu-after $I/cpu-base.json $I/cpu-after.json
sh $B diff identity2 final-vs-cpu-after $I/final.json $I/cpu-after.json
for f in $I/diff.*.txt; do say "$(basename $f): $(grep '^summary' $f | tr '\n' ' ')"; done
sh $B race race2 base,final 1,10,32,64 4000,1; say "race rc=$?"; cat $O/race2/race_summary.tsv 2>/dev/null | head -30
sh $B probe sabotage2 final_selsabo KSS_KS=32,64 KSS_ROWS=4000; say "probe sabotage rc=$?"
: > $O/finish.done; say done
