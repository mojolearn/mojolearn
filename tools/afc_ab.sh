#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# afc_ab.sh <tag> <lane> <dataset> <reps> <rounds> <envA> [<envB>]
#
# Alternating A/B of the FAST tier on an Apple box, one build serving both
# arms: arm A runs with the space-separated VAR=VALUE list envA, arm B with
# envB ("-" = no extra env), alternated A, B, A, B ... reps times, each a
# `bench_board_algos.py race --arms ours-fast` of <rounds> timed rounds at the
# board shape (rows-full). Prints one AFC-AB line per run and a summary of the
# median of each arm's run medians, its digests and its quality.
# AFC_ARM=ours races the IDENTICAL tier instead. With no envB, only arm A runs (a baseline). Output under ~/mq/out/race-<tag>/
# so `lq log <box> <tag>` greps it. Run inside a built tree (FAST bindings).
set -u
TAG=$1 LANE=$2 DS=$3 REPS=$4 ROUNDS=$5 EA=$6 EB=${7:-}
for b in board-0834 board-0833; do [ -d $HOME/$b/cache/algos-data/rows-full ] && { B=$HOME/$b; break; }; done
VP=$B/cache/venv/bin/python
# AFC_FAMILY: algos (default, tools/bench_board_algos.py), classical2
# (tools/bench_board_more.py) or classical (tools/classical_two_datasets.py)
FAM=${AFC_FAMILY:-algos}
case $FAM in
  algos) DRV=tools/bench_board_algos.py; PFX=ALGOS; DATA=$B/cache/algos-data/rows-full; XARGS= ;;
  classical2) DRV=tools/bench_board_more.py; PFX=MORE
     DATA=$(find $B -maxdepth 4 -type d -name more-data 2>/dev/null | head -1)/rows-full; XARGS= ;;
  classical) DRV=tools/classical_two_datasets.py; PFX=CTD
     DATA=$(find $B -maxdepth 4 -type d -name ctd-data 2>/dev/null | head -1)/rows-full; XARGS="--root $PWD" ;;
esac
OUT=$HOME/mq/out/race-$TAG; mkdir -p $OUT
LOG=$OUT/race.log
arms="A"; [ -n "$EB" ] && arms="A B"
for r in $(seq 1 $REPS); do
  for a in $arms; do
    E=$EA; [ $a = B ] && E=$EB; [ "$E" = - ] && E=
    d=$OUT/$a-$r; rm -rf $d; mkdir -p $d
    env $E MOJOLEARN_BENCH_INSTALLED=0 PYTHONPATH=$PWD/python timeout 3600 \
      $VP $DRV race --lane $LANE --dataset $DS --data $DATA --arms ${AFC_ARM:-ours-fast} \
      --rounds $ROUNDS --out $d/res --work $d/work $XARGS > $d/race.txt 2>&1
    rm -rf $d/work
    m=$(grep -o "$PFX lane=.*" $d/race.txt | grep -o 'median_ms=[0-9.]*' | tail -1 | cut -d= -f2)
    st=$(grep -o "$PFX lane=.*" $d/race.txt | grep -o 'status=[A-Za-z_]*' | tail -1 | cut -d= -f2)
    q=$(grep -o "$PFX lane=.*" $d/race.txt | grep -o 'quality=.*' | tail -1 | cut -c1-200)
    g=$(grep -o 'digest=[0-9a-f]*' $d/race.txt | tail -1 | cut -d= -f2 | cut -c1-16)
    echo "AFC-AB $TAG arm=$a rep=$r lane=$LANE ds=$DS status=${st:-none} median_ms=${m:-none} digest=${g:-none} env='$E' $q" | tee -a $LOG
    [ -z "$m" ] && grep -E -m 5 'Error|error|REFUSED|Traceback' $d/race.txt | cut -c1-300 | tee -a $LOG
  done
done
for a in $arms; do
  ms=$(grep "AFC-AB $TAG arm=$a " $LOG | grep -o 'median_ms=[0-9.]*' | cut -d= -f2 | sort -n | tr '\n' ' ')
  md=$(echo $ms | tr ' ' '\n' | grep . | awk '{v[NR]=$1} END{if(NR==0)print "none"; else if(NR%2)print v[(NR+1)/2]; else print (v[NR/2]+v[NR/2+1])/2}')
  dg=$(grep "AFC-AB $TAG arm=$a " $LOG | grep -o 'digest=[0-9a-f]*' | sort -u | tr '\n' ' ')
  echo "AFC-AB-SUMMARY $TAG arm=$a lane=$LANE ds=$DS median_of_medians_ms=$md runs=[$ms] $dg" | tee -a $LOG
done
