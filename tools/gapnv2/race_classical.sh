#!/bin/bash
# lane/gap-nv-classical2: race classical/classical2 lanes (bench_board_more.py, classical_two_datasets.py)
# in this tree on the box's board data. Usage (lq CMD): bash tools/gapnv2/race_classical.sh lane:ds[:VENV=x] ...
# Prints one SUMMARY line per lane x dataset with every round's ms and digest.
cd "$(dirname "$0")/../.."
R=$PWD
for b in /root/board-0834 /root/board-0833; do [ -d $b/cache ] && { B=$b; break; }; done
PY=$B/cache/venv/bin/python
O=$(mktemp -d /root/gapnv2-race.XXXX)
for spec in "$@"; do
  case $spec in *:*) ;; *) continue;; esac
  IFS=: read -r lane ds envs <<< "$spec"
  case $lane in
    hdbscan) cmd="tools/classical_two_datasets.py race --lane $lane --dataset $ds --data $B/cache/ctd-data/rows-full --out $O/out-$lane-$ds --work $O/work --root $R --arms ours --rounds 2 --round-seconds 300 --warmup-seconds 600 --ours-python $PY --theirs-python $PY";;
    *) cmd="tools/bench_board_more.py race --lane $lane --dataset $ds --data $B/cache/more-data/rows-full --arms ours --rounds 2 --out $O/out-$lane-$ds --work $O/work";;
  esac
  env ${envs//,/ } MOJOLEARN_BENCH_INSTALLED=0 PYTHONPATH=$R/python timeout 2400 $PY -u $cmd > $O/$lane-$ds.log 2>&1
  echo "SUMMARY $lane $ds ${envs:-default} rc=$? $(grep -o 'ms=[0-9.]* .*digest=[0-9a-f]*' $O/$lane-$ds.log | sed 's/infer_ms=[^ ]* //' | tr '\n' ' ') $(grep -m1 -E 'Error|error:' $O/$lane-$ds.log | cut -c1-200)"
done
rm -rf $O/work
