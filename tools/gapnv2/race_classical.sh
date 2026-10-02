#!/bin/bash
# lane/gap-nv-classical2: the classical-family lanes (bench_board_more.py: lasso, elasticnet;
# classical_two_datasets.py: hdbscan) raced in THIS tree on the box's board data. No harness patch needed.
#   bash tools/gapnv2/race_classical.sh speed lane:ds ...   device, full rows, 2 rounds -> SPEED lines
#   bash tools/gapnv2/race_classical.sh id    lane:ds ...   device + host column on a 50k-row copy -> IDCHECK lines
# Builds any binding it needs that is not already in the tree (device + host).
cd "$(dirname "$0")/../.."
R=$PWD; MODE=$1; shift
for b in /root/board-0834 /root/board-0833 $HOME/board-0834 $HOME/board-0833; do
  [ -d $b/cache/more-data/rows-full ] && { B=$b; break; }; done
[ -n "${B:-}" ] || { echo "GAPNV2 ERROR no board more-data on this box"; exit 1; }
PY=$B/cache/venv/bin/python; [ -x $PY ] || PY=$R/.pixi/envs/default/bin/python
[ -x $PY ] || PY=$(command -v python3)
need() {  # need <family> : device + host build if missing
  if ! ls $R/python/mojolearn/identical/_mojolearn_$1.so $R/python/mojolearn/_mojolearn_$1.so 2>/dev/null | grep -q .; then
    MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-4} bash bindings/build_$1.sh > /tmp/gapnv2-build-$1.log 2>&1; echo "GAPNV2 built $1 rc=$?"; fi
  if [ "$MODE" = id ] && [ -f bindings/build_${1}_host.sh ] && ! ls $R/python/mojolearn/*/_mojolearn_${1}_host.so $R/python/mojolearn/_mojolearn_${1}_host.so 2>/dev/null | grep -q .; then
    env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_TARGET_COLUMN=cpu bash bindings/build_${1}_host.sh > /tmp/gapnv2-build-$1-host.log 2>&1; echo "GAPNV2 built ${1}_host rc=$?"; fi
}
O=$(mktemp -d ${TMPDIR:-/tmp}/gapnv2-race.XXXX)
small() {  # small <kind>: a 50k-row copy of rows-full (same bits on every box)
  local d=$O/$1-small; [ -d $d ] || $PY tools/gapnv2/make_small_rows.py $B/cache/$1/rows-full $d 50000 > /dev/null 2>&1; echo $d; }
dig() { grep -o 'digest=[0-9a-f]*' $1 2>/dev/null | tail -1 | cut -d= -f2 | cut -c1-16; }
run() {  # run <lane> <ds> <datadir> <tag> [env...]
  local lane=$1 ds=$2 data=$3 tag=$4; shift 4
  case $lane in
    hdbscan) cmd="tools/classical_two_datasets.py race --lane $lane --dataset $ds --data $data --out $O/out-$tag --work $O/work --root $R --arms ours --rounds ${ROUNDS:-2} --round-seconds 600 --warmup-seconds 900 --ours-python $PY --theirs-python $PY";;
    *) cmd="tools/bench_board_more.py race --lane $lane --dataset $ds --data $data --arms ours --rounds ${ROUNDS:-2} --out $O/out-$tag --work $O/work --ours-python $PY --theirs-python $PY";;
  esac
  env "$@" MOJOLEARN_BENCH_INSTALLED=0 PYTHONPATH=$R/python timeout 3000 $PY -u $cmd > $O/$tag.log 2>&1
}
for spec in "$@"; do
  case $spec in *:*) ;; *) continue;; esac
  lane=${spec%%:*}; ds=${spec#*:}
  case $lane in hdbscan) fam=hdbscan kind=ctd-data;; *) fam=solver kind=more-data;; esac
  if ! ls $R/python/mojolearn/identical/_mojolearn.so $R/python/mojolearn/_mojolearn.so 2>/dev/null | grep -q .; then
    MOJOLEARN_COMPILE_JOBS=${MOJOLEARN_COMPILE_JOBS:-4} bash bindings/build.sh > /tmp/gapnv2-build-base.log 2>&1; echo "GAPNV2 built base rc=$?"; fi
  if [ "$MODE" = id ] && ! ls $R/python/mojolearn/*/_mojolearn_core_host.so $R/python/mojolearn/_mojolearn_core_host.so 2>/dev/null | grep -q .; then
    env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_TARGET_COLUMN=cpu bash bindings/build_core_host.sh > /tmp/gapnv2-build-core-host.log 2>&1; echo "GAPNV2 built core_host rc=$?"; fi
  need $fam
  if [ "$MODE" = speed ]; then
    run $lane $ds $B/cache/$kind/rows-full $lane-$ds
    echo "SPEED $lane $ds $(grep -o 'ms=[0-9.]*' $O/$lane-$ds.log | grep -v infer | tr '\n' ' ') digest=$(dig $O/$lane-$ds.log) $(grep -m1 -E 'Error|error:' $O/$lane-$ds.log | cut -c1-200)"
  else
    sd=$(small $kind)
    ROUNDS=1 run $lane $ds $sd $lane-$ds-dev
    ROUNDS=1 run $lane $ds $sd $lane-$ds-host MOJOLEARN_VENDOR=cpu
    g=$(dig $O/$lane-$ds-dev.log); h=$(dig $O/$lane-$ds-host.log)
    st=DIFFER; [ -n "$g" ] && [ "$g" = "$h" ] && st=MATCH; [ -z "$g" ] && st=ERROR
    echo "IDCHECK $lane $ds rows=small device=${g:-none} host=${h:-none} $st $( [ $st = ERROR ] && grep -h -m1 -E 'Error|error:' $O/$lane-$ds-dev.log | cut -c1-200)"
  fi
done
rm -rf $O/work
echo "GAPNV2 done $MODE"
