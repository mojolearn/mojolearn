#!/bin/sh
# usage: run.sh <script> <binding .mojo> <arm name> [closure file to break]
set -u
SP=/private/tmp/claude-501/-Users-andrewhendel-CascadeProjects-mojolearn/4e8829df-5da1-4c85-8680-6b52df16132c/scratchpad
W=$SP/wt-bincache; D=$SP/bincache/closure/tree-$3
MOJO=/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/default/bin/mojo
python3 $SP/bincache/closure/prep.py "$W" "$D" "$1" ${4:-} || exit 9
cd "$D" || exit 9
t0=$(date +%s)
nice -n 19 env OMP_NUM_THREADS=1 MOJOLEARN_CPU_THREADS=1 pixi run --frozen --manifest-path /Users/andrewhendel/CascadeProjects/mojolearn/pixi.toml mojo build -j 1 --emit shared-lib --target-cpu apple-m1 \
  ${DEFS:--D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_CPU} -I . -I bindings "$2" -o "$SP/bincache/closure/$3.so" > "$SP/bincache/closure/$3.log" 2>&1
rc=$?
echo "arm=$3 exit=$rc seconds=$(( $(date +%s) - t0 ))"
grep -m3 -E "error|@@@" "$SP/bincache/closure/$3.log" | cut -c1-220
