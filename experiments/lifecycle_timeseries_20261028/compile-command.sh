#!/bin/bash
# lifecycle-timeseries compile: xseq dispatcher + TSA dispatcher roots, Mac CPU column, identical, O0, -j 10
set -u
W=/Users/andrewhendel/mojolearn-wt/lifecycle-timeseries-20261028; E=/Users/andrewhendel/mojolearn-evidence/lifecycle-timeseries-20261028; M=/Users/andrewhendel/CascadeProjects/mojolearn
cd $W
for pair in "xseq:checks/peer6_forecast_convenience_compile.mojo:_peer6_forecast_compile.so" "tsa:checks/peer6_autoarima_lifecycle_compile.mojo:_peer6_autoarima_compile.so"; do
  IFS=: read name entry out <<< "$pair"
  start=$(date +%s)
  bash $HOME/mojolearn-evidence/compile_slot.sh pixi run --manifest-path $M/pixi.toml --environment default --locked --no-install \
    --executable $M/.pixi/envs/default/bin/mojo build --optimization-level 0 -j 10 --emit shared-lib --target-cpu apple-m1 \
    -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_CPU -I $W -I $W/bindings $W/$entry -o $E/$out > $E/compile_$name.log 2>&1
  rc=$?
  echo "$name rc=$rc seconds=$(($(date +%s)-start)) sha=$(shasum -a 256 $E/$out 2>/dev/null | cut -c1-64)" >> $E/compile_summary.txt
done
