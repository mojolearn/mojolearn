#!/bin/sh
set -u
SP=/private/tmp/claude-501/-Users-andrewhendel-CascadeProjects-mojolearn/4e8829df-5da1-4c85-8680-6b52df16132c/scratchpad; W=/private/tmp/claude-501/-Users-andrewhendel-CascadeProjects-mojolearn/4e8829df-5da1-4c85-8680-6b52df16132c/scratchpad/wt-bincache; EV=/private/tmp/claude-501/-Users-andrewhendel-CascadeProjects-mojolearn/4e8829df-5da1-4c85-8680-6b52df16132c/scratchpad/wt-bincache/bench/results/bincache_2026-09-15
cd $W/tools
echo "# python3 -m unittest -v test_bincache   (M4, one core, shared machine, $(date -u +%FT%TZ), tree $(git -C $W rev-parse --short HEAD) + working changes)" > $EV/unit_tests.txt
nice -n 19 python3 -m unittest -v test_bincache >> $EV/unit_tests.txt 2>&1
python3 $SP/bincache/make_arms.py $W/tools/bincache.py $SP/bincache/sab > $EV/sabotage_arms.txt
for arm in no_sha_check image_not_keyed no_sabotage_refusal jobs_not_keyed; do
  cp $W/tools/test_bincache.py $SP/bincache/sab/$arm/
  echo "== arm $arm (a copy of tools/bincache.py with one guard removed; the suite must FAIL)" >> $EV/sabotage_arms.txt
  (cd $SP/bincache/sab/$arm && nice -n 19 python3 -m unittest test_bincache 2>&1 | grep -E "^(FAIL|ERROR):|^FAILED|^OK|^Ran") >> $EV/sabotage_arms.txt
done
{
echo "# every .mojo OUTSIDE the binding's computed closure replaced by a garbage line, then a real mojo build on the M4 (-j 1)"
echo "# arm	exit	seconds	(expected: outside-garbage arms exit 0; break-inside arms exit 1)"
for spec in "bindings/build_preprocessing_host.sh|bindings/_mojolearn_preprocessing_host.mojo|prep-host-outside-garbage||"             "bindings/build_preprocessing_host.sh|bindings/_mojolearn_preprocessing_host.mojo|prep-host-break-numerics|checks/numerics.mojo|"             "bindings/build_preprocessing_host.sh|bindings/_mojolearn_preprocessing_host.mojo|prep-host-break-hostptr|bindings/hostptr.mojo|"             "bindings/build_hdbscan_host.sh|bindings/_mojolearn_hdbscan_host.mojo|hdbscan-host-outside-garbage||"             "bindings/build_linalg.sh|bindings/_mojolearn_linalg.mojo|linalg-metal-outside-garbage||gpu"             "bindings/build_metrics.sh|bindings/_mojolearn_metrics.mojo|metrics-metal-outside-garbage||gpu"             "bindings/build_gbdt.sh|bindings/_mojolearn_gbdt.mojo|gbdt-metal-outside-garbage||gpu"; do
  s=$(echo "$spec" | cut -d'|' -f1); m=$(echo "$spec" | cut -d'|' -f2); a=$(echo "$spec" | cut -d'|' -f3); br=$(echo "$spec" | cut -d'|' -f4); g=$(echo "$spec" | cut -d'|' -f5)
  if [ "$g" = gpu ]; then DEFS="-D MOJOLEARN_NUMERIC_IDENTICAL=1"; export DEFS; else unset DEFS; fi
  sh $SP/bincache/closure/run.sh "$s" "$m" "$a" $br
done
} > $EV/closure_compiles.txt 2>&1
sh $SP/bincache/e2e/run.sh > $EV/r2_end_to_end.txt 2>&1
(cd $W && sh tools/bincache_leg.sh selftest) >> $EV/r2_end_to_end.txt 2>&1
