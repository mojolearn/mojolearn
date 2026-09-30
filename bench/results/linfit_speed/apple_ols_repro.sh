#!/bin/bash
# lane/linfit-speed: glm/checks/ols_check.mojo returned 0.0 once on the m2pro
# (single-column check). Run the check binary REPS times, count failures by
# check, and read the system log for Metal command-buffer errors.
set -uo pipefail
REPS=${REPS:-30}
T=$(mktemp -d)
start=$(date -u +"%Y-%m-%d %H:%M:%S")
sh tools/with_identical_mode.sh pixi run mojo build -I . glm/checks/ols_check.mojo -o $T/ols_check.bin > $T/build.log 2>&1 || { tail -20 $T/build.log; exit 1; }
fails=0
for r in $(seq 1 $REPS); do
  if ! $T/ols_check.bin > $T/run$r.log 2>&1; then
    fails=$((fails + 1))
    echo "RUN $r FAIL: $(grep -m1 'Unhandled exception\|Error' $T/run$r.log | cut -c1-240)"
  fi
done
echo "OLS-REPRO fails=$fails of $REPS"
log show --start "$start" --predicate 'eventMessage CONTAINS[c] "command buffer" OR eventMessage CONTAINS[c] "Impacting Interactivity" OR eventMessage CONTAINS[c] "GPU Timeout" OR eventMessage CONTAINS[c] "kIOGPUCommandBuffer"' 2>/dev/null | tail -20
echo "LOG-END"
