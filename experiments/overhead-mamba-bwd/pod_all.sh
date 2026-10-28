#!/bin/bash
# Lane overhead-mamba-bwd self-merge leg on one CPU pod: (1) the integration
# pod smoke over the merged tree (all host families through the binary cache,
# import + tier_dir probe), (2) the before/after same-bits and same-refusals
# fixture, (3) the two scripted gates. Exit codes land in $LEG_OUT/lane.rc.
set -u
OUT=${LEG_OUT:?}
bash experiments/overhead-mamba-bwd/pod_smoke_5b.sh > "$OUT/smoke.out" 2>&1
echo "smoke_rc=$?" >> "$OUT/lane.rc"
bash experiments/overhead-mamba-bwd/pod_fixture.sh > "$OUT/fixture.out" 2>&1
echo "fixture_script_rc=$?" >> "$OUT/lane.rc"
pixi run -e default python tools/tests/test_no_python_runtime.py > "$OUT/gate-no-python.log" 2>&1
echo "gate_no_python_rc=$?" >> "$OUT/lane.rc"
pixi run -e default python tools/tests/test_tombstones_preserved.py > "$OUT/gate-tombstones.log" 2>&1
echo "gate_tombstones_rc=$?" >> "$OUT/lane.rc"
cat "$OUT/lane.rc"
