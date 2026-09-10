#!/bin/sh
# Build and exercise the opt-in FAST tile4 public AOT extension, then restore
# the installed RF extension. This wrapper owns the public fit smoke.
set -eu
cd "$(dirname "$0")/../.."
if [ "${MOJOLEARN_BUILD_LOCK_HELD:-}" != 1 ]; then
    exec tools/with_build_lock.sh sh ensemble/checks/rf_column_tiles_public.sh
fi
out="$PWD/bench/results/rf_column_tiles_2026-09-09"
mkdir -p "$out" build
module=python/mojolearn/_mojolearn_rf.so
saved=$(mktemp -d "${TMPDIR:-/tmp}/rf-column-public.XXXXXX")
had_module=0
if [ -f "$module" ]; then
    cp "$module" "$saved/original.so"
    had_module=1
fi
restore() {
    if [ "$had_module" = 1 ]; then
        cp "$saved/original.so" "$module"
    else
        rm -f "$module"
    fi
    rm -rf "$saved"
}
trap restore EXIT INT TERM
export MOJOLEARN_SKIP_BUILD_GATE=1
export MOJOLEARN_NUMERIC_MODE=fast
export MOJOLEARN_EXTRA_DEFINES='-D MOJOLEARN_RF_HIST_COLUMNS4=1'
export RF_LAUNCH_LOG="$out/public-fast-columns4.launches.log"
: > "$RF_LAUNCH_LOG"
sh bindings/build_rf.sh > "$out/public-fast-columns4.build.log" 2>&1
cp "$module" build/rf_columns4_fast_public.so
# The stock build smoke copies unrelated local bindings without their runtime
# dylibs. This gate owns equivalent public RF fits with the dev runtime set.
export DYLD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
export PYTHONPATH="$PWD/python${PYTHONPATH:+:$PYTHONPATH}"
.pixi/envs/default/bin/python - <<'PY'
import os
import numpy as np
from mojolearn.randomforest import RandomForestClassifier, RandomForestRegressor
rng = np.random.default_rng(19)
x = rng.random((1031,13), dtype=np.float32)
y = (x[:,0] > .5).astype(np.int32)
kw = dict(n_estimators=3, max_depth=8, n_bins=128, max_features=1.0, random_state=7)
c = RandomForestClassifier(**kw).fit(x,y)
p = c.predict_proba(x)
assert p.shape == (1031,2) and np.allclose(p.sum(axis=1),1)
assert (c.predict(x)==y).mean() > .9
assert np.array_equal(p, RandomForestClassifier(**kw).fit(x,y).predict_proba(x))
r = RandomForestRegressor(**kw).fit(x,x[:,1])
assert np.abs(r.predict(x)-x[:,1]).mean() < .1
from pathlib import Path
trace = Path(os.environ['RF_LAUNCH_LOG']).read_text().splitlines()
count = sum(line.startswith('histogram_binned_columns4_') for line in trace)
assert count > 0, 'AOT smoke did not reach column tile4'
print('PASS public FAST tile4: classifier/regressor public fits and repeat predictions; tiled launches', count)
PY
