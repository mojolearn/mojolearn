# RF column-tile tuning on NVIDIA

`bench/speed/rf_higgs_columns_ab.py` supplies the missing large real-data public
fit comparison. The earlier native tile driver uses synthetic data; the earlier
public gate only exercised FAST tile4. This driver requires NVIDIA IDENTICAL and real HIGGS. Performance evidence uses
training rows >=1 million and >=5 measured fits per arm. Smaller runs warn and
are marked smoke-only, with timing_valid=false. No production
flag or default changes. The candidate is experimental, with no speed claim.

Prepare three **same-source, same-toolchain, same-GPU** IDENTICAL RF bindings,
with all other build options equal. Preserve the installed reference before
building candidates; retain each build log and source commit alongside these
binaries. On the authorized NVIDIA machine, from repository root:

```sh
# Run the whole preparation under tools/with_build_lock.sh.
# Existing base/GBDT bindings and the shared HIGGS cache must be available.
export MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_GPU_ARCHS=sm_90  # H100; use the actual GPU architecture.
export MOJOLEARN_SKIP_BUILD_GATE=1
mkdir -p build/rf-higgs-columns/{reference,columns2,columns4}
for arm in reference columns2 columns4; do
  export MOJOLEARN_EXTRA_DEFINES=''
  if [ "$arm" = columns2 ]; then
    export MOJOLEARN_EXTRA_DEFINES='-D MOJOLEARN_RF_HIST_COLUMNS2=1'
  elif [ "$arm" = columns4 ]; then
    export MOJOLEARN_EXTRA_DEFINES='-D MOJOLEARN_RF_HIST_COLUMNS4=1'
  fi
  sh bindings/build_rf.sh > "build/rf-higgs-columns/$arm/build.log" 2>&1 || exit
  cp python/mojolearn/identical/_mojolearn_rf.so "build/rf-higgs-columns/$arm/"
done
cp build/rf-higgs-columns/reference/_mojolearn_rf.so python/mojolearn/identical/
unset MOJOLEARN_EXTRA_DEFINES MOJOLEARN_SPEED_FORTRAN RF_LAUNCH_LOG
# Require the existing real HIGGS cache; this driver never downloads/falls back.
CUDA_VISIBLE_DEVICES=0 MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python \
  timeout 1800 tools/with_build_lock.sh python3 bench/speed/rf_higgs_columns_ab.py \
  --bindings build/rf-higgs-columns --rows 1000000 --rounds 6 \
  --output bench/results/rf_higgs_columns_nvidia_unique_run
```

The default workload is 100 trees, depth16, 128 bins, sqrt features, bootstrap,
seed7, and public default streams, using the shared RF benchmark configuration.
Each binary must expose `rf_numeric_mode()==1` and `rf_vendor()=='cuda'`.
Separate one-tree processes verify actual tile2/tile4 launch paths, and absence
of tiled launches in the reference. This is stronger than trusting a binary
filename. These probes are reachability checks, not performance samples.
RF launch logging latches its environment at first use, so probe processes
must stay separate from timing. No native tile-definition getter exists yet;
actual route checks and retained build definitions supply that witness.

All timing arms load in one process and use the existing public RF Arm adapters.
A temporary class binding override survives public fit's configuration refresh;
this driver is single-threaded and restores it after each fit/score call. Fits
include construction, host packing, upload, the whole forest, and synchronization.
Model hashing and held-out quality/prediction checks are outside the timer.
Every fit must produce identical full model arrays and complete native Float32
two-column held-out probability bits across all arms. The full probability call
runs once outside fit timing. A cached-prediction proxy supplies that same
probability array to the existing shared quality scorer, avoiding a second
native traversal while preserving its metric definitions. Six balanced
orders (ABC,CBA,BCA,ACB,CAB,BAC) place each arm twice in every position; shorter
or incomplete cycles retain an order imbalance. One warmup per arm is excluded. Raw `fits.jsonl` is written incrementally, plus a final summary with
binary/data/driver hashes, argv, actual order, quality, medians and per-arm max/min spread. A spread >1.10
marks timing invalid; no performance promotion follows from an invalid run.
A successful correctness exit does not imply valid performance. No external
competitor is included: this is internal candidate tuning, not cuML parity.

Only Python syntax and mocked binding-route checks were run when preparing this
driver. The real NVIDIA workload remains unexecuted until separately authorized.

Each fit records `prediction_ms` for the single public probability call and
`verification_ms` for all post-fit checks (including that prediction), separately
from `fit_ms`. These quantify verification cost without charging it to training.
