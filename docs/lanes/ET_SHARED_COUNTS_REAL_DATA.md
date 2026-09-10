# Extra Trees shared counts: real-data A/B

`extratrees/bench/shared_score_ab.py` retains the small synthetic diagnostic and
adds public-fit measurements on the shared HIGGS/covertype loaders. This is an
ET-versus-ET optimization comparison; cuML RandomForest is not a matched ET
competitor. No H100 speed claim or default change follows from this harness.

Stage data before opening the timing window. The driver uses
`tools/speed_gbdt_arm.py:load_dataset`, including its unchanged splits/caps and
`GBM_BENCH_DATA` cache for HIGGS. Covertype uses sklearn's dataset cache.
HIGGS retains its 500,000-row test tail. Actual train/test shapes, input hashes,
class vocabulary and shared `dataset_scale` classification are recorded.
Small datasets remain runnable with an explicit reminder; they are useful for
smoke/correctness checks, not sufficient evidence for a large-data default.

## Build and qualify explicit artifacts on the existing H100

Run in the repository with its configured Mojo/Python environment. These
commands build locally on an already provisioned machine; they do not acquire
or extend resources. Keep build logs and source provenance with the artifacts.
The build script writes the normal IDENTICAL binding, so copy each arm before
building the next. The benchmark itself never replaces an installed binding.

```sh
mkdir -p /tmp/et-shared-artifacts/baseline /tmp/et-shared-artifacts/candidate
env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=nvidia \
  MOJOLEARN_GPU_ARCHS=sm_90 \
  MOJOLEARN_EXTRA_DEFINES='-D MOJOLEARN_ET_NO_SHARED_CLASS_COUNTS=1' \
  tools/with_build_lock.sh sh bindings/build_trees.sh \
  > /tmp/et-shared-artifacts/baseline/build.log 2>&1
cp python/mojolearn/identical/_mojolearn_trees.so /tmp/et-shared-artifacts/baseline/
env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=nvidia \
  MOJOLEARN_GPU_ARCHS=sm_90 \
  MOJOLEARN_EXTRA_DEFINES='-D MOJOLEARN_ET_SHARED_CLASS_COUNTS=1' \
  tools/with_build_lock.sh sh bindings/build_trees.sh \
  > /tmp/et-shared-artifacts/candidate/build.log 2>&1
cp python/mojolearn/identical/_mojolearn_trees.so /tmp/et-shared-artifacts/candidate/
```

The force-off definition takes precedence over opt-in; do not pass both.
The real driver checks the loaded artifact's own numeric mode, vendor and
shared-count mask (0 baseline, 15 candidate). Filename/environment labels are
not accepted as proof. Artifact SHA256 is checked again by each worker.

The existing independent gate remains:

```sh
tools/with_build_lock.sh bash extratrees/tools/check_shared_score.sh \
  bench/results/h100-et-shared-gate
```

That gate compares 27 native complete-forest/probability fingerprints in each
mode and runs the independent score-cell oracle/sabotages. Its synthetic
fixtures cover accumulator widths, bootstrap, best-first, Gini and entropy;
large real timing does not replace those checks.

## Representative timing

Use a new output directory for every run. For HIGGS:

```sh
tools/with_build_lock.sh pixi run python extratrees/bench/shared_score_ab.py \
  bench/results/h100-et-shared-higgs-1m \
  --dataset higgs --rows 1000000 --trees 100 --depth 16 \
  --baseline-binding /tmp/et-shared-artifacts/baseline/_mojolearn_trees.so \
  --candidate-binding /tmp/et-shared-artifacts/candidate/_mojolearn_trees.so \
  --mode identical --vendor cuda --warmups 2 --repeats 3
```

Repeat with `--dataset covtype --rows 581012` to cover seven real classes
(binary HIGGS alone does not qualify multiclass shared-count behavior).
Covertype is naturally smaller and receives the shared size reminder; retain
that qualification alongside its results. `--dataset covtype2` isolates its
binary variant. A reduced real-data smoke can use `--rows 10000 --trees 3
--depth 4 --warmups 1 --repeats 1`; it is allowed, but has insufficient samples
for timing qualification. Default settings use Gini, `max_features='sqrt'`,
no bootstrap and seed `0xACC2021`; `--max-features all`, `--criterion entropy`
and `--bootstrap` explicitly expose other workload choices.

ABBA passes run baseline/candidate in separate fresh processes. Each process
warms up its own artifact and then retains all requested fit samples. A second
`--cycles 2` cycle reverses the order to BAAB. Full public `fit` is timed,
including host input packing, GPU training and model export. No profiling
synchronizations are added inside fit. Data parsing, binary loading, warmups,
model hashing, prediction and quality calculations are excluded. Input arrays
are prepared once and memory-mapped by workers; their bytes are verified before
each pass. This requires disk space for one Float32 copy of train/test inputs.

Every fit, including warmups, hashes all exported prediction arrays, classes
and fit metadata. It also predicts **every train and test row**, in bounded
chunks, and hashes every probability bit. Model and probability hashes must
agree within each arm and across both arms. This can make validation much
longer than the measured fit. Existing ET prediction traversal is used for
verification; this is not a claim that prediction itself is GPU accelerated.
Accuracy is computed outside timing as benchmark validation, not a product CPU
metric backend. Native gate fingerprints additionally cover internal node
fields not exported by the public binding.

`config.json` records source/script hashes, Git state or `SHIPPED_COMMIT.txt`
for archive deployments, artifact hashes, exact parameters, data hashes and
`nvidia-smi`. Per-pass JSON/log files retain warmup and measured samples,
compiled witnesses, exact fingerprints and accuracy. `timing.json` reports
median/min/max/standard deviation/MAD, pass-median drift, max/min sample
spread, pass order and sample counts. `qualified_speedup` is null unless both
arms have at least five measured samples, pass drift at most 10%, and sample
spread at most 10%. The observed ratio remains visible even when noisy;
workload-size reminders remain independent of statistical stability.

The build lock spans the run and the driver additionally takes the shared
benchmark lock. Ensure the existing device is otherwise idle and retain the
rental owner's deadline; this driver neither provisions hardware nor manages
leases. Never use the synthetic diagnostic alone to justify a production
optimization/default change.
