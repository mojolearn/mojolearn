# Dedicated GPU measurement next

Latest user priority: IDENTICAL versus competitors on NVIDIA for every
learner. FAST performance is only a MacBook decision-tree target. Use IDENTICAL
for NVIDIA performance runs, including symmetric GBDT against CatBoost GPU.
RF uses cuML RF; ET comparisons require an equivalent learner or explicit
context-only labeling. Internal mode speed ratios are not the success metric.

Choose representative **large real datasets** for training-speed decisions and
performance-default acceptance. HIGGS 1M is an example; include feature/class
counts, bins, depth and device memory pressure when choosing scale, rather than
using a universal row threshold. Record held-out quality and stable end-to-end
fit timing. Small synthetic runs, including the allocation-reuse diagnostic
below, validate correctness or isolate costs; they do not establish a large-data
speed gain or justify a default change without the representative measurement.

RunPod tests are authorized when warranted. Use a bounded single-GPU session
with automatic teardown, separate from the protected Samba training pod.
The last read-only inventory found only that protected pod; see
[availability](../../bench/results/tree_gpu_availability_2026-09-10/README.md).

The following allocation-reuse experiment is a diagnostic subtask; it does not
replace the primary competitor benchmarks.

The immediate comparison is the same sampled learner before/after allocation
reuse, not sampled versus full-feature learning. Build both artifacts on the
same dedicated GPU and toolchain, with matching mode, target and definitions.
Use an isolated reference checkout of the final integrated source, replacing
only these three files with their `e05e889b` versions:

- `gbdt/gpu_data/feature_sampling.mojo`
- `gbdt/methods/doc_parallel_boosting.mojo`
- `gbdt/methods/greedy_subsets_searcher/greedy_search_helper.mojo`

This keeps other integrated changes common to both arms, including hardware
column detection. In particular, current main auto-detects AMD RDNA separately
from CDNA; do not compare different detector revisions and attribute the result
to allocation reuse. Preserve reference sources/diffs and compiled definitions.

Build each mode with `bindings/build_gbdt.sh`; retain the reference extensions
under a separate `{mode}/_mojolearn_gbdt.so` directory before building candidate
extensions. Never replace the environment or checkout of an active training
job. First run the native projection oracle and public feature-fraction checks,
including compiled vendor/mode readback. Retain default and sampled model hashes.

Then, from the candidate checkout on the dedicated device:

```sh
PYTHONPATH=python python checks/gbdt_feature_fraction_ab.py \
  --reference-root /path/to/reference-bindings \
  --expected-vendor cuda --modes identical \
  --rows 8192 65536 262144 --rounds 5 \
  --output /path/to/results/sampled_reuse_ab.json
```

Use `hip` for AMD identity qualification. Explicitly select IDENTICAL for
NVIDIA timing; the driver otherwise defaults to FAST/IDENTICAL. It covers all
three growth policies, with fractions 0.5/0.25. It warms both arms, alternates order, checks
complete model and prediction hashes every fit, and records raw whole-fit
seconds, both timing spreads, source data and binary hashes. It does not time
prediction/hash construction as training. These are synthetic profiles; follow
with the application's real shapes/data and a memory-usage trace before a broad
speed claim. Reject noisy cells, rather than repeating until a win appears.

Packing still runs once per sampled tree. Next profile time in projection,
metadata refresh/allocation, histogram construction and synchronization. Reusing
arenas can help without modifying histogram numerics; a mapped original-layout
histogram is a separate candidate. Preserve original feature IDs, sampled masks,
row order and split ties in either implementation.

Full-feature versus sampled timing is a separate learning tradeoff: include
heldout quality and tree size. It cannot establish identical-output speedup.

## Focused NVIDIA competitor runner

`bench/speed/nvidia_identical_trees.py` reuses the existing dataset loaders,
learner configurations, competitor constructors, scoring and timed runner.
It requires explicit IDENTICAL/CUDA selection and verifies the native mode and
vendor getters before timing. RF now exports `rf_numeric_mode()`; a directory
name alone is not compiled-mode evidence.

On a separate, automatically expiring pod, set up with
`MOJOLEARN_TREES_SKIP_LIGHTGBM_CUDA=1 sh tools/trees_identical_remote.sh` for an
RF/cuML and symmetric GBDT/CatBoost session. Wait until setup finishes before
collecting performance samples. Then run under an external timeout:

```sh
MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SPEED_EXPECTED_VENDOR=cuda \
PYTHONPATH=python timeout -k 15 1260 python3 -u \
  bench/speed/nvidia_identical_trees.py --lane rf --dataset higgs \
  --rows 1000000 --rounds 5 --output /root/trees_out/rf_higgs_1m.json
```

RF includes cuML default streams and a separately named one-stream arm.
`--lane gbdt-symmetric` selects CatBoost GPU only. `--lane et` records our
baseline without inventing an equivalent GPU competitor. Missing real datasets
fail instead of falling back to synthetic data. `--size smoke --dataset synthclf
--rows 32768 --rounds 1` exercises the runner but provides no performance or
independent-repeatability conclusion.

The fit timer includes construction, host packing, training and completion;
scoring/model hashing is outside it. Already-packed input is refused for this
particular comparison. JSON records native binding/source/data hashes, library
versions, parameters, raw samples, spread, quality, and repeated model/prediction
witnesses. Keep the companion stdout logs as well. This is a full-fit baseline,
not an isolated inference benchmark or cross-vendor identity qualification.


## Diagnose timing variation before another speed claim

The focused runner accepts `--nvtx` to label each constructor/fit/completion
region as `mojolearn-fit/<lane>/<arm>/round-<n>` (round 0 is warm-up).
Scoring and full model hashing remain outside these ranges. An exception closes
the range before the runner records the refusal. This optional instrumentation
changes no learner dispatch, stream policy, RNG, or product defaults.

On a dedicated NVIDIA device with Nsight Systems installed, run a separate
**diagnostic** large-data pass, for example:

```sh
MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SPEED_EXPECTED_VENDOR=cuda \
PYTHONPATH=python timeout -k 15 300 nsys profile --trace=cuda,nvtx,osrt \
  --force-overwrite=true -o /root/trees_out/symmetric_higgs_trace \
  python3 -u bench/speed/nvidia_identical_trees.py \
  --lane gbdt-symmetric --dataset higgs --rows 1000000 --rounds 2 --nvtx \
  --output /root/trees_out/symmetric_higgs_trace.json
```

Inspect CPU launch gaps, allocation/transfer time, kernel durations and device
synchronization inside each arm's ranges. Correlate with clock/power/temperature
telemetry and CPU/NUMA placement; do not infer a cause from one-second GPU
utilization alone. The JSON captures CPU affinity and common thread environment
settings even without profiling. Retain the trace, logs, tool versions and raw
samples. **Profiled timings are diagnostic:** repeat the matched comparison
without `nsys` or `--nvtx`, with at least five rounds, for performance acceptance.
NVTX integration has host control-flow checks; its device trace still needs
validation on NVIDIA with the profiling tool installed.
