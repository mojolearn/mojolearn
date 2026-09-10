# Dedicated GPU measurement next

Latest user priority: IDENTICAL versus competitors on NVIDIA for every
learner. FAST performance is only a MacBook decision-tree target. Use IDENTICAL
for NVIDIA performance runs, including symmetric GBDT against CatBoost GPU.
RF uses cuML RF; ET comparisons require an equivalent learner or explicit
context-only labeling. Internal mode speed ratios are not the success metric.

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
