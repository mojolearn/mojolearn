# extratrees: the histogram-free Extremely Randomized Trees learner

**This directory mirrors no incumbent GPU source for its SPLIT RULE, and it
mirrors cuML file for file for everything else.** Both halves of that sentence
are load-bearing, so they are stated separately.

## The split rule is a PORT FROM PAPER

The algorithm is Extremely Randomized Trees in its original, histogram-free
formulation:

> Geurts, Ernst & Wehenkel, *Extremely randomized trees*, Machine Learning
> 63(1):3-42, 2006. `Split_a_node` / `Pick_a_random_split` are the two
> procedures being implemented (paper section 2.1, Table 1).

Per node, per candidate feature, ONE real-valued threshold is drawn uniformly
inside that feature's [min, max] over the node's rows, the induced partition is
scored, and the best feature wins. There are no global quantiles, no bin index
and no histogram anywhere in this directory.

The reference implementation of that procedure — the thing whose branches are
transcribed rather than paraphrased — is **scikit-learn's `RandomSplitter`**,
pinned at tag `1.9.0` (`77def0e`) in
`~/CascadeProjects/upstream/scikit-learn`, and identical byte for byte to the
`.pyx` files shipped in this repo's `bench` pixi environment (verified by
`diff` on 2026-08-21 for `_splitter.pyx`, `_partitioner.pyx`, `_criterion.pyx`).
Every branch we take cites `sklearn/tree/_splitter.pyx:<line>`.

**Nobody ships this on a GPU** (re-verified 2026-08-21). The only two
ExtraTrees-on-GPU artifacts in existence are LightGBM's `USE_RAND` CUDA kernel,
which draws a random BIN in histogram space and is therefore a different
algorithm, and cuML's FEA #8133, which is an unlanded design. So there is no
`.cu` file to be faithful to for the split rule, and the house rule that
forbids invention is satisfied the only way it can be: **the paper and
sklearn's `.pyx` are the upstream, cited by line, and every place the paper
under-specifies is a numbered deviation in `DEVIATIONS.md`** (this lane's
reserved range is 130-159).

## Everything around the split rule MIRRORS cuML

A random split rule does not need a new tree builder, and inventing one would
be exactly the mistake `PORTING_RULES.md` 0c catalogues eleven times. So the
control plane, the node queue, the split record, the tie-break, the flat node
layout, the feature sampling and the RNG keying are **ports of cuML**, pinned
at `00094f7` in `~/CascadeProjects/upstream/cuml`, mirrored file for file with
their constant prefix dropped the way `catboost/cuda/` is dropped in `gbdt/`:

    cpp/src/decisiontree/   ->  extratrees/ported/decisiontree/
    (RAFT primitives)       ->  extratrees/mojo_only/

| ours | cuML |
|---|---|
| `ported/decisiontree/batched_levelalgo/split.mojo` | `batched-levelalgo/split.cuh` |
| `ported/decisiontree/batched_levelalgo/dataset.mojo` | `batched-levelalgo/dataset.h` |
| `ported/decisiontree/batched_levelalgo/objectives.mojo` | `batched-levelalgo/objectives.cuh` |
| `ported/decisiontree/batched_levelalgo/builder.mojo` | `batched-levelalgo/builder.cuh` |
| `ported/decisiontree/batched_levelalgo/kernels/builder_kernels.mojo` | `batched-levelalgo/kernels/builder_kernels.cuh` |
| `ported/decisiontree/flatnode.mojo` | `cpp/include/cuml/tree/flatnode.h` |
| `mojo_only/pcg_rng.mojo` | `raft/random/detail/rng_device.cuh` |

The one directory-name change is `batched-levelalgo` -> `batched_levelalgo`,
because a Mojo package directory cannot contain a dash. Recorded in
`PORTED_MAP.tsv`.

**`quantiles.cuh` is deliberately absent** and always will be: it is the file
this formulation exists to delete. See `UNPORTED.tsv`.

**One kernel-matrix row departs from cuML's launch shape.** Their
`TPB_DEFAULT = 128` gives every frontier block 128 rows, one per thread; on
a 64-lane wavefront that is a two-wave workgroup doing one compare per lane
and the MI325X is dispatch-bound (range + score passes 17.0 s of a higgs 1M
fit). `DEVICE_TPB` in `builder.mojo` is therefore `128 if WARP_SIZE <= 32
else 512` (DEVIATION 1943), a no-op on NVIDIA and Apple, 2.7x on the two
hot kernels on AMD, tree bits unchanged on every tier. The whole-fit AMD
time is a separate open question, DEVIATION 1945.

## Why this is a sibling directory and not part of `ensemble/`

`ensemble/` is a parallel lane porting cuML's Random Forest. Both lanes need a
flat tree node, a predict traversal and a row partition. Rule 12 says the
predictor of integration pain is FILE CONVERGENCE, not delegation, so that
substrate is **duplicated here on purpose, in minimal form**, and
deduplication into `core/` is a merge-time decision belonging to neither lane.

## Determinism, and the oracle that is actually exact

sklearn draws every random number from ONE sequential 32-bit xorshift stream
(`our_rand_r`, `sklearn/utils/_random.pxd:20-34`) whose draw ORDER depends on
the Fisher-Yates feature walk and on which features were discovered constant.
That order cannot be reproduced by a builder that evaluates features in
parallel. So **bitwise parity with sklearn is impossible by construction**, and
pretending otherwise would be the confound rule 5 warns about. It is recorded
as DEVIATION 130 rather than worked around.

What replaces it is a counter-based keyed draw — cuML's own discipline, ported
rather than invented (`builder_kernels.cuh:165-172`: an fnv1a32 chain over
`(threadIdx, treeid, nodeid)` feeding `raft::random::PCGenerator`), extended by
one key component because a threshold is drawn per FEATURE as well as per node.
Order-independent, and bit-reproducible across Metal, CUDA and HIP by
construction.

Three oracles, in the house pattern:

1. **Exact** — a host-side Mojo transcription of `node_split_random` using the
   SAME keyed draws. Ranges, thresholds, scores and the chosen split are
   compared per node, per cell, on hashed 4096-row fixtures.
2. **Analytic** — fixtures where the correct split is hand-computable for ANY
   threshold inside a known interval, plus adversarial constant features.
   One sabotage per mechanism (rule 8).
3. **Quality band** — sklearn `ExtraTreesClassifier`/`Regressor` at a fixed
   seed, holdout accuracy/MSE only. Never bitwise, never a gate, never on a
   real dataset (rule 4).

## Instrumentation (env-gated, off by default)

Both flags are read ONCE per forest fit, in
`ported/decisiontree/batched_levelalgo/builder.mojo`; unset, the fit is
byte-for-byte the uninstrumented program.

* `MOJOLEARN_IDENTITY_TRACE=<path>` — `core/identity_trace.mojo` stage
  checkpoints, so a cross-backend bit difference has an ADDRESS. The ET fit
  records: `dataset.data` / `dataset.labels` (regression adds
  `dataset.labels.quantized` and `dataset.scale`, the fixed-point boundary —
  ET has no quantile/binning stage, so the resident dataset is that
  boundary), then per group `gN`: `gN.bootstrap.rowids` (the drawn row
  slots, only when `bootstrap=True` -- DEVIATION 460), and per level cycle
  `cM`: `gN.cM.reduce.*` (the
  reduced per-node winners off the readback — `colid`/`num`/`den` for
  classification, where under `criterion='entropy'` `num` is the
  sign-magnitude key of the float gain and `den` is 1 (DEVIATION 459);
  `colid`/`gain` for regression), `gN.cM.split.*`
  (post-rescue selected splits: `thresh`/`colid`/`nleft`/`gain`),
  `gN.cM.partition.rowids` (the partition's output permutation), and
  `gN.leaves` (the leaf pass's output). That file's four rules govern every
  record; rule 4 means a traced run is NEVER a timing.
* `MOJOLEARN_STAGE_TIMES=1` — the shipping forest entry points run under an
  ENABLED `PhaseClock` and print stage -> seconds at fit end. `PhaseClock`'s
  caution applies: every boundary is a `synchronize`, so the phases are
  forbidden to overlap — attribution, never a benchmark; compare the total
  to an untimed run to see the measurement's own distortion.
