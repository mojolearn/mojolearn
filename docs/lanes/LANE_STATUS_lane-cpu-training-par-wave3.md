# LANE STATUS: lane/cpu-training-par-wave3 (2026-09-15)

Wave 3 of the CPU verification column for the two-device `par-*` lanes: the
triage of the 28 lanes waves 1 and 2 left, and what each one needs. Branch
`lane/cpu-training-par-wave3`, cut from and merged up to origin/main
f624c41de. Wave 1 is `docs/lanes/LANE_STATUS_lane-cpu-training-par-classical.md`
(par-scaler, par-arima, par-holtwinters); wave 2 landed in the same file's
successors and the pool's own docstring (par-queries-knn, par-queries-radius,
par-queries-kde, par-reference-knn, par-reference-knn-reg, par-forest,
par-forest-et, par-mlp).

Counts at this head, all read from the tree, not from prose:

    grep -o '@lane("par-[a-z0-9-]*")' tools/identity_break.py | wc -l   # 39
    python3 python/mojolearn/host_surface.py --covered-lanes \
      | tr ',' '\n' | grep -c '^par-'                                    # 11

39 par lanes, 11 covered, 28 triaged below: **2 coverable now, 24 coverable
after named work, 2 not coverable on a CPU column at all.**

## How a par lane can be covered honestly

`python/mojolearn/_parallel_pool.py` admits an operation on a CPU-only
install only when the CPU column would then check something the plain lane
does not already check. Three shapes qualify:

1. **The driver splits and merges in Python.** One worker request per logical
   shard from a NON-cooperative pool, the parts joined in shard order by the
   driver's own code. That code runs unchanged on CPU, so the CPU column
   checks the same sharding logic the GPU columns check. Waves 1 and 2 are
   all of this shape.
2. **The driver folds the shards in Python.** The one cooperative request
   carries the per-shard results and the worker folds them with
   `parallel_training.ordered_sum_gradients` before one optimizer step. Wave
   2 admitted `mlp_update` this way, from a ONE-device pool only: the GPU
   binding's gradient columns, clip tensors and optimizer ranges are split
   only above `MOJOLEARN_OPTIMIZER_DEVICE_COUNT=1`, so a one-device update
   hides no partition a host binding would have to restate. Two or more
   devices refuse by name.
3. **The host binding restates the split.** The family's host binding carries
   the device tile split keyed by a logical tile count, so the CPU column
   checks the host restatement against the GPU record.

What does NOT qualify: routing a cooperative driver's single request to the
plain host fit. Every par lane already holds itself to the plain fit with
`_same_bytes`, so such a CPU cell would equal the plain lane's cell by
construction and would hash the plain fit under a par label. Wave 1 refused
that and wave 3 keeps refusing it.

## The triage table

`split` is where the work is actually divided. `record` is the committed GPU
columns that carry the lane, which `--require-columns 4` has to clear.

### (a) Coverable now, smallest first

| lane | driver | split | record | what it needs |
|---|---|---|---|---|
| par-samba | `ParallelNeuralTrainer` (`samba_gradient` per shard, `samba_update` folds) | Python, shape 2 above | 166-lane, 3 columns | pool admission only |
| par-samba-clip | the same with `max_norm=0.5` | the same | 166-lane, 3 columns | the same admission |

Both lanes send one `samba_gradient` request per logical shard from
`self._pool = DevicePool(devices)`, which is not cooperative, exactly as
par-mlp sends `mlp_gradient`; the shard gradients come back and
`_parallel_worker.execute` folds them with `ordered_sum_gradients` inside the
one `samba_update` request that `self._update_pool` carries. Since
lane/cpu-training-samba merged, the training, mamba and transformer host
bindings serve every call `SambaStack` makes, and the plain `samba` and
`samba-untied-dropout-accum` lanes are covered. The change is two names:

    CPU_OPERATIONS                 gains 'samba_gradient'
    CPU_SINGLE_DEVICE_COOPERATIVE  gains 'samba_update'

plus the two lanes on the training family's `training_lanes` and
`TRAINING_LANE_NAMES` in `python/mojolearn/host_surface.py`, and the
`LANES`/`DRIVERS` maps in
`python/mojolearn/tests/test_cpu_training_par_classical.py`. No Mojo.

### (b) Coverable after named work

#### b1. The code is small; the RECORD is short a third vendor (2 lanes)

| lane | driver | split | record | what it needs |
|---|---|---|---|---|
| par-rbf-sampler | `transform_rbf_sampler` (`rbf_sampler_rows`) | Python row shards, shape 1 | NVIDIA and AMD only | a third column |
| par-forest-pool | `ParallelForestPredictor` (`forest_prepare`/`forest_predict`/`forest_release`) | 32 logical groves, already restated on the host | NVIDIA and AMD only | `forest_pool_available` on the host bindings, and a third column |

- **par-rbf-sampler.** The driver cuts rows in Python and copies the parts
  back in input order, and the kernel_methods host binding already serves
  `rbf_sampler_transform` (the plain `rbf-sampler` lane is covered). The code
  is one name in `CPU_OPERATIONS`. The block is the record, and it is the
  same block wave 1 reported: the only committed columns that carry the lane
  are
  `bench/results/identity_break/2026-09-15_par-lanes-new/{nvidia-2xh100-new8,amd-2xmi300x-new8}/{one,two}.json`,
  four JSONs from TWO vendors at one and two devices, with no Apple column;
  the 166-lane record does not carry the lane. **Re-checked on 2026-09-15
  against release/0.8.6 (head 9f2ccff71): the 0.8.6 record adds no column
  that carries par-rbf-sampler, so the answer is still no.** It becomes
  coverable the day a record carries the lane on Apple as well, or the day
  its cells are admitted as OWED (`identity_break --diff --require-columns 4
  --owed-json`, which needs the CPU column STABLE over at least two repeats,
  every column that hashes the part to agree, and every owed part to MOVE
  under the host sabotage build: `tools/cpu_identity_gate_check.py owed`).
- **par-forest-pool.** Cooperative, but what it checks is the 32 logical
  grove split, and the rf and trees HOST bindings already register the
  resident grove entries `forest_prepare_gpu`,
  `forest_predict_resident_reuse_gpu` and `forest_release_gpu` over
  `core/forest_host_groves.mojo`. The one missing piece is the capability
  name `_parallel_worker.execute` probes, `forest_pool_available`, which only
  the GPU bindings register (`bindings/forest_inference_binding.mojo` returns
  1). Work: export it from `bindings/_mojolearn_rf_host.mojo` and
  `bindings/_mojolearn_trees_host.mojo`, add it to both families' `exports`,
  and admit the three forest pool operations from a one-device cooperative
  pool. Records: NVIDIA
  `bench/results/multi_gpu/2026-09-14/par-lanes-h100/` and AMD
  `bench/results/identity_break/2026-09-15_par-lanes-new/amd-2xmi300x-old4/`.
  No Apple column of this lane has been recorded.

#### b2. The trainer does not use the pool at all (1 lane)

| lane | driver | split | record | what it needs |
|---|---|---|---|---|
| par-byte-lm | `ParallelByteLanguageModelTrainer` | a native session, no `DevicePool` | 166-lane, 3 columns | a host parallel session |

The trainer opens `byte_lm_parallel_create`, `byte_lm_parallel_open_pooled`,
`step`, `export`, `rollback` and `close` on `bindings/_mojolearn_byte_lm.mojo`
and never touches `DevicePool`, so no pool admission reaches it. Covering it
needs `_mojolearn_byte_lm_host` to register the same session names, restating
the logical-shard loop and the ordered sum on the host. That is a lane of its
own, beside the covered `byte-lm-host-train`.

#### b3. The cooperative families: the split lives in the GPU binding (21 lanes)

Each of these hands the WHOLE fit to one worker from
`DevicePool(devices, cooperative=True)`. The partition is device row tiles,
chunks or replicate ranges inside the GPU binding, read from
`MOJOLEARN_<X>_DEVICE_COUNT`, and at a count of 1 that code takes the plain
single-device path. No host binding restates the split, and the host bindings
deliberately omit the `*_parallel_available` probes the worker demands, so
every one of these refuses by name today. Covering one needs shape 3 above:
its host binding restates the tile split keyed by a logical tile count, and
exports the probe. That is one Mojo change per family, and it checks the host
restatement, not the device code.

| family | lanes | the file that would have to restate the split |
|---|---|---|
| gbdt | par-boosting, par-boosting-pointwise, par-ordered-rmse, par-feature-freq | the packed feature group partition behind `gbdt_parallel_available` and `pointwise_parallel_available` |
| cluster | par-kmeans | `cluster/multi_gpu.mojo` |
| estimators | par-gram, par-logistic, par-dbscan | `core/gram_multi_gpu.mojo`, `glm/impl/qn/multi_gpu.mojo`, the DBSCAN neighborhood rows |
| solver | par-cd | `solver/multi_gpu.mojo` |
| svm | par-svm, par-iforest | `svm/impl/distance/kernel_matrices.mojo` |
| gp | par-gp | the distributed covariance rows |
| mixture | par-gmm | `mixture/multi_gpu.mojo` |
| resample | par-resample | `resample/estimator.mojo`, the global replicate and chunk ranges |
| hdbscan | par-hdbscan | the neighbor and distance row drivers |
| linalg/gp | par-cholesky | `cholesky/multi_gpu.mojo` |
| kernel_methods | par-kernel-ridge, par-nystroem | the SVM kernel row seam plus the Cholesky driver |
| solver/metrics | par-graph-agglomerative, par-graph-spectral, par-graph-umap | the hierarchy and graph row drivers |

Twenty-one lanes. All of them are in the 166-lane record on three columns
except par-gmm, par-resample, par-hdbscan, par-cholesky, par-kernel-ridge and
par-nystroem, which carry the same two-vendor block as b1 and would need a
third column too.

### (c) Not coverable on a CPU column (2 lanes)

| lane | driver | why not |
|---|---|---|
| par-byte-lm-model-pool | `PooledByteLanguageModelTrainer` | its claim is that decoder layers OWNED by separate devices equal a REPLICATED trainer on the first device. With one CPU worker both sides are the same host arithmetic in the same process, so the equality is true by construction and checks nothing. There is no layer ownership to restate. |
| par-byte-lm-offload | `OffloadedByteLanguageModelTrainer` | it admits exactly one device, and its claim is that optimizer state staged in HOST memory equals resident DEVICE state. On a CPU-only install every array is already host memory, so the lane has no second side to compare. |

These two are a result, not a gap. A CPU route for either would compare host
arithmetic with itself under a pooled label.

## Done on this branch

- This triage. No code change is merged yet.

## Resume steps for a session with none of this context

Read `docs/lanes/LANE_STATUS_lane-cpu-training-par-classical.md` (wave 1, the
route and the refusal sentences) and the docstring at the top of
`python/mojolearn/_parallel_pool.py` (waves 1 and 2, the admitted operations)
first. Then, in order:

**1. Take a worktree and sync.**

    cd /Users/andrewhendel/CascadeProjects/mojolearn   # SHARED: never build or commit here
    git worktree add -b lane/cpu-training-par-wave3 <scratch>/wt-par-wave3 origin/main
    cd <scratch>/wt-par-wave3 && git fetch origin main && git merge origin/main

**2. Cover par-samba and par-samba-clip (section (a)).** Edit, in this order:

    python/mojolearn/_parallel_pool.py       CPU_OPERATIONS += 'samba_gradient'
                                             CPU_SINGLE_DEVICE_COOPERATIVE += 'samba_update'
    python/mojolearn/host_surface.py         training family training_lanes += the two lanes,
                                             TRAINING_LANE_NAMES += the two entries
    python/mojolearn/tests/test_cpu_training_par_classical.py
                                             LANES += {"par-samba": "training", "par-samba-clip": "training"}
                                             and the CPU_OPERATIONS assertion
    python3 tools/docs_facts.py --write && python3 tools/docs_facts.py --check

**3. Prove it on ONE RunPod CPU pod.** No GPU, no AMD. Dry run first (no
`--rent` creates nothing), then rent. The four families SambaStack reaches are
core, training, mamba and transformer:

    bash tools/runpod_cpu_leg.sh --lane par-wave3 \
      --build core,training,mamba,transformer \
      --sabotage-build core,training,mamba,transformer \
      --envs default,test \
      --cmd 'python3 tools/identity_break.py --lanes par-samba,par-samba-clip,samba \
               --fixtures base,odd,denormal --repeats 2 --json "$LEG_OUT/cpu-x86.json" && \
             MOJOLEARN_HOST_DIR=python/mojolearn/host-sabotage MOJOLEARN_HOST_ALLOW_SABOTAGE=1 \
             python3 tools/identity_break.py --lanes par-samba,par-samba-clip \
               --fixtures base,odd,denormal --repeats 2 --json "$LEG_OUT/cpu-x86.host-sabotage.json"' \
      --rent

Then on the Mac, against the 166-lane record's three GPU columns:

    python3 tools/identity_break.py --diff <leg_out>/cpu-x86.json \
      bench/results/identity_break/2026-09-14_166-lanes/apple-m4.json \
      bench/results/identity_break/2026-09-14_166-lanes/nvidia-h100-sm_90a.json \
      bench/results/identity_break/2026-09-14_166-lanes/amd-mi325x-gfx942.json \
      --require-columns 4 --lanes par-samba,par-samba-clip

Evidence the lane owes, all four:

- every new cell IDENTICAL x4 against the record, and byte-equal to the plain
  `samba` lane's own CPU cell;
- the host sabotage build moves every new cell (DIVERGENT on all of them);
- the shards are REAL: a probe that counts the pool's requests sees two
  `samba_gradient` requests per step and one `samba_update`, never a plain
  fit. Wave 1's probe is the pattern, in
  `test_cpu_training_par_classical.py::test_sharded_scaler_equals_the_plain_fit_when_built`:
  wrap `DevicePool._call`, record `request[2][0]` when `request[0]` is
  `cpu_reference`, and assert the exact list;
- a scratch reversed fold (reverse the parts handed to
  `ordered_sum_gradients`, then `git checkout` the file) makes the lane read
  REFUSED or DIVERGENT, and the other 26 par lanes still read REFUSED, with
  `tools/cpu_identity_gate_check.py column --covered <covered list>` at 0
  failures.

**4. Merge.** `git merge origin/main`, then `python3 tools/docs_facts.py
--check`, `python3 packaging/wheel_ci.py pins .`, `python3
packaging/wheel_ci.py inventory python/mojolearn`, then `git push origin
HEAD:lane/cpu-training-par-wave3 && git push origin HEAD:main`, and confirm it
landed. Never wait on CI. Main only (0.8.7); never touch release/0.8.6.

**5. Next batch, in this order of increasing cost.** par-forest-pool (b1, the
`forest_pool_available` export, cheapest Mojo change), then par-byte-lm (b2),
then the b3 families one at a time, smallest tile split first. par-rbf-sampler
and par-forest-pool also need a third vendor column before their cells can
clear `--require-columns 4`; neither is worth a GPU rental of its own, so
carry them into the next release record, or admit them as OWED with the
sabotage arm.

## Rules this lane runs under

One CPU core locally, one worker process at a time in the pool. CPU builds and
runs go to ONE RunPod CPU pod at a time through `tools/runpod_cpu_leg.sh`,
from origin/main, verified deleted after. No GPU pods and no AMD. Existing
lanes get a base-fixture spot check only; new cells are proven in full.
Remove the worktree after the last merge.

## Pods

None rented on this branch so far.
