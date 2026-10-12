# Multi-GPU training

mojolearn can split some training and inference work across several GPUs in one machine. Multi-GPU use is
opt-in: you name the devices. It covers the trainers and estimators listed on this page, not every
estimator. The names are in the [multi-GPU API reference](../api/multi_gpu.rst), and the full design notes
are in [docs/multi_gpu/README.md](https://github.com/mojolearn/mojolearn/blob/main/docs/multi_gpu/README.md).

## The rule: logical shards, not GPUs

Each step is split into a fixed number of **logical shards** (microbatches) that you choose. Each shard's
gradient is computed with the same kernels a one-GPU step uses. The shard gradients are then added in
ascending shard order with one fixed left fold: the first gradient is copied, and each later one is added
in FP32. No collective chooses an addition order at run time.

So the number of logical shards belongs to the run and its checkpoint, and the number of GPUs does not.
If the starting state, optimizer settings, shard count, shard order, shard shape and data bytes are the
same, a step on one GPU and a step on eight give the same bits. The docs call this one-GPU replay.

A few consequences:

- Shard losses are means over their own microbatch, and the aggregate gradient is their **sum**. There is
  no division by the shard count.
- With more than one shard, a step matches the one-GPU replay over the same shards, not a single step over
  one batch that is K times larger.
- Across vendors, the result also depends on the model kernels agreeing for the chosen shapes, which is what
  `identical` mode provides (see [Cross-vendor training](cross_vendor_training.md)).
- You own the data schedule and must supply the next logical window after a restore.

## Byte language model, data parallel

`ParallelByteLanguageModelTrainer` keeps a model replica on each GPU. It starts from a
`LanguageModelTrainer.state_dict()`:

```python
from mojolearn import ParallelByteLanguageModelTrainer

initial = trainer.state_dict()        # a LanguageModelTrainer
with ParallelByteLanguageModelTrainer(initial, devices=(0, 1), logical_shards=8) as par:
    losses = par.train_step(microbatches)      # exactly 8, in logical order
    checkpoint = par.checkpoint()
    gradients = par.export_gradients()

# The same logical shards replayed on one GPU.
with ParallelByteLanguageModelTrainer.from_checkpoint(checkpoint, devices=(0,)) as replay:
    replay.train_step(next_microbatches)
```

Each microbatch is int32 ids of shape `(batch, length + 1)`. Losses come back in logical order. By default
the AdamW moments are split across the GPUs in disjoint ranges (`pool_optimizer=True`): moments and
rollback copies total 20 bytes per parameter across all GPUs, instead of 20 bytes per parameter on each.
Parameters, gradients and activations stay replicated. `optimizer_ownership()` reports the ranges and bytes.
Checkpoints are Python dictionaries; export them before `close()`.

## Byte language model, model pooling and offload

| Class | What is split | Use |
|---|---|---|
| `PooledByteLanguageModelTrainer` | Decoder layers and their optimizer state, one owner per layer; embedding and head on the first GPU | A model too large for one GPU |
| `OffloadedByteLanguageModelTrainer` | One decoder layer at a time on one GPU; full state in host memory | Replay a pooled run on a single GPU |

```python
from mojolearn import PooledByteLanguageModelTrainer, OffloadedByteLanguageModelTrainer

with PooledByteLanguageModelTrainer(state, devices=(0, 1), logical_shards=3) as pooled:
    pooled.train_step([mb0, mb1, mb2])
    checkpoint = pooled.checkpoint()

with OffloadedByteLanguageModelTrainer.from_checkpoint(checkpoint, devices=(0,)) as replay:
    replay.train_step(next_microbatches)
```

All three byte-LM trainers share one checkpoint format and logical shard order, so a checkpoint moves
between them. Pooling runs layers in sequence and copies activations between GPUs; it pools memory and
makes no speed claim. Each layer, and the embedding and head, must fit on its owner. The offload trainer
needs exactly one GPU and enough host memory for the full state.

## MLP and Samba

`ParallelNeuralTrainer` wraps a `SmallMLPTrainer` or a `SambaStack`:

```python
from mojolearn import ParallelNeuralTrainer

with ParallelNeuralTrainer(model, devices=(0, 1), logical_shards=3) as par:
    losses = par.train_step([(x0, y0), (x1, y1), (x2, y2)])
    checkpoint = par.checkpoint()

with ParallelNeuralTrainer.from_checkpoint(checkpoint, devices=(0,)) as replay:
    replay.train_step(next_shards)
```

Global gradient clipping keeps its original cross-tensor norm. A `SambaStack` must use
`accumulation_steps=1`, because `logical_shards` takes over accumulation. Workers are subprocesses on one
machine that exchange bytes through host memory. This is not NCCL, distributed autograd or a multi-node
service.

## Partitioned estimators

These schedule independent pieces of work and keep the original output order:

| Module | Entries | What is split |
|---|---|---|
| `parallel_forecasting` | `forecast_arima`, `predict_arima`, `forecast_exponential_smoothing`, `predict_exponential_smoothing` | Independent series of a fitted model |
| `parallel_gaussian_process` | `fit_gaussian_process_classifier`, `predict_gaussian_process_classifier` | One-vs-rest classes; each binary problem must fit on one GPU |
| `parallel_ivf` | `DistributedIVFIndex` | Stored candidate rows; the coarse quantizer is shared |
| `parallel_model_selection` | `cross_val_score` | Whole cross-validation folds, one per GPU at a time |

```python
from mojolearn.parallel_forecasting import forecast_arima
from mojolearn.parallel_model_selection import cross_val_score
from mojolearn import DistributedIVFIndex

predictions = forecast_arima(fitted_arima, 12, devices=(0, 1), series_per_shard=2)
scores = cross_val_score(estimator, X, y, devices=(0, 1), cv=5)

with DistributedIVFIndex.from_index(fitted_index, devices=(0, 1)) as index:
    distances, row_ids = index.search(queries)
```

`DistributedIVFIndex` does not distribute index training or `extend`. Forests, boosting, k-means, linear
models, SVMs, Gaussian process regression and scalers have similar entries in `parallel_ensemble`,
`parallel_classical` and `parallel_preprocessing`; the README's table lists what each one splits.

## What has been measured

The ordered byte-LM, MLP and Samba trainers passed cloud gates on two RTX 4090s and on two H100s from one
source snapshot, with gradients and model states equal bit for bit for the tested configurations. An
earlier identity record matched sixteen parallel lanes on two MI300X GPUs and two H100s against
single-device AMD and Apple replay. Layer pooling and offload have NVIDIA receipts only (H100 and RTX 5090).
Several partitioned estimators have software tests only, with physical NVIDIA and AMD qualification
pending. No throughput or scaling claim is made. The per-path status is in
[docs/multi_gpu/README.md](https://github.com/mojolearn/mojolearn/blob/main/docs/multi_gpu/README.md).

The 162M model on [Language models](language_models.md) was trained with 64 logical shards per step,
summed in a fixed order, on segments of one and two GPUs. Its two training routes agree bit for bit at
every step.
