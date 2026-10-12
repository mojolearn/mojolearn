# Cross-vendor training

In `identical` mode, a training step gives the same bits on NVIDIA and AMD GPUs for the same release,
input bytes, model state, settings and supported configuration. Two things follow for language models:

1. **Checkpoint handoff.** Stop a run on one vendor's GPUs and resume it from its checkpoint, optimizer
   state included, on another vendor's GPUs.
2. **Joint training (experimental).** Train one model at the same time on machines with GPUs from
   different vendors, with every replica holding the same bits after every step.

The release requirement is NVIDIA and AMD. Apple GPU and CPU results are recorded, and for the
configurations tested so far they agree too. `fast` mode makes no identity promise, so none of this
applies to it. See [Numeric modes](numeric_modes.md).

## Checkpoint handoff

A binary checkpoint holds the parameters, both AdamW moments, the per-tensor flags, the step count and
your data schedule. Write it on one machine and restore it on another:

```python
import mojolearn

# On the first machine (for example, NVIDIA GPUs)
sha = trainer.export_checkpoint_binary("step-1000.blm")   # returns the file's SHA-256

# On the second machine (for example, AMD GPUs), with the same mojolearn release
trainer = mojolearn.LanguageModelTrainer.from_checkpoint_binary("step-1000.blm")
result = trainer.train_step(next_ids)                     # the batch for step 1,001
```

The multi-GPU trainers do the same with their own checkpoints. `ParallelByteLanguageModelTrainer`,
`PooledByteLanguageModelTrainer` and `OffloadedByteLanguageModelTrainer` share one checkpoint format, and
`from_checkpoint(checkpoint, devices=...)` reopens it with a different number of GPUs. Keep the logical
shard count, shard order, shapes and data bytes the same; see [Multi-GPU training](multi_gpu.md).

The trainer does not fetch data. You must feed the same batches in the same order after the handoff, and
`data_schedule` (stored in the checkpoint) is where you record how.

## Joint training across vendors (experimental)

`mojolearn.cross_vendor` runs one data-parallel step across several machines. A coordinator, which needs no
GPU, assigns each worker a set of logical shards. Each worker computes its shards' gradients with the same
kernels a one-GPU step uses. The coordinator adds them in shard order with the same fixed left fold
(`ordered_fold`), and every worker applies the total with the same AdamW update. The result is also the
step a single GPU computes over the same shards.

Start the coordinator on any machine:

```sh
python -m mojolearn.cross_vendor coordinator --port 7777 --workers 2 --shards 4 --steps 100
```

Start a worker on each GPU machine:

```python
from mojolearn.cross_vendor import Worker

Worker(state,                   # a LanguageModelTrainer state_dict(), the same on every worker
       shards=[0, 1],           # this worker's logical shards; together the workers own 0..K-1 once
       batches=lambda step, shard: ids_for(step, shard),   # int32 (batch, length + 1), same everywhere
       address=("coordinator-host", 7777),
       name="nvidia-box").run()
```

Every step, each worker reports the SHA-256 of its whole state (parameters, both moments, flags). The
coordinator stops and names the workers if any two differ. It refuses to start if the workers begin from
different states or do not own every shard exactly once. `Worker(..., lr_for_step=f)` sets the learning
rate from your schedule before each step.

| Protocol | How | Traffic per step |
|---|---|---|
| Gathered (default) | Every worker sends each shard's gradient to the coordinator, which folds them and sends the total back | K + W gradients (K shards, W workers) |
| Chained (`--chained`, `chained=True`) | Each worker owns a contiguous block of shards and continues the fold from the previous worker's partial sum | W + (W - 1) gradients |

A gradient is 4 bytes per parameter: 649 MB at 162M parameters. In the chained protocol a worker holds its
block's gradients in host memory until its turn.

Limits:

- No authentication or encryption. Use a trusted network or an ssh tunnel.
- Each worker drives one GPU.
- The protocol may change between releases.

## What was demonstrated: the 162M model

The published 162M model (see [Language models](language_models.md)) was trained twice from the same
initialization on the same 2.62 billion tokens, along two different routes through NVIDIA, AMD and Apple
GPUs. Each step used 64 logical shards of four 2,048-token sequences.

| Steps | Route A | Route B |
|---|---|---|
| 1-1,000 | 2 x NVIDIA H100 | 1 x AMD MI325X |
| 1,001-2,000 | 2 x NVIDIA H100 | 2 x AMD MI300X |
| 2,001-2,400 | 1 x H100 + 1 x MI325X, jointly, in separate machines | 1 x NVIDIA H100 |
| 2,401-3,900 | 2 x AMD MI300X | 2 x NVIDIA H100 |
| 3,901-4,900 | 2 x NVIDIA H100 | 2 x AMD MI300X |
| 4,901-5,000 | Apple M3 Ultra | NVIDIA L40S |

Each route resumed from its own checkpoints at every change of hardware. Steps 2,001-2,400 of route A ran
as a joint cross-vendor group. Across all 5,000 steps the two routes agree bit for bit in training state,
summed gradients, all 64 per-shard losses and the learning rate. All 56 common saved checkpoints match,
and the final checkpoint files have the same SHA-256:

```
4129921e99ed404db1ca4b7b94ebc43c0bee18e7711deb326b31ec89e0b402c8
```

The segments also ran different mojolearn releases (0.8.15 to 0.8.24). That they agreed is a
measurement for those releases. In general, bits may change between releases, so resume a run with the
release that wrote its checkpoint.

The evidence and a checker that needs no GPU are in
[release/gpt3-small-162m](https://github.com/mojolearn/mojolearn/blob/main/release/gpt3-small-162m/README.md).
The checker confirms the stored records are complete and agree; it does not recompute training. The live
cross-vendor protocol's own records are described in
[docs/CROSS_VENDOR_TRAINING.md](https://github.com/mojolearn/mojolearn/blob/main/docs/CROSS_VENDOR_TRAINING.md).
