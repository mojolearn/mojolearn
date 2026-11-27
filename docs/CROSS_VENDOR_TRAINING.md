# Live cross-vendor training (experimental)

`mojolearn.cross_vendor` trains one language model on several machines at the
same time, each with its own GPU from any vendor (an Apple Mac, an NVIDIA box,
an AMD box), and every replica holds the same bits after every step. The step
is also the same bits a single GPU produces for the same logical shards.

## Use

A coordinator on any machine (no GPU needed):

```
python -m mojolearn.cross_vendor coordinator --port 7777 --workers 2 --shards 4 --steps 100
```

On each worker, start from the same trainer state and the same token batches.
Save the batches as an int32 NumPy array of shape
`(steps, logical_shards, batch, length + 1)`. For the coordinator above,
the first two dimensions are 100 and 4. This worker owns shards 0 and 1.
On the second machine, use shards 2 and 3 and a different worker name.

```python
import numpy as np
from mojolearn.cross_vendor import BatchFeed, LearningRateTable, Worker

# state is the same LanguageModelTrainer.state_dict() on both machines.
start = state["completed_steps"]
tokens = np.load("token_batches.npy", allow_pickle=False)
worker = Worker(state, shards=[0, 1], batches=BatchFeed(tokens, start_step=start),
                lr_for_step=LearningRateTable([0.001] * len(tokens), start_step=start),
                address=("coordinator-host", 7777), name="worker-0")
worker.run()
```

## Why the bits agree

A step is K logical shards, and K belongs to the recipe, not the hardware.
Each worker computes the gradients of its own shards with the IDENTICAL
kernels a one-GPU step uses, so a shard's gradient does not depend on the GPU
or its vendor. The coordinator sums them in shard order with the same fixed
left fold the GPU reduction uses (`ordered_fold`), and every worker applies
that total with the same elementwise AdamW. No collective picks an addition
order at run time.

## What is checked

Every step, every worker reports the sha256 of its whole state (parameters,
both AdamW moments, flags). The coordinator stops, naming the workers, if any
two differ. It also refuses to start if the workers begin from different
states or do not own every shard exactly once.

## Two protocols

**Gathered** (the default): every worker sends every shard's whole gradient
to the coordinator, which folds them all and sends the total back. K + W
gradients on the wire per step. Right for a small model on a local network;
41 GB a step at 162M parameters and K = 64.

**Chained** (`--chained` on the coordinator, `chained=True` on every worker):
each worker owns a CONTIGUOUS block of shards in shard order. On `step`
every worker computes its block's gradients and holds them. Then, in shard
order, the coordinator hands each worker the fold so far; the worker
continues the same left fold onto it with its own shards one at a time
(`ordered_fold(held, prefix=...)`) and returns the result; the last worker's
result is the total, which the coordinator sends to every other worker
(the last one already holds it and is sent only its hash). W + (W - 1)
gradients on the wire per step instead of K + W: two workers with the
coordinator on the first owner's box put two gradients on the wide-area
link. The bits are the flat fold's bits because the fold is a left fold, and
the M4 column, the gathered group and chained groups of two and three
workers (unequal blocks) agree step for step
(`bench/results/lm_segment_t0_2026-09-22/`). The host fold runs in compiled Mojo.

A worker holds its block's gradients in host memory until its turn
(`shards x 4 bytes x n_total`; 44 shards of the 162M shape is 28 GB).

To use the learning rate in the trainer state, omit `lr_for_step`. For a
schedule, pass a `LearningRateTable` with one rate per absolute training step.

## Limits

- The gathered protocol sends each shard's full gradient (4 bytes per
  parameter) to the coordinator and the total back; the chained protocol
  sends one gradient per worker each way, still 649 MB at 162M.
- No authentication or encryption. Use a trusted network or an ssh tunnel.
- Each worker drives one GPU. On a Mac that is the only choice.
- Experimental: the protocol may change.

## Evidence

`bench/results/live_xvendor/` records live runs; `bench/results/par_lm_xvendor/`
records the one-process columns the live runs are held to.
