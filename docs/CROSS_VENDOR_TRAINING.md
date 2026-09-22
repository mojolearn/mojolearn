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

A worker on each GPU machine:

```python
from mojolearn.cross_vendor import Worker

Worker(state,                      # a LanguageModelTrainer state_dict(), the same on every worker
       shards=[0, 1],              # this worker's logical shards; together the workers own 0..K-1 once
       batches=lambda step, shard: ids_for(step, shard),   # int32 [batch, length + 1], same function everywhere
       address=("coordinator-host", 7777),
       name="my-mac").run()
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
(`bench/results/lm_segment_t0_2026-09-22/`). The host fold runs the
vectorized NumPy spelling when NumPy is installed and the exact pure-Python
one otherwise; `test_cross_vendor.py` holds the two equal.

A worker holds its block's gradients in host memory until its turn
(`shards x 4 bytes x n_total`; 44 shards of the 162M shape is 28 GB).

`Worker(..., lr_for_step=f)` applies `f(step)` through `trainer.set_lr`
before each step's gradients, so a recipe's learning-rate table drives a
live group exactly as it drives `tools/lm_segment.py`.

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
