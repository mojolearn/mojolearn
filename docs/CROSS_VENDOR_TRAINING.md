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

## Limits

- Every step sends each shard's full gradient (4 bytes per parameter) to the
  coordinator and the total back. It suits small models on a local network.
- No authentication or encryption. Use a trusted network or an ssh tunnel.
- Each worker drives one GPU. On a Mac that is the only choice.
- Experimental: the protocol may change.

## Evidence

`bench/results/live_xvendor/` records live runs; `bench/results/par_lm_xvendor/`
records the one-process columns the live runs are held to.
