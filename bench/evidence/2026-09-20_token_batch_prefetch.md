# Ordered prefetch for staged pretokenized corpora

Base: `c0c702d6374b75123274cf7e22fd199b53c3f064`. Hardware: Apple M4,
10 logical CPUs, arm64, macOS 26.5.2 (25F84).

## Inventory

- Fixed-order microbatch accumulation already ships as
  `training.accumulate_grads`: a balanced ascending-microbatch tree. Its
  multi-device implementation partitions gradient columns, so device count
  does not alter any cell's reduction.
- Transformer backward checks already exercise selective recomputation of
  cheap intermediates, including normalization statistics.
- `lm_corpus.TokenBatches` verifies and memory-maps a pretokenized artifact
  (including corpora staged once from R2), but previously exposed only the
  synchronous `ids(step)` call. There was no bounded loading/compute overlap.

## Qualified stage

`TokenBatches.prefetch(start_step, steps, depth=2)` has one daemon producer
over the already verified local mmap and a bounded ordered queue. Credentials,
remote state, and network failures never enter the training process. It yields
the original logical step with each owning batch, propagates a producer error
at that step, and joins the worker when exhausted, refused, or closed early.

Benchmark command:

```sh
PYTHONPATH=python python bench/bench_token_prefetch.py
```

The fixture is a 64,000,000-token (256 MB) verified int32 artifact. Each batch
is 256 x 2049 tokens (2,098,176 bytes), 48 steps are consumed, and the 1.5 ms
consumer interval models a native/GPU train call that releases the Python GIL.
This isolates the host loading/compute overlap; it is not a GPU throughput
claim.

Interleaved synchronous/prefetch seconds:

- synchronous: 0.110188, 0.141880, 0.141942, 0.148079, 0.150261
- prefetch: 0.093675, 0.097251, 0.100287, 0.101574, 0.101420

Medians are 0.141942 and 0.100287 seconds, a 29.3% reduction. All five pairs
were faster. Independent synchronous and prefetched streams hashed eight full
batches, including their logical step numbers, to the same SHA-256:
`1c7a4320bdd6fed5b5034bdad2c1b827e7291b68d2d0ffa4522bd502d2aa090f`.
Peak process RSS was 205,996,032 bytes. The explicit queue bound permits at
most `depth` prepared batches beyond the batch held by the consumer (about
4.0 MiB additional payload at the benchmark's default depth two).

The added regression checks exact bytes/order, validation refusals, and worker
exception propagation. `py_compile` passed for product, test, and benchmark.
The local Python environments do not contain pytest, so that test was not run
through pytest in this lane.
