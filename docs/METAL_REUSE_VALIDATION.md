# Metal synchronization/readback validation

This checks the synchronization and readback changes through `f529fab91` without
changing production code or weakening existing checks. The prior reports cover
wait/launch reductions on the B2/L4 fixture. This follow-up broadens correctness
coverage and accounts for the temporary host memory added by batched readback.

## What was exercised

The new `transformer/checks/metal_reuse_validation.mojo` selects exactly one
existing fixture per invocation. It allocates weights, rotary tables, forward
stages, backward stages, GEMM workspace and KV cache once. The cache and stage
capacity is L+4, so every logical readback excludes an unused tail.

Each invocation performs two complete forward/backward executions. On the second
it changes the first input by 0.25 and the first output gradient by 0.5, resets
only the cache's logical position to zero, and reuses the allocated device
objects. Both executions compare all 30 forward stages and all 37 backward
stages bitwise against freshly computed host oracles. The second must also
produce changed final forward and input-gradient outputs, preventing stale
results from passing a repeated-input test. It records all 67 stages on the
second execution; the runner checks record count and unique tags.

The input upload is fresh each time, as are host readback allocations. This
checks device scratch/cache/workspace reuse at fixed shape, not resizing a live
stage object, incremental decode, or reuse of a host staging allocation.

## Results: Apple M4 / Metal / IDENTICAL, 2026-09-17

Four separate rounds, each one fixture and two executions, passed. In total:
536 stage comparisons and 666,296 cell comparisons, with no mismatches.

| Shape | Forward + backward cells per execution | Forward staging bytes | Backward staging bytes | Process peak RSS, MiB | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| B2, L4, head16 | 13,752 + 15,328 | 55,008 | 61,312 | 35.59 | PASS |
| B3, L16, head16 | 46,120 + 50,176 | 184,480 | 200,704 | 36.77 | PASS |
| B1, L4, head24 | 15,876 + 20,176 | 63,504 | 80,704 | 36.42 | PASS |
| B1, L64, head16 | 83,336 + 88,384 | 333,344 | 353,536 | 37.72 | PASS |

All jobs completed inside their 60-second queue-plus-execution deadlines.
Compiles used two workers; runs were serialized through the shared scheduler
with single-threaded compute settings. No full-suite retry or fixture reduction
was used.

Staging sizes are exact logical element counts times four bytes. The largest
single staging allocation was 353,536 bytes (345.25 KiB); forward and backward
staging buffers do not coexist. The returned lists and host oracle lists do
coexist with staging, and process RSS measured by macOS `/usr/bin/time -l`
includes them and runtime overhead. RSS is not GPU memory usage or a leak test.
The previous largest single-stage staging sizes are preserved in each summary;
batching increases temporary memory as documented in the readback reports.

The logs also time device work plus dumps, excluding host-oracle construction
and comparison, and scheduler receipts capture whole-job duration. There is no
baseline arm here. Traced and untraced times are not interchangeable, and these
results do not establish a full-suite speedup.

## Reproduce one round

Compile once from the source checkout:

```sh
python3 tools/mac_slot.py run pixi run mojo build -j 2 \
  -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_STEP_PHASE_TIMERS=1 -I . \
  transformer/checks/metal_reuse_validation.mojo -o /tmp/metal-reuse-validation
```

Choose exactly one of `base`, `batch`, `head`, or `long`, and a new output path:

```sh
python3 bench/results/metal_reuse_validation/2026-09-17-apple-m4/run_round.py long \
  --binary /tmp/metal-reuse-validation --scheduler tools/mac_slot.py \
  --out /tmp/metal-reuse-long
```

The runner enforces the 60-second shared deadline and preserves logs on failure.
Raw logs, scheduler receipts, stage cards, summaries, binary/source hashes and
the runner are under `bench/results/metal_reuse_validation/2026-09-17-apple-m4/`.
Build log and binary: `~/mojolearn-evidence/metal-validation-2026-09-17/`.

## Remaining limits

CUDA/HIP have not been executed in this follow-up. Neither NVIDIA nor ROCm tools
are available on this local Mac, and no remote GPU was used. Those columns
remain pending; Metal agreement does not establish cross-vendor identity.

The longest tested sequence is 64 and largest head dimension is 24. Large model
sizes, longer sequences, live capacity growth, incremental decode, and sustained
memory-pressure/leak tests remain outside this evidence. In particular, attention
readback storage grows quadratically with sequence length. The sub-megabyte
staging sizes above are not a general memory bound.

Keep the validated optimizations, but make no further wait-removal or full-suite
speed claim from this follow-up. Future validation should target these remaining
coverage gaps rather than repeat already-passing shapes.
