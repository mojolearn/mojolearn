# GPT-3-small IDENTICAL GEMM dispatch on Apple

Mode: `MOJOLEARN_NUMERIC_IDENTICAL`; Apple M4, 16 GiB; source base
`90d4fd880`. Builds were capped at two jobs.

## Candidate

For transformer-training products with at least 1,024 rows and one matrix
dimension equal to GPT-3-small's `d_model=768`, select the existing exact
64x64/reg4x4 plan instead of 128x128/reg8x8 on Apple. Both plans use the same
ascending leaf products, FTZ seams, and fold tree; this changes only which
thread owns a cell and the output tile geometry. Other vendors and FAST and
DETERMINISTIC modes retain their prior dispatch.

The production benchmark now includes QKV at B1/L1024 and B1/B2/B4/B8 with
L2048, plus both MLP directions at repository FFN width 2,048 and standard
GPT-3-small width 3,072. It alternates explicit plan 10 and dispatched plan 9
inside each repetition and reports both raw arrays.

## Apple timing and exactness

Two quiet sequential runs used `tools/mac_slot.sh metal`. The second run's
medians are representative below; the first independently showed the same
direction. Every shape reported `mismatches_vs_plan10 0` across the complete
output and the same stable output hash in all earlier forced-plan and final
dispatched runs.

| OP_NT shape | plan 10 median | plan 9 median | reduction |
| --- | ---: | ---: | ---: |
| 1024 x 768 x 768 | 15.812 ms | 8.684 ms | 45.1% |
| 2048 x 768 x 768 | 17.335 ms | 9.747 ms | 43.8% |
| 4096 x 768 x 768 | 33.053 ms | 18.949 ms | 42.7% |
| 8192 x 768 x 768 | 64.004 ms | 37.391 ms | 41.6% |
| 16384 x 768 x 768 | 123.577 ms | 75.498 ms | 38.9% |
| 2048 x 2048 x 768 | 43.873 ms | 25.641 ms | 41.6% |
| 2048 x 768 x 2048 | 48.576 ms | 30.853 ms | 36.5% |
| 2048 x 3072 x 768 | 71.498 ms | 44.736 ms | 37.4% |
| 2048 x 768 x 3072 | 86.485 ms | 57.552 ms | 33.5% |

Both plans require one retained workspace float at these shapes, so retained
device memory is unchanged.

Gates:

* IDENTICAL production dispatch boundary gate: PASS.
* Non-IDENTICAL dispatch isolation gate: PASS.
* Transformer forward oracle gate: 17 cases, all 30 stages and 349,206 cells
  bit-identical.

## Rejected alternatives

* A QKV kernel sharing the input tile would triple the tuned reg8x8 kernel's
  accumulator set from roughly 64 to 192 FP32 values per thread. That creates
  a severe register/occupancy risk before any timing evidence. A grid-z-only
  fusion shares no input traversal and merely trades three launches for a
  more complex pointer-selecting kernel. Neither was promoted.
* Pre-sizing the reusable GEMM workspace cannot improve these GPT-shaped
  calls: production dispatch reports `workspace_floats 1` for QKV and both
  MLP directions, so no first-call workspace growth or synchronization occurs.

No cloud resource was used. NVIDIA and AMD dispatch are unchanged.
