# GPT-3-small NVIDIA IDENTICAL GEMM forced-plan qualification

## Scope and environment

- Mode: `MOJOLEARN_NUMERIC_IDENTICAL` only; FP32, no reduced precision.
- Device: NVIDIA L40S, 46,068 MiB, driver 580.178.04, GPU UUID
  `GPU-8fc5fcbb-3818-f68b-8a29-8fd29cd95d9a`.
- Source base: `77b1162b0` (already containing the shipped Apple exact-GEMM
  dispatch change).
- Harness: `bench/gemm_production_shapes_main.mojo`, five alternating samples
  for baseline and candidate after warm-up.
- Shapes cover QKV rows 1,024 through 16,384, single-microbatch MLP up/down at
  widths 2,048 and 3,072, and large rows=32,768 QKV, MLP, and LM-head-chunk
  projections.
- Every result below reported `mismatches_vs_plan10 0` over the complete output
  and a stable output hash. Workspace remained one float for all candidates.

## Shipped dispatch versus plan 10

Ratios are median shipped time / median plan-10 time; lower is faster.

| shape `(m,n,k)` | ratio |
|---|---:|
| `(1024,768,768)` | 0.611 |
| `(2048,768,768)` | 0.987 |
| `(4096,768,768)` | 0.834 |
| `(8192,768,768)` | 0.827 |
| `(16384,768,768)` | 0.823 |
| `(2048,2048,768)` | 0.835 |
| `(2048,768,2048)` | 0.746 |
| `(2048,3072,768)` | 0.825 |
| `(2048,768,3072)` | 0.737 |
| `(32768,2304,768)` | 0.799 |
| `(32768,3072,768)` | 0.787 |
| `(32768,1024,768)` | 0.799 |

The existing shipped k-packed exact plan is therefore already a broad L40S
winner for production GPT-3-small shapes.

## Forced alternatives

Plans 9 and 16 were competitive only at small row counts. Against plan 10,
plan 9 ranged from 0.613--0.889 at rows 1,024--4,096 but regressed to
1.048--1.063 at rows 8,192--16,384, 1.052--1.108 for MLP up projections, and
1.126--1.178 for the large shapes. Plan 16 showed the same shape dependence.
Plan 17 regressed every measured shape by roughly 12--20%; plan 19 was neutral
at small shapes and regressed large shapes by roughly 13--17%.

An additional direct alternating comparison used the shipped dispatch as the
baseline and forced plan 9 as the candidate. Two independent runs agreed:

| shape | run 1 ratio | run 2 ratio |
|---|---:|---:|
| QKV `(1024,768,768)` | 0.991 | 1.017 |
| QKV `(2048,768,768)` | 0.921 | 0.922 |
| QKV `(4096,768,768)` | 1.035 | 1.046 |
| QKV `(8192,768,768)` | 1.266 | 1.285 |
| QKV `(16384,768,768)` | 1.314 | 1.323 |
| MLP projections, rows 2,048 | 1.194--1.344 | 1.212--1.351 |
| large rows 32,768 | 1.416--1.485 | 1.414--1.487 |

## Disposition

No production dispatch change is promoted. The sole repeatable alternative win
is plan 9 for exactly `(2048,768,768)`, about 7.8% on this L40S, while adjacent
and larger shapes regress materially. More importantly, the current dispatch
key is vendor-wide (`COLUMN_NVIDIA`); an L40S-only result cannot safely select a
plan for every NVIDIA architecture. A future architecture-qualified matrix may
revisit the narrow case, but generic NVIDIA routing must retain the shipped
k-packed plan.

The harness now supports `MOJOLEARN_PROD_GEMM_BASELINE_SHIPPED=1`, allowing an
explicit forced plan to be compared directly and alternately with the current
production selector rather than only plan 10. This changes benchmark code only.
