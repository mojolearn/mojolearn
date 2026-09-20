# GPT-3-small IDENTICAL weight-gradient GEMM screen

Mode is FP32 IDENTICAL. The five repeated non-head OP_TN shapes represent
Q/K/V/O dWeight (48 calls), gate/up dWeight (24), and down dWeight (12) in a
12-layer GPT-3-small step. The LM-head dWeight occurs once.

## Apple M4

Forced plan 9 versus shipped dispatch was bitwise identical over every output
cell. The shipped gate/up shapes already selected plan 9. The uncovered shapes
were:

| shape | shipped median | plan 9 median | approximate reduction |
|---|---:|---:|---:|
| `(768,768,2048)` Q/K/V/O | 17.19 ms | 12.92 ms | 25% |
| `(768,2048,2048)` down | 61.39 ms | 40.78 ms | 34% |
| `(768,3072,2048)` down | 108.20 ms | 70.46 ms | 35% |

Weighted by call count, these three changes save approximately 0.90 seconds,
about 16% of the measured non-head dWeight subtotal. A dispatched candidate
using plan 9 for `m == 768 && n >= 768 && k >= 1024` repeated the same result
with zero mismatches against plan 10.

## NVIDIA L40S

Two independent alternating runs found the opposite geometry preference.
Plan 9 was roughly 1.7x slower for Q/K/V/O, 1.2--1.3x slower for the 2048
gate/up and down shapes, neutral only at the 3072 shapes, and roughly 1.6x
slower for LM-head dWeight. Every complete output still matched plan 10.
The candidate must therefore remain Apple-only; a portable plan-9 change is
rejected. Raw repetitions and full-output hashes are in `nvidia/`.

Owned pod `8g19xmakj0d7pe` was deleted at 2026-09-20 16:00:00 EDT: DELETE
returned HTTP 204 and immediate verification GET returned HTTP 404.

## AMD status

The read-only allocator reported no free AMD route: Hot Aisle had no matching
stock, DigitalOcean already had one live GPU droplet, and RunPod reported no
MI300X stock. No AMD resource was touched or created. The production guard is
compile-time restricted to `NUMERIC_IDENTICAL` plus `COLUMN_APPLE`; AMD and
NVIDIA therefore retain their existing routing byte-for-byte. The dispatch
gate asserts plan 10 for the new shapes on every non-Apple column and in every
non-IDENTICAL mode.

Local gates passed on Apple: IDENTICAL and default production dispatch,
`check-gemm-identity`, `check-transformer-backward` (17 fixtures and 37 exact
stages), and `check-train-step`.
