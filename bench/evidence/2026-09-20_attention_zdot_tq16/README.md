# Attention-v1 zdot 16x16 schedule (opt-in, AMD pending)

The production full-estash backward's largest NVIDIA compute leaf was zdot:
about 1.02 ms at B1/H12/L2048/HD64, versus 0.88 ms dQ and 0.74 ms dK/dV.
`MOJOLEARN_ATTN_ES_TQ16` changes only that kernel's 256-thread work tile from
8 query rows x 32 keys to 16 rows x 16 keys.  Every row retains its ascending
key fold and pinned dot/division spelling; no LxL allocation is added.

## NVIDIA L40S

RunPod `2w7wair8gqbquf`, L40S 46,068 MiB, driver 580.126.09.  Its 60-minute
watchdog was armed before work; teardown returned DELETE 204 and GET 404.
Raw build, exactness, resource, timing, and stage logs are under `nvidia/`.

All seven ctx/amax/denom/zdot/dQ/dK/dV hashes matched eager and the current
default. Median non-serialized measurements (15 rounds, four warmups):

| Shape | Current backward ms | 16x16 backward ms | Change |
|---|---:|---:|---:|
| B1 L1024 causal | 0.843477 | 0.821904 | -2.56% |
| B1 L1536 causal | 1.686773 | 1.599733 | -5.16% |
| B1 L2048 causal | 2.763871 | 2.594126 | -6.14% |
| B1 L2048 window512 | 1.217909 | 1.111179 | -8.76% |

Serialized target-shape zdot fell from about 1.02 ms to 0.84 ms.  The 16x16
page is 10,432 bytes versus 12,480 bytes, with four resident 256-thread blocks
per SM reported for the clean dres kernel.

## Apple M4

All seven hashes matched eager on L1024, L2048, and L2048/window512.  Raw logs
are under `apple/`.  Derived backward medians were 43.629 -> 43.826 ms
(+0.45%), 184.483 -> 186.109 ms (+0.88%), and 100.399 -> 100.063 ms (-0.33%).
The schedule is therefore effectively neutral on Apple, not an Apple speed
claim.

## AMD MI300X

Hot Aisle deployment `7597dc06-f5e6-4513-98b8-5befac22ea98`, tagged
`mojolearn:attention-tq16:20260920T195525Z`, ran on one MI300X (`gfx942`).
The 30-minute cap was $1.50; the measured account charge was $0.20.  Both
watchdogs were verified before work.  Teardown returned DELETE 204 followed
by GET 404 and `listed=no`; `teardown.txt` records `destroy_confirmed=1`.

All seven hashes matched eager at every shape.  Median derived backward
times (15 rounds, four warmups) were:

| Shape | Current backward ms | 16x16 backward ms | Change |
|---|---:|---:|---:|
| B1 L1024 causal | 2.453379 | 2.464943 | +0.47% |
| B1 L1536 causal | 3.673991 | 3.607469 | -1.81% |
| B1 L2048 causal | 5.522206 | 5.398358 | -2.24% |
| B1 L2048 window512 | 1.988148 | 1.925888 | -3.13% |

Raw logs plus provider, watchdog, cost and teardown receipts are under
`amd-hotaisle/`.

## Disposition

The source remains opt-in.  NVIDIA and AMD show useful medium/large-shape
wins, while Apple is neutral to slightly slower and the smallest AMD case is
within noise but slightly slower.  A production-default change therefore
needs a separately qualified shape/vendor dispatch matrix rather than turning
the schedule on globally.
