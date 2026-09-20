# GPT-3-small IDENTICAL embedding-backward plan

Apple M4, production dimensions `positions=32768`, `vocab=50257`,
`width=768`, repeated-token fixture with 30,840 non-padding positions. The
existing PLAN_SCAN and stable total-key PLAN_SORT paths were timed through
synchronization and then compared over all 38,597,376 FP32 dWeight cells,
50,257 counts, 50,258 run boundaries, and 30,840 used permutation entries.
Every value matched exactly.

Three repeated process runs (milliseconds):

| run | scan baseline | scan repeat | sort |
|---:|---:|---:|---:|
| 1 | 80.999 | 65.611 | 18.109 |
| 2 | 86.244 | 65.231 | 18.052 |
| 3 | 83.488 | 64.863 | 19.234 |

The production ByteLM call now selects PLAN_SORT only at 16,384 or more token
positions. Smaller calls retain PLAN_SCAN and its zero-allocation metadata
path. Explicit embedding API plan selection is unchanged. PLAN_SORT allocates
the next-power-of-two UInt64 key array: 256 KiB at 32,768 positions. Its total
key `(id, position)` preserves every duplicate token's ascending-position FP32
fold.

The pre-existing H100 evidence at V=128256,D=4096,T=4096 also favored sort
(4.168 ms at its best measured geometry versus 9.807 ms shipped scan), while
matching all 525,336,576 FP32 cells. AMD performance remains unpriced; both
paths already have cross-vendor exactness evidence.

`pixi run check-train-step` passed all clauses after the routing change.
