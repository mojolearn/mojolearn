# Exact attention zdot TQ16 production routing

The previously qualified 16-query x 16-key full-estash zdot schedule is now
the NVIDIA default and is selected on CDNA AMD only for `L >= 1536`.  Apple,
RDNA and unknown columns retain 8x32.  This threshold is deliberately based
only on shape and compiled column, never values:

| Column/regime | Route | Qualification |
|---|---:|---|
| NVIDIA, all full-estash shapes | TQ16 | L40S: 2.56--8.76% faster at L1024--2048 |
| CDNA AMD, L < 1536 | TQ8 | MI300X L1024 TQ16 was 0.47% slower |
| CDNA AMD, L >= 1536 | TQ16 | MI300X: 1.81--3.13% faster, including L2048/window512 |
| Apple/RDNA/other | TQ8 | unchanged |

The raw timings, all-seven-stage hashes, provider receipts, and teardown
records remain in [the qualification directory](2026-09-20_attention_zdot_tq16/README.md).
The routing adds one host integer comparison on AMD before an already much
larger GPU launch; NVIDIA and the fallback columns fold to a compile-time
constant. It does not add a kernel launch, allocation, synchronization, or
device-side branch.

`MOJOLEARN_ATTN_ES_TQ16` remains an explicit force-on reproduction override.
`MOJOLEARN_ATTN_ES_TQ8` is the force-off/rollback override; defining both is
refused. The route is used only by the full-layout estash backward. Packed
estash and recompute remain mutually separate profiles and do not call it.

## Gates

- `attention_zdot_routing_check.mojo`: PASS for Apple, NVIDIA, AMD, RDNA,
  forced TQ8 and forced TQ16; includes AMD 1535/1536 boundary and window case.
- Current launcher, forced TQ16 on Metal: exact all seven stages against eager
  at `(L,window) = (31,0), (33,0), (65,17), (129,64)`, including both tile
  tails and masked/window tails.
- The same four current-launcher cases pass under the unchanged TQ8 fallback.
- `attention_masked_tail_check`, `attention_tail_guard_check`, and
  `transformer_fused_check` pass under IDENTICAL on Apple.

The arithmetic kernel is unchanged from the three-vendor qualification; this
change promotes only its measured selection boundary.
