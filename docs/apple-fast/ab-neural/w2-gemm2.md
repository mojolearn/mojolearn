# w2-gemm2 A/B requests (branch lane/apple-fast-neural-w2-gemm2)

Arm A in every line is the wave-1 kernel (`-D MOJOLEARN_AFN_GEMM_SIMDGROUP -D
MOJOLEARN_AFN_GEMM_BF16_MMA`); arm B adds one GEMM2 define. All candidates are
exact f32 accumulate (fp32 or bf16-bit products); only fold order moves.

**MOJOLEARN_AFN_GEMM2_BIGTILE** (gemm, gemm-bf16). 128x64 tiles with 8
simdgroups of 32x32 (16 accumulators each) when the grid keeps >= 160 tiles.
Half the B staging per output cell and twice the MMAs per barrier; expect
gains at 4096^3. Risk: register pressure (16 accumulators + prefetch) lowers
occupancy.

**MOJOLEARN_AFN_GEMM2_DBUF** (gemm, gemm-bf16). Two threadgroup pages always
(KB 24 at 64x64), one barrier per window instead of two, next window staged
while this one multiplies. Expect fewer stalls; risk: shallower windows
(24 vs 32) mean more windows.

**MOJOLEARN_AFN_GEMM2_DIRECT_B** (gemm, gemm-bf16). B fragments read per lane
from device memory, no B page; A panel KB 64. Saves the B staging stores and
half the threadgroup traffic; risk: uncoalesced 8-byte loads per lane and B
re-read per simdgroup row; the per-lane fragment write is the k-NN idiom,
not yet used in a GEMM.

**MOJOLEARN_AFN_GEMM2_SWIZZLE** (gemm, gemm-bf16). Grouped block order (8 tile
rows) for L2 reuse; bits identical to arm A's (pure scheduling). Expect a
small gain at large grids, none at small ones.

**MOJOLEARN_AFN_GEMM2_ALL** (gemm, gemm-bf16, transformer-forward). All four:
128x64, KB 24, two A pages, direct B, swizzle. transformer-forward checks the
in-tree callers through `identical_gemm_into`.
