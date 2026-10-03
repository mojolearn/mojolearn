# afn-gemm A/B defines (binding linalg; board lanes gemm, gemm-bf16, gemm-int8)

Each line of `gemm.txt` races one FAST linalg build with the define against the
FAST build without it (arm A = today's FAST: MAX's vendor matmul for fp32, the
scalar tuned kernel for bf16, the sliced flat kernel for int8). Code:
`gemm/afn_apple_fast.mojo`; profile and entry-point map:
`docs/apple-fast/notes/neural-gemm.md`.

**MOJOLEARN_AFN_GEMM_SIMDGROUP** (gemm, gemm-bf16). fp32 products on Apple's
`simdgroup_matrix` 8x8 fp32 MMA from a FAST-only kernel: 64x64 tile, 4
simdgroups, 32-step windows staged through threadgroup memory, next window
prefetched into registers, no leaf fold stack, no per-window admission
reductions, no flush, no exact-step seam (the IDENTICAL matrix kernel carries
all four). Expected: the 4096^3 fp32 cell moves from MAX's matmul to our
kernel; whether that is faster is the measurement (the M3 Ultra peaks ~27
TFLOPS through this unit; the IDENTICAL scalar kernel read ~1 TFLOPS). Risk:
MAX's kernel may already be on the matrix unit with a better tile; a loss here
is a clean negative. On gemm-bf16 it replaces only the GEMM launch (the two
widen launches stay). Quality: exact products, f32 accumulate, fold order only.

**MOJOLEARN_AFN_GEMM_SPLITK** (gemm). The SIMDGROUP kernel plus split-k for
under-filled grids (fewer than 160 tiles): `grid.y` splits of whole windows
adding into a zeroed C with global f32 atomics, one zero launch and one GEMM
launch. At 4096^3 the grid is 4096 tiles and nothing splits, so on the board
this define should read as SIMDGROUP (a sanity pair: equal within noise). It
exists for the LM's weight gradients (384 x 384 x 2048: 36 tiles) and the
small projections; those lanes see it through the training binding when
the afn branches merge. Risk: atomic contention on tiny outputs; the order
across splits is nondeterministic (FAST allows it).

**MOJOLEARN_AFN_GEMM_TILESHAPE** (gemm). The SIMDGROUP kernel with the tile
picked from (m, n) on the host: 128x32 for n <= 32, 32x128 for m <= 32, 32x32
when the 64x64 grid is under 160 blocks, else 64x64. At 4096^3 it picks 64x64,
so on the board it should read as SIMDGROUP (second sanity pair). It is for
the decode rows and the under-filled gradients. Risk: none on the board.

**MOJOLEARN_AFN_GEMM_BF16_MMA** (gemm-bf16). Both bf16 operands go to the
matrix kernel as bits (uint16), widened by a 16-bit shift on the way into
threadgroup memory (exact, contract L-1). Removes the two widen launches, the
two float32 images (2 x 64 MB at the board shape) and the workspace growth
wait, and halves the global operand bytes. Expected: the largest relative win
of this lane, since today's bf16 path is widen + the scalar tuned kernel.
Quality: bf16 x bf16 products are exact in f32; f32 accumulate.

**MOJOLEARN_AFN_GEMM_INT8_MMA** (gemm-int8). `identical_gemm_int8_into` runs
the exact chunked matrix-unit kernel that lane lowbit-units wrote and gated
(`gemm/checks/gemm_int8_apple_chunk.mojo`): codes staged as float32, 1024-step
chunks so every partial sum stays below 2^24 and is exact, Int32 between chunks,
the flat kernel's dequantizing epilogue. One launch replaces 64 sliced launches
with 63 waits at 4096^3. Same Int32 per cell, same bits. Risk: the chunk
argument assumes the unit computes in IEEE fp32 at every step (probed on the
M4 and the M2 Pro; the M3 result is the measurement).

**MOJOLEARN_AFN_GEMM_EPILOGUE** (no board lane). Fused epilogue entry points
(bias, bias + residual, bias + SiLU, bias + GELU) for the attention, mamba and
MLP lanes to call next round; nothing calls them this round, so no request
line. It compiles and is reachable through `linalg_fast.matmul_fused` on an
EPILOGUE build.

**MOJOLEARN_AFN_GEMM_ALL** (all three lanes). Every define together. On gemm it
should equal SIMDGROUP; on gemm-bf16 it equals BF16_MMA; on gemm-int8 it equals
INT8_MMA. The three ALL lines are the merge candidate's own numbers.
