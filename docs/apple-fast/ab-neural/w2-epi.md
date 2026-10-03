# w2-epi: what each define does (for reading the A/B results)

Branch `lane/apple-fast-neural-w2-epi`. Every define is FAST + Apple column only, default off. Nothing was
compiled or run by the lane (the peer compiles).

**MOJOLEARN_AFN_MAMBA_PROJ_EPILOGUE** (binding mamba; mamba1/2/3-forward, and the Mamba-3 layers of
samba-forward and samba-train-step). in_proj and out_proj of every Mamba block run on the gemm lane's FAST
matrix-unit kernel (`afn_gemm_mma_kernel`, 64x64 tile, simdgroup 8x8 f32 MMA) instead of
`identical_gemm[False]`, which on Apple FAST is the scalar tuned kernel. out_proj also takes the block's
residual: `residual_out` is seeded with `x` by one device copy, and the kernel adds its tile into it (one f32
atomic add per cell), so the separate residual kernel never launches and the out_proj stage is never written.
Expected: the two largest GEMMs of each block move from scalar FMA to the matrix unit (the main effect), and
one compute launch per block becomes a blit. Risk: the atomic store path is slower per cell than a plain
store (one add per cell, no contention); a traced run keeps main's two steps. Quality: f32 throughout; only
the fold order of the products moves.

**MOJOLEARN_AFN_MAMBA_PROJ_SPLITK** (binding mamba; samba-forward, samba-train-step). The same route, plus a
split of out_proj's `k = d_inner` over `grid.y` when its tiles cover fewer than 2 x 80 blocks, every split
adding into the `x` seed (no zero launch, no fold launch). At the mamba board shape (m = 2048, 192 tiles) it
never splits, so the lines run on the Samba lanes (m = 1024, 96 tiles, 2 splits of 384), with
PROJ_EPILOGUE as arm A so the A/B isolates the split. Risk: two splits double the atomic adds per cell; the
split partials add in a free order (FAST allows it).

**MOJOLEARN_AFN_SAMBA_HEAD_GEMM** (binding training; samba-forward, samba-train-step). The tied head GEMM
(`logits = hn . W^T`, 1024 x 256 x 384) and its two backward GEMMs (`dA = dC . W`, `dW = dC^T . hn`, from
`gemm_backward_*_call`'s operand table) on the matrix-unit kernel instead of `identical_gemm_into` (no
workspaces allocated). dW (24 tiles, k = 1024 tokens) splits k four ways into a zeroed output with f32
atomics. Reached only through afn-samba's fused entries and resident head loss, so both arms carry
`MOJOLEARN_AFN_SAMBA_FUSE`. Risk: low; the head is a small share of the step.

**MOJOLEARN_AFN_EPI_ALL**: all three. The mamba-binding lines compare it with no define; the training line
compares FUSE against FUSE + EPI_ALL.
