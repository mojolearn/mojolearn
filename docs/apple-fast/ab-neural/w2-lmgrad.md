# w2-lmgrad A/B requests: the byte LM block backward on the merged FAST GEMM

Binding `byte_lm`, shape `full` (M = 2048 tokens, d_model 384, ff 1024, 8 layers).
Branch `lane/apple-fast-neural-w2-lmgrad`. Every define is compiled only on a FAST
Apple build (training/byte_lm_afn_grad.mojo `AFN_LMGRAD_APPLE`); IDENTICAL and FAST
on NVIDIA or AMD compile main's launches. Nothing here has been compiled or run.

**MOJOLEARN_AFN_LM_WGRAD_SPLIT** (`afn-w2-lmgrad-wgrad-split-train`). The weight
gradients contract over the 2048 tokens into small outputs: 384x384 (q, k, v, o),
1024x384 (gate, up), 384x1024 (down), and 1x384 (the two norm weights, `ones .
dprod`). At the 64x64 tile those are 36, 96 and 6 blocks, fewer than the M3
Ultra's 80 cores twice over. Each now runs the gemm lane's simdgroup kernel with
`k` split over `grid.y` (5, 2 and 8 splits, at least 256 steps each): one zero launch
plus one GEMM launch whose splits add with f32 atomics. 72 products per step.
`_route_b`'s workspace sizing and the norm route's per-call `getenv` are bypassed.
Expected: a shorter backward from a filled GPU. Risk: atomic contention at 5 to 8
splits per cell; the zero launch adds 72 small launches per step. The split
order across blocks is nondeterministic (allowed under FAST). Arm A is plain FAST,
so the change also includes the switch from the vendor matmul to the simdgroup
kernel. `afn-w2-lmgrad-wgrad-vs-gemmsplitk-train` isolates the routing. Its arm A
is the gemm lane's SPLITK, which splits every under-filled GEMM through
`GemmWorkspace.run`. Its arm B is SIMDGROUP plus this route.

**MOJOLEARN_AFN_LM_BWD_EPILOGUE** (`afn-w2-lmgrad-bwd-epilogue-train`). Two
activation fan-ins follow a dA GEMM. `d(norm2.out) = gate_dA + up_dA` and
`d(norm1.out) = q_dA + k_dA + v_dA` now run in the BIAS_RESID store of the next
dA GEMM, with a bias of 384 zeros. That bias is the block's `dw_norm1`, zeroed once
per block and overwritten whole by the norm1 weight GEMM later in the same block.
The order stays q, k, v, left associative. Per layer this removes the add2 and add3
launches and about 4 full [2048 x 384] passes, and adds one tiny zero launch.
The fused dA GEMMs run on the simdgroup kernel whatever the gemm defines are.
Risk: small. On plain FAST, two or three dA GEMMs per fan-in move from the vendor
matmul to the simdgroup kernel, and that move can cost or gain more than the add.

**MOJOLEARN_AFN_LM_BWD_NORM1_RESID** (`afn-w2-lmgrad-norm1-resid-train`). The
input RMSNorm backward's dx kernel already has a `fuse_residual` arm, which norm2
uses. Norm1 now takes it too: `d_x = ftz(ftz(dx) + ftz(d_residual1))` is written
in the same kernel, which is bwd_add2_kernel's arithmetic and the same bits. The
block's last add2 launch goes: 8 launches and 8 passes of [2048 x 384] per step.
Risk: none expected.

**MOJOLEARN_AFN_LMGRAD_ALL** (`afn-w2-lmgrad-all-train`): the three together.
`afn-w2-lmgrad-all-on-stack-train` measures them on top of the wave-1 stack (lm
ALL plus gemm SIMDGROUP), where every other GEMM is already on the simdgroup kernel.

**lm-forward** (`afn-w2-lmgrad-fwd-*-forward`). The brief's FWD_EPILOGUE was
not built as a new define. The byte LM forward's GEMM tails (o_proj + residual,
gate/up + SiLU gate, down_proj + residual) live in
transformer/impl/llama/modeling_llama.mojo, which afn-attn owns. afn-attn's
FUSE_MLP already folds them into GEMM epilogues. Its helpers serve `forward_only`
calls, and the byte LM's logits path (training/byte_lm_logits.mojo, `forward_only=True`)
is one. These two lines test whether those defines reach the byte_lm binding's
lm-forward. In the train step `forward_only` is False, so they do not apply there:
the backward needs the pre-activations they skip writing. The byte LM's own forward
GEMM, the head, has no bias, residual or activation tail.
