# w2-lmgrad notes: the byte LM block backward on the merged FAST GEMM

Lane w2-lmgrad, branch `lane/apple-fast-neural-w2-lmgrad` (base
lane/apple-fast-neural @ 675a74a28), 2026-10-03. This file covers code reading
and writing only. Nothing was compiled, run or measured.

## The block backward's GEMMs and fan-ins (one layer, FAST, trace off)

Read from transformer/checks/transformer_backward.mojo `llama_decoder_layer_backward_device`.
Shapes are at the board's (M 2048, dm 384, it 1024, qw = kw = 384).

| stage | product / op | shape (m', n', k') | route on base |
|---|---|---|---|
| 2 | down dA | 2048 x 1024 x 384 | `_route_a` -> GemmWorkspace.run |
| 3 | down dW | 384 x 1024 x **2048** | `_route_b` |
| 4-6 | SwiGLU VJP | elementwise | 1 to 2 kernels |
| 7, 8 | gate, up dW | 1024 x 384 x **2048** | `_route_b` x2 |
| 9 | gate, up dA, then add2 | 2048 x 384 x 1024 x2 | 2 GEMMs + **add2** |
| 10-12 | norm2 bwd (+ residual1 fused under BWD_FUSE) | | 2-3 kernels, then dW `ones . dprod` 1 x 384 x **2048** |
| 13 | residual1 add (not fused) | | add2 |
| 15, 16 | o dA, o dW | 2048x384x384, 384x384x**2048** | |
| 17-28 | attention backward, RoPE | | afn-attn / main |
| 29-31 | q, k, v dW | 384 x 384 x **2048** x3 | `_route_b` x3 |
| 32 | q, k, v dA, then add3 | 2048 x 384 x 384 x3 | 3 GEMMs + **add3** |
| 33-35 | norm1 bwd | | 2-3 kernels, then dW 1 x 384 x **2048** |
| 36 | d_x = norm1_dx + d_residual1 | | **add2** |

Per layer: 7 projection dW and 2 norm dW GEMMs contract over the tokens. At
64x64 tiles they are 36, 96, 96, 36 and 6 blocks: none fills 2 x 80 cores. There
are 3 fan-in adds after GEMMs or norms. Host waits: none on these stages (the
norm fences are lm's BWD_NOSYNC). The norm dW route reads an env var per call
(`_norm_dw_own_workspace`), 16 host getenv calls per step.

## Candidates (training/byte_lm_afn_grad.mojo; arms in transformer_backward.mojo)

| define | mechanism | where |
|---|---|---|
| `MOJOLEARN_AFN_LM_WGRAD_SPLIT` | `_route_b` and `bwd_rms_norm_routed`'s dW go to `afn_lm_wgrad_split_into`. That is the gemm lane's `afn_zero_kernel` plus `_afn_launch_tile[f32, f32, SPLIT=True]`, with this lane's split policy: the grid aims at `MOJOLEARN_AFN_LM_WGRAD_BLOCKS` (default 160), at least 256 steps per split, at most 16 splits. It falls through when a product does not split. It is independent of the `MOJOLEARN_AFN_GEMM_*` defines. With lm's PARAM_VIEWS the outputs are the flat-gradient views. | `_route_b`, `bwd_rms_norm_routed`, `_afn_wgrad` |
| `MOJOLEARN_AFN_LM_BWD_EPILOGUE` | up dA writes `d_norm2_out = up_dA + tmp0` (gate dA). k dA writes `tmp1 = k_dA + tmp0` (q dA). v dA writes `d_norm1_out = v_dA + tmp1`. Each is the `AFN_EPI_BIAS_RESID` store through `_afn_launch_tile[..., AFN_EPI_BIAS_RESID]`, and the bias is `dw_norm1` zeroed once per block. Per layer: -2 adds, +1 zero launch. | stages 7-9 and 32 |
| `MOJOLEARN_AFN_LM_BWD_NORM1_RESID` | The norm1 `bwd_norm_dx_kernel` call passes `residual_out = d_x`, `residual_branch = d_residual1`, `fuse_residual = True`, and the stage-36 add2 is not compiled. The bits are the same as add2. | stages 33-36 |
| `MOJOLEARN_AFN_LMGRAD_ALL` | all three | |

Guard: `AFN_LMGRAD_APPLE` = FAST, `has_apple_gpu_accelerator()`, not
`MOJOLEARN_COLUMN_CPU`, and `TARGET_COLUMN == COLUMN_APPLE`, and in
transformer_backward also `not BWD_ANY_SABOTAGE`. IDENTICAL compiles main's launches.
In stages 7-9 and 32, main's lines sit under `if not afn_epi:`, and `afn_epi`
is the constant False unless BWD_EPILOGUE is compiled in.

## Gaps and requests (not built)

- **SwiGLU VJP in the down dA epilogue.** No epilogue kind fits. The VJP reads
  `gate_proj`, `up_proj` and `silu_out` and writes two outputs (`d_gate`, `d_up`).
  Request for the gemm lane: an `AFN_EPI_SWIGLU_BWD` kind (a second output pointer,
  three extra inputs per cell) on the non-split kernel.
- **A bias-free residual epilogue** (`AFN_EPI_RESID`). It would drop the zero-bias
  launch and the `dw_norm1` borrow. BIAS_RESID with zeros adds `+0.0`, which turns a
  `-0.0` dA cell into `+0.0` when the residual is `-0.0`: numerically equal, but not
  the add2 bits.
- **An accumulate-into (beta = 1) split.** The split route zeroes C first. A
  `C += A.B` form would let one dW GEMM add into a gradient that already holds a value
  (microbatch accumulation). The byte LM does not need it today.
- **FWD_EPILOGUE**: see ab-neural/w2-lmgrad.md. The forward tails are in
  modeling_llama.mojo (afn-attn's FUSE_MLP and FUSE_PRE). They reach lm-forward
  through `forward_only=True` but not the train step, whose backward needs the
  pre-activations. No define was added.

## Compile table

| build | rc |
|---|---|
| FAST `-D MOJOLEARN_AFN_LM_WGRAD_SPLIT` | UNCOMPILED (peer compiles) |
| FAST `-D MOJOLEARN_AFN_LM_BWD_EPILOGUE` | UNCOMPILED (peer compiles) |
| FAST `-D MOJOLEARN_AFN_LM_BWD_NORM1_RESID` | UNCOMPILED (peer compiles) |
| FAST `-D MOJOLEARN_AFN_LMGRAD_ALL` | UNCOMPILED (peer compiles) |
| FAST, no define | UNCOMPILED (peer compiles) |
| IDENTICAL | UNCOMPILED (peer compiles) |

Build: `MOJOLEARN_COMPILE_JOBS=1 MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_SKIP_BUILD_GATE=1
MOJOLEARN_BYTE_LM_OUTDIR=<fresh> MOJOLEARN_MOJO_BUILD_FLAGS="-D <define>" sh bindings/build_byte_lm.sh`.
The transformer, samba and offload/layer-pool paths import transformer_backward,
so the same defines also reach those bindings' backward.

Compile risks to check first:
- training/byte_lm_afn_grad.mojo imports the gemm lane's underscore names
  `_afn_launch_tile` and `_afn_strides`. Mojo does not enforce privacy, but if
  the compiler rejects them, the fix is a public wrapper in gemm (owned by afn-gemm).
- `afn_lm_zero_into` raises in its `comptime if not AFN_LMGRAD_APPLE` arm and
  returns `None` otherwise.
- `_afn_ptr(...)` is called several times as arguments of one call on fields of
  the same `mut bst`. Each returns a `MutAnyOrigin` pointer, the same idiom as
  modeling_llama's `_afn_p`.
