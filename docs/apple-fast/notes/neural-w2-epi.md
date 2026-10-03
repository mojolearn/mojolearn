# w2-epi: fused GEMM epilogues in the Mamba blocks and the Samba head (wave 2, 2026-10-03)

Branch `lane/apple-fast-neural-w2-epi`, base `lane/apple-fast-neural` @ 675a74a28. Bindings mamba and
training. Code: `mamba/impl/modules/afn_proj_gemm.mojo` (new), call sites in
`mamba/impl/modeling/modeling_mamba.mojo`, `mamba/impl/modules/mamba2.mojo`, `mamba/impl/modules/mamba3.mojo`,
and `training/samba_afn.mojo`. The gemm kernel (`gemm/afn_apple_fast.mojo`) is read and launched, never edited.

## Profile (code reading, per block call, board shapes)

| site | today on Apple FAST | shape (m x n x k) |
|---|---|---|
| in_proj (Mamba-1/2/3) | `identical_gemm[False]`: vendor route closed, scalar `PLAN_TUNED_128_8X8` kernel (the matrix unit only with `MOJOLEARN_AFN_GEMM_SIMDGROUP`) | 2048 x d_in_proj x 384 (Samba: 1024 x d_in_proj x 384) |
| out_proj | the same scalar kernel, writes the `out_proj` stage | 2048 x 384 x 768 (Samba: 1024 x 384 x 768, 96 64x64 tiles) |
| residual | `residual_add_kernel` launch reading `x` and `out_proj`, then the stage wait | m x 384 |
| Samba head forward | `identical_gemm_into` + workspace buffer | 1024 x 256 x 384 |
| Samba head dA / dW | two `identical_gemm_into` calls + two workspace buffers | dA 1024 x 384 x 256; dW 256 x 384 x 1024 (24 tiles) |

Per block: 2 scalar GEMM launches + 1 residual launch; per Samba tail: 3 GEMMs, 3 workspace allocations.

## Candidates

| define | mechanism | files |
|---|---|---|
| `MOJOLEARN_AFN_MAMBA_PROJ_EPILOGUE` | in_proj on the matrix-unit kernel; out_proj fused with the residual (`residual_out` seeded with `x` by one copy, kernel adds into it); traced runs and declined shapes keep main's steps | afn_proj_gemm.mojo; modeling_mamba.mojo (mixer in_proj/out_proj, block residual); mamba2.mojo, mamba3.mojo (in_proj, out_proj + residual) |
| `MOJOLEARN_AFN_MAMBA_PROJ_SPLITK` | the same route; out_proj splits k over grid.y when tiles < 2 x AFN_GEMM_CORES, every split adding into the `x` seed | afn_proj_gemm.mojo (`afn_proj_k_split`) |
| `MOJOLEARN_AFN_SAMBA_HEAD_GEMM` | head GEMM + dA + dW on the matrix-unit kernel, no workspaces; dW splits k into a zeroed output | training/samba_afn.mojo |
| `MOJOLEARN_AFN_EPI_ALL` | all three | |

Not delivered: the z-gate SiLU in the in_proj epilogue. The kernel stores `c[i * n + j]` (row stride = n), so
it cannot write the z column slice of the wider in_proj row; a separate z launch with the SiLU epilogue into
its own buffer would add a launch and a live buffer while the gate kernel still launches for `y * silu(z)`.
The gemm kernel's BIAS epilogues all need a bias buffer, which no projection here has (all bias-free).

## Guard

`AFN_W2EPI_APPLE = FAST and has_apple_gpu_accelerator() and not MOJOLEARN_COLUMN_CPU and TARGET_COLUMN ==
COLUMN_APPLE` (the gemm kernel's own guard); Samba: `AFN_SAMBA_APPLE_FAST and TARGET_COLUMN == COLUMN_APPLE`.
Every call site is `comptime if <switch>:` with main's spelling as the `else` arm; the entry points return
False before touching anything when the switch is off, and the gemm kernel is instantiated only inside the
switched arms. IDENTICAL compiles main's code unchanged.

## Watch for in the peer's builds

- `_afn_launch_tile`, `_afn_strides` are underscore names imported from `gemm/afn_apple_fast.mojo`.
- The fused residual uses the kernel's `SPLIT = True` store (f32 `Atomic.fetch_add`) with `splits = 1` when
  not split; the gemm lane compiled that instantiation under `MOJOLEARN_AFN_GEMM_SPLITK`.
- `ctx.enqueue_copy(dst_buf=residual_out, src_buf=x)` copies whole buffers; the route declines unless both
  hold exactly `m n` floats (guard-banded poison builds fall back).

## Compile results

| build | status |
|---|---|
| mamba FAST + each of PROJ_EPILOGUE, PROJ_SPLITK, EPI_ALL | UNCOMPILED (peer compiles) |
| training FAST + SAMBA_FUSE + SAMBA_HEAD_GEMM; + EPI_ALL | UNCOMPILED (peer compiles) |
| mamba / training FAST no define | UNCOMPILED (peer compiles) |
| mamba / training IDENTICAL | UNCOMPILED (peer compiles) |
