# lane/apple-fast-tsa2: VAR, STL and KPSS under FAST on Apple (GAPS-m3-fast-oct2: var 5.2x, stl 3.0x, kpss 1.2x)

Written without a Mojo toolchain (cloud peer); the first M3 builds of `x_sequence` and `tsa` (FAST) are the compile
check. Every change is compiled under FAST + Apple only and defaults OFF; IDENTICAL compiles main's code unchanged.
Bindings for `afc_ab_def.sh`: `x_sequence` (var, stl), `tsa` (kpss). Second dataset (synthetic) only after taxi-hourly wins.

| switch | what it changes under FAST on Apple | files |
|---|---|---|
| `-D MOJOLEARN_TSA2_VAR=1` (`TSA2_VAR` in `sequence/ops.mojo`) | `var_fit_py` -> `_var_fit_queued`: `ex.bind` for y (no zero fill), ONE workspace alloc for the nine buffers (one fill), eight launches queued with nothing between them (design, column scale, Z'Z and Z'Y GEMMs, threadgroup Cholesky solve, fused residual `op_var_resid` = GEMM chain then `sub`, fused `op_var_sigma` = GEMM chain then the 1/(R-m) `mul`, row scale), then the status word AND the three outputs as `download_async` on one `sync`. Main: sync after the Cholesky, a blocking status download, three async downloads and a sync (4 waits + the executor's drain). On a non-positive pivot the outputs now hold finite unsolved products instead of main's untouched zeros; the Python layer raises and never reads them. `var_forecast_py` binds its two inputs. `DeviceExec.launch` takes the threadgroup VAR kernels without the `MOJOLEARN_SEQ_VAR_BLOCK` env read. | `sequence/pyapi.mojo` (`_var_fit_queued`, `var_forecast_py`), `sequence/vecar.mojo` (two new ops), `sequence/exec_device.mojo` (guarded dispatch), `sequence/ops.mojo`, `sequence/dispatch.mojo`, `sequence/exec.mojo` (weights) |
| `-D MOJOLEARN_TSA2_STL=1` (`TSA2_STL` in `sequence/ops.mojo`) | `stl_py` -> `_stl_grid_py` when every jump is 1, `outer_iter == 0` and `n >= 2 period` (the board: period 24, seasonal 7, non-robust, inner 5): per inner pass seven launches over the batch, no wait: `op_stl_seas` (a thread per slot of the extended seasonal series `[B, n + 2 np]`: subseries j, position m; the `_ess` point, or the two extrapolated ends; detrending on the fly), `op_stl_ma` x3 (direct window sums / len), `op_stl_loess` (low pass, window `low_pass`), `op_stl_deseas` (season and the deseasonalised series), `op_stl_loess` (trend, window `trend`); `op_stl_finish` (residual, unit weights); four `download_async` on one `sync`. Main: `op_stl`, one thread per series, 64 threads for 1,440 points x 5 passes. Bits: LOESS points reproduce `stl_est`'s chain (weights recomputed per pass by the same operations); the moving averages are direct sums where main runs a serial running sum, FAST only. | `sequence/stl_grid.mojo` (new), `sequence/pyapi.mojo` (`_stl_grid_py`), `sequence/ops.mojo`, `sequence/dispatch.mojo`, `sequence/exec.mojo` |
| `-D MOJOLEARN_TSA2_KPSS=1` (`TSA2_KPSS` in `tsa/impl/timeSeries/kpss_fused.mojo`) | `kpss_test_host` for `d = D = 0` and `n_obs <= 4096`: the upload queued (no wait), one launch `kpss_fused_kernel` (a block of 256 per series, the centred series in 16 KB of threadgroup memory: mean, s2A, s2B in the eight-launch path's fold order; the partial sums as a chunked block scan + `pinned_block_prefix_sum`, eta from the running sums: FAST's fold of the serial scan), the finite check in the kernel (a per-series index word, raised by name after the wait), three downloads on one `synchronize`. Main: upload wait, host finite scan with its wait, eight launches and a wait, download wait. | `tsa/impl/timeSeries/kpss_fused.mojo` (new), `tsa/estimator.mojo` (guarded branch) |

## Risky compile sites

- `sequence/ops.mojo`: new imports `has_apple_gpu_accelerator` (std.sys.info) and `GLOBAL_NUMERIC_MODE`, `NUMERIC_FAST`
  (checks.numerics) at module level of a file both bindings compile; `_TSA2_FAST_APPLE` is a comptime Bool.
- `sequence/stl_grid.mojo`: `stl_est_point` returns `Tuple[Bool, Float32]` as `stl_est` does; `_ess_window` returns
  `Tuple[Int, Int]` (read as `w[0]`, `w[1]`); `var v: Float32` assigned in every branch of `op_stl_seas`.
- `sequence/pyapi.mojo`: `comptime if TSA2_STL:` / `comptime if TSA2_VAR:` with a `return` inside a runtime `if` (the
  helper functions are generic over `E: Exec`, as the callers); `ex.bind` and `download_async` are trait methods; the
  status word via `FP(unsafe_from_address=Int(st.unsafe_ptr()))` as main's `var_fit_py`.
- `sequence/exec_device.mojo`: `comptime if TSA2_VAR and OP == OP_CHOLSOLVE:` before main's `comptime if OP == ...`
  chain (an additive block; `TSA2_VAR` imported from `sequence.ops`).
- `tsa/impl/timeSeries/kpss_fused.mojo`: kernel parameters `MutPointer[T, MutAnyOrigin]` launched with bare
  `buf.unsafe_ptr()` through `enqueue_function`, the idiom of the sibling kernels in `stationarity.mojo`;
  `pinned_block_prefix_sum[KPSS_FUSED_TPB, exclusive=True]` (block of 256, a multiple of 32); `isfinite` from
  `std.math` on a device `Float32`; `stack_allocation[..., Scalar[DType.int32], address_space = AddressSpace.SHARED]`
  as `sequence/vecar_block.mojo`.
- `tsa/estimator.mojo`: the guarded branch returns after `kpss_fused(ctx, ...)` without `_ = ctx^` (the context's last
  use on that path is the call; the helper's buffers die inside it).
