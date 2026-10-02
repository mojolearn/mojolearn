# lane/apple-fast-kapprox: the chi2 samplers' fit on the device

Written without a Mojo toolchain (cloud peer); the first M3 build is the compile check.
One switch, compiled under FAST + Apple only and default OFF; IDENTICAL compiles main's code.
gaussian-rp is `lane/apple-fast-decomp-sparse`'s (`MOJOLEARN_DECOMP_FAST_RP_DIRECT`), not touched here.

| switch | kind | site | what it changes under FAST on Apple |
|---|---|---|---|
| `-D MOJOLEARN_KAPPROX_DEVICE=1` | build define (`x_neighbors` binding), read back by the Python layer through `x_neighbors_kapprox_fast()` | `x_neighbors/kapprox_dev.mojo` `XN_KAPPROX_FAST`; `python/mojolearn/_expansion_neighbors.py` `AdditiveChi2Sampler`, `SkewedChi2Sampler` | fit and transform run as kapprox ops: the samplers' refusal (`x < 0`, `x <= -skewedness`) is a device flag over every cell instead of the host `X.min()`; `SkewedChi2Sampler.fit` draws `random_weights_` (d x n_components) and `random_offset_` on the device from a counter-based uniform, one thread per draw, instead of d x n_components Python draws of the legacy MT19937 stream; `SkewedChi2Sampler.transform` takes the log into a device scratch and runs the product + cosine on the same stream (one upload of X, one download), instead of a `unary` op, a download, and a second upload; `AdditiveChi2Sampler.transform` fuses the check into the map launch |

Why: the board clocks `fit(X)` at 100k rows for both samplers (`task="transform"`:
`ms` = fit, `infer_ms` = transform of 1000 rows), and the fit of both was host work only:
`_f32(X)` then `X.min()` over every cell (additive-chi2 istella 11.0 s FAST = 11.0 s IDENTICAL,
sklearn 3.8 s) and, for the skewed sampler, the Python draws plus a tiny `skew_weights` op
(skewed-chi2 taxi 2.7 s, sklearn 0.6 s). No device work was on the clock to make faster;
the change moves the clocked work onto the device.

Bits: FAST only. The skewed sampler's fitted parameters are i.i.d. uniforms of the same law
as sklearn's, not sklearn's numbers (the serial MT19937 stream would be a one-thread launch);
the quality metric (relative Frobenius error of Z Z^T against the exact skewed chi2 kernel)
is a statistic of the draw, so it must land within FAST's run-to-run spread. The additive map
is the same statements as main's `achi2_item`. NaN cells pass both checks, as the comparisons
on sklearn's `X.min()` do.

Files: `x_neighbors/kapprox_items.mojo` (items), `x_neighbors/kapprox_dev.mojo` (GPU drivers +
readback), `x_neighbors/kapprox_host.mojo` (the CPU binding's twin), `x_neighbors/gen.py`
(OWN_DRIVERS entries + the readback export; bindings and `_surface_neighbors.py` regenerated,
`device_ops.mojo` / `host_ops.mojo` unchanged), `python/mojolearn/_expansion_neighbors.py`.

Risky compile sites (no toolchain here):
- `kapprox_items.mojo` `kapprox_uniform`: `UInt32(seed)` / `UInt32(t)` from `Int`, the wrapping
  `UInt32` multiplies with literals above 2^31 (`UInt32(0x846CA68B)`), and
  `(h >> 8).cast[DType.float32]()`.
- `kapprox_dev.mojo`: `comptime if not XN_KAPPROX_FAST: _refuse(); return` ahead of the launches
  (the `op_group_mean` shape in device_ops.mojo); `_buf(ctx, 0, n * d, False)` for the scratch;
  `enqueue_function[...](d_x.unsafe_ptr(), ...)` as device_ops.mojo spells it; the import of
  `skew_transform_kernel` from device_ops.mojo (iter_device.mojo imports other kernels the same way).
- `kapprox_host.mojo`: `IP(unsafe_from_address=flag)` and the `List[Float32]` scratch
  (host_ops.mojo's `op_pcs` shape).

Requests (`kapprox.txt`): `kap-schi2-taxi` (skewed-chi2 taxi) and `kap-achi2-istella`
(additive-chi2 istella), old FAST vs the define, one alternation. After a win: skewed-chi2
istella (1.84x) and additive-chi2 taxi (1.50x) with the same define.
