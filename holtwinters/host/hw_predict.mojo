# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SHIPS: compiled into a CPU host binding (python/mojolearn/host_surface.py names which); product, not only a check.
"""Holt-Winters prediction from the fitted state alone (lane/inference-holtwinters,
2026-09-15): the forecast and the in-sample one-step predictions, with no fit,
no decomposition and no line search in reach.

`hw_forecast_from_state` is the body `hw_oracle.mojo::oracle_forecast` calls
(`HoltWintersForecastHelper`: the last fitted row of level and trend and the
last `frequency` rows of season), so the reference host binding, the inference
binding and the oracle the device is held to share ONE spelling.

`hw_predict_in_sample` is the one-step prediction the fit's final evaluation
computes and does not store (`hw_eval.mojo`'s header):

    leveltrend = ftz(plevel + ptrend)
    xhat       = ftz(leveltrend + stmp)   (additive) | ftz(leveltrend * stmp)

The fitted components are indexed `i = t - frequency` (the evaluation's shift
is the frequency). At step `i >= frequency` every input is a stored component:
`plevel = level[i - 1]`, `ptrend = trend[i - 1]` and `stmp = season[i -
frequency]` (the scratch season of phase `i % frequency` was last written at
step `i - frequency`). Steps `i < frequency` read the start level, trend and
season of the decomposition, which the fit does not return, so times
`t < 2 * frequency` are the canonical quiet NaN, by name, as ARIMA's in-sample
prediction is NaN before `d + s*D`.
"""

from std.memory import bitcast

from checks.numerics import ftz, identical_mul_add


@always_inline
def _f[dt: DType](x: Scalar[dt]) -> Scalar[dt]:
    comptime if dt == DType.float32:
        return rebind[Scalar[dt]](ftz(rebind[Float32](x)))
    else:
        return x


@always_inline
def _mad[dt: DType](a: Scalar[dt], b: Scalar[dt], c: Scalar[dt]) -> Scalar[dt]:
    comptime if dt == DType.float32:
        return rebind[Scalar[dt]](
            identical_mul_add(
                rebind[Float32](a), rebind[Float32](b), rebind[Float32](c)
            )
        )
    else:
        return a * b + c


def hw_forecast_from_state[
    dt: DType
](
    level: List[Scalar[dt]],
    trend: List[Scalar[dt]],
    season: List[Scalar[dt]],
    n: Int,
    batch_size: Int,
    frequency: Int,
    additive: Bool,
    h: Int,
) -> List[Scalar[dt]]:
    """`h x batch_size`, time-major. The components are time-major,
    `(n - frequency) x batch_size` each."""
    var bs = batch_size
    var f = frequency
    var n_minus = n - f
    var lt_shift = (n_minus - 1) * bs
    var s_shift = (n_minus - f) * bs
    var out = List[Scalar[dt]]()
    out.reserve(h * bs)
    for _ in range(h * bs):
        out.append(Scalar[dt](0))
    for s in range(bs):
        var lv = level[lt_shift + s]
        var tr = trend[lt_shift + s]
        for i in range(h):
            var sn = season[s_shift + s + (i % f) * bs]
            var lt = _f[dt](_mad[dt](tr, Scalar[dt](i + 1), lv))
            if additive:
                out[s + i * bs] = _f[dt](lt + sn)
            else:
                out[s + i * bs] = _f[dt](lt * sn)
    return out^


def hw_forecast_from_state_ptr(
    components: MutPointer[Float32, MutUntrackedOrigin],
    components_len: Int,
    n: Int,
    batch_size: Int,
    frequency: Int,
    additive: Bool,
    h: Int,
) -> List[Float32]:
    """The float32 forecast above, reading an already-packed host model.

    Host bindings own the packed component buffer.  Reading it in place is
    important for inference: a forecast needs the final level/trend row and
    one seasonal cycle, not copies of all ``3 * (n - frequency) * batch``
    fitted-history values.
    """
    var bs = batch_size
    var f = frequency
    var n_minus = n - f
    var lt_shift = (n_minus - 1) * bs
    var s_shift = (n_minus - f) * bs
    var out = List[Float32](length=h * bs, fill=Float32(0.0))
    for s in range(bs):
        var lv = components.unsafe_load(lt_shift + s)
        var tr = components.unsafe_load(components_len + lt_shift + s)
        for i in range(h):
            var sn = components.unsafe_load(
                2 * components_len + s_shift + s + (i % f) * bs
            )
            var lt = ftz(identical_mul_add(tr, Float32(i + 1), lv))
            if additive:
                out[s + i * bs] = ftz(lt + sn)
            else:
                out[s + i * bs] = ftz(lt * sn)
    return out^


def hw_predict_in_sample(
    level: List[Float32],
    trend: List[Float32],
    season: List[Float32],
    n: Int,
    batch_size: Int,
    frequency: Int,
    additive: Bool,
    start: Int,
    end: Int,
) -> List[Float32]:
    """The one-step predictions at times `[start, end)`, `0 <= start < end <=
    n`, `(end - start) x batch_size`, time-major. NaN where `t < 2 *
    frequency` (see the module docstring)."""
    var bs = batch_size
    var f = frequency
    var ld = end - start
    var qnan = bitcast[DType.float32](UInt32(0x7FC00000))
    var out = List[Float32]()
    out.reserve(ld * bs)
    for _ in range(ld * bs):
        out.append(qnan)
    for k in range(ld):
        var t = start + k
        if t < 2 * f:
            continue
        var i = t - f
        for s in range(bs):
            var leveltrend = ftz(
                level[s + (i - 1) * bs] + trend[s + (i - 1) * bs]
            )
            var stmp = season[s + (i - f) * bs]
            if additive:
                out[s + k * bs] = ftz(leveltrend + stmp)
            else:
                out[s + k * bs] = ftz(leveltrend * stmp)
    return out^


def hw_predict_in_sample_ptr(
    components: MutPointer[Float32, MutUntrackedOrigin],
    components_len: Int,
    n: Int,
    batch_size: Int,
    frequency: Int,
    additive: Bool,
    start: Int,
    end: Int,
) -> List[Float32]:
    """The float32 in-sample prediction above over packed host pointers."""
    var bs = batch_size
    var f = frequency
    var ld = end - start
    var qnan = bitcast[DType.float32](UInt32(0x7FC00000))
    var out = List[Float32](length=ld * bs, fill=qnan)
    for k in range(ld):
        var t = start + k
        if t < 2 * f:
            continue
        var i = t - f
        for s in range(bs):
            var leveltrend = ftz(
                components.unsafe_load(s + (i - 1) * bs)
                + components.unsafe_load(components_len + s + (i - 1) * bs)
            )
            var stmp = components.unsafe_load(
                2 * components_len + s + (i - f) * bs
            )
            if additive:
                out[s + k * bs] = ftz(leveltrend + stmp)
            else:
                out[s + k * bs] = ftz(leveltrend * stmp)
    return out^
