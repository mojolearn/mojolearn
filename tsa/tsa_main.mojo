# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The KPSS test and the `d` choice on one hashed batch, with an identity card.

    tools/with_build_lock.sh     pixi run mojo run -I . tsa/tsa_main.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . tsa/tsa_main.mojo
    MOJOLEARN_IDENTITY_TRACE=/tmp/tsa.card tools/with_identical_mode.sh pixi run mojo run -I . tsa/tsa_main.mojo

    python3 tools/identity_trace_diff.py /tmp/tsa.mac.card /tmp/tsa.other.card

The card (`core/identity_trace.mojo`) carries the INPUT bytes first, then
every per-series stage of the test at (d, D, s) = (1, 0, 0) -- `tsa.diff`,
`tsa.mean`, `tsa.resid` (the centered series), `tsa.s2A`, `tsa.s2B`,
`tsa.cumsum`, `tsa.eta`, `tsa.stat`, `tsa.flags` -- and last `tsa.d`, the
chosen differencing order per series (an INTEGER stage; read it first when
two cards disagree, because a different `d` is a different model, not a
rounding). Tags are unique and carry no launch parameter.

Not a port: cuML ships one backend and needs no card. This driver is a
CONSTRUCTION plus one Apple device's run; no second vendor has run it.
"""

from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace
from original.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name
from tsa.original.fixtures import bits32, download_f32, kpss_fixture, upload_f32
from tsa.derived.timeSeries.stationarity import download_results
from tsa.derived.tsa.auto_arima import select_d
from tsa.derived.tsa.stationarity import kpss_test


comptime IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
comptime N_OBS = 520
comptime SALT = 1


def _mode_name() -> String:
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29; see the note
    on that function. A local two-way IDENTICAL-or-FAST answers "FAST"
    for a DETERMINISTIC build, which mislabels every line the driver
    prints.
    """
    return numeric_mode_name()


def main() raises:
    print("== tsa/tsa_main.mojo [" + _mode_name() + "] ==")
    var ctx = DeviceContext()
    var trace = IdentityTrace()
    trace.header("tsa_main mode=" + _mode_name() + " n_obs=" + String(N_OBS) + " batch=8 salt=" + String(SALT))
    var f = kpss_fixture(N_OBS, SALT)
    var y = upload_f32(ctx, f.y)
    trace.record_device[DType.float32](ctx, "tsa.input", y, f.batch_size * f.n_obs)

    var res = kpss_test(ctx, y, f.batch_size, f.n_obs, 1, 0, 0, 0.05)
    var n = res.n_obs_diff
    trace.record_device[DType.float32](ctx, "tsa.diff", res.y_diff, f.batch_size * n)
    trace.record_device[DType.float32](ctx, "tsa.mean", res.scratch.y_means, f.batch_size)
    trace.record_device[DType.float32](ctx, "tsa.resid", res.scratch.y_cent, f.batch_size * n)
    trace.record_device[DType.float32](ctx, "tsa.s2A", res.scratch.s2A, f.batch_size)
    trace.record_device[DType.float32](ctx, "tsa.s2B", res.scratch.s2B, f.batch_size)
    trace.record_device[DType.float32](ctx, "tsa.cumsum", res.scratch.accumulator, f.batch_size * n)
    trace.record_device[DType.float32](ctx, "tsa.eta", res.scratch.eta, f.batch_size)
    trace.record_device[DType.float32](ctx, "tsa.stat", res.scratch.stat, f.batch_size)
    trace.record_device[DType.uint8](ctx, "tsa.flags", res.scratch.results, f.batch_size)
    var rs = download_results(ctx, res, f.batch_size)
    var flags = rs[0].copy()
    var stats = rs[1].copy()
    for b in range(f.batch_size):
        print("  " + f.names[b] + ": stat " + String(stats[b]) + " " + bits32(stats[b])
              + " stationary(d=1) " + String(flags[b]))

    var d = select_d(ctx, y, f.batch_size, f.n_obs, 0, 0, 2, 0.05)
    trace.record_list_i32("tsa.d", d)
    for b in range(f.batch_size):
        print("  " + f.names[b] + ": d = " + String(d[b]))
    print("tsa_main done [" + _mode_name() + "]" + (" card: " + trace.path if trace.enabled else " (no card: set MOJOLEARN_IDENTITY_TRACE)"))
    _ = y^
