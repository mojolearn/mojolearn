# SPDX-License-Identifier: Apache-2.0
"""Typed KPSS flag and select_d loops moved word for word from bindings/kpss_host_test.mojo."""
from std.sys.compile import is_defined
from std.math import isfinite
from core.hostptr import f32_ptr, i32_ptr, read_f32
from tsa.checks.kpss_oracle import kpss_host_f32


# A reduction-order perturbation can change the statistic without crossing the
# stationarity threshold. This separate negative control targets the decision
# consumed by select_d. Production builds contain neither arm.
comptime KPSS_DECISION_SABOTAGE = is_defined["MOJOLEARN_KPSS_DECISION_SABOTAGE"]()


def _kpss_flags_host(
    y_address: Int,
    batch_size: Int,
    n_obs: Int,
    d: Int,
    D: Int,
    s: Int,
    pval: Float32,
    fp: MutPointer[Int32, MutUntrackedOrigin],
    sp: MutPointer[Float32, MutUntrackedOrigin],
) raises:
    """`kpss_test`'s guards and `kpss_host_f32`, writing `batch_size` flags
    (1 stationary) at `fp` and statistics at `sp`. Shared by `kpss_test`
    and the host `select_d` (lane py-runtime-b), so both raise the same
    words in the same order."""
    # `kpss_test_host`'s `_refuse_empty_shape`.
    if batch_size < 1:
        raise Error(
            "kpss_test: batch_size must be >= 1 (batch_size=" + String(batch_size) + ")"
        )
    if n_obs < 1:
        raise Error("kpss_test: n_obs must be >= 1 (n_obs=" + String(n_obs) + ")")
    # `kpss_test`'s, in its order.
    var d_sD = d + s * D
    if n_obs <= d_sD:
        raise Error(
            "stationarity: n_obs (" + String(n_obs)
            + ") must be greater than d + s*D (" + String(d_sD) + ")"
        )
    var y = read_f32(y_address, batch_size * n_obs)
    for i in range(batch_size * n_obs):
        if not isfinite(y[i]):
            raise Error(
                "kpss_test: y contains a non-finite value at index "
                + String(i) + "; missing or infinite observations are refused by name"
            )
    # `prepare_data`'s, reached only when there is differencing to do.
    if d != 0 or D != 0:
        if d + D > 2:
            raise Error(
                "prepare_data: d + D must be <= 2 (d=" + String(d) + ", D="
                + String(D) + "), refused by name (arima.pyx:313)"
            )
        if D > 0 and s < 2:
            raise Error(
                "prepare_data: seasonal differencing needs s >= 2 (s=" + String(s)
                + "), refused by name (arima.pyx:310)"
            )
    var st = kpss_host_f32(y, batch_size, n_obs, d, D, s, pval)
    for b in range(batch_size):
        fp[b] = Int32(1) if st.stationary[b] else Int32(0)
        comptime if KPSS_DECISION_SABOTAGE:
            fp[b] = Int32(1) - fp[b]
        sp[b] = st.stat[b]


def kpss_select_d_host(
    y_address: Int, batch_size: Int, n_obs: Int, D: Int, s: Int, d_max: Int, pval: Float32,
    dp: MutPointer[Int32, MutUntrackedOrigin],
) raises:
    var flags = List[Int32](length=max(batch_size, 1), fill=Int32(0))
    var stat = List[Float32](length=max(batch_size, 1), fill=Float32(0))
    var decided = List[Bool](length=max(batch_size, 1), fill=False)
    for b in range(batch_size):
        dp[b] = Int32(d_max)
    var left = batch_size
    for order in range(d_max):  # small-loop(d_max: differencing orders): at most two KPSS orders, each a whole-batch test
        if left == 0:
            break
        _kpss_flags_host(y_address, batch_size, n_obs, order, D, s, pval,
                         i32_ptr(Int(flags.unsafe_ptr())), f32_ptr(Int(stat.unsafe_ptr())))
        for b in range(batch_size):
            if not decided[b] and flags[b] != 0:
                dp[b] = Int32(order)
                decided[b] = True
                left -= 1
