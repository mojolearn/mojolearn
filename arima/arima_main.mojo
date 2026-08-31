# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The batched ARIMA Kalman filter on one hashed batch, with an identity card.

    tools/with_build_lock.sh     pixi run mojo run -I . arima/arima_main.mojo
    tools/with_identical_mode.sh pixi run mojo run -I . arima/arima_main.mojo
    MOJOLEARN_IDENTITY_TRACE=/tmp/arima.card tools/with_identical_mode.sh \
        pixi run mojo run -I . arima/arima_main.mojo

    python3 tools/identity_trace_diff.py /tmp/arima.mac.card /tmp/arima.other.card

The card (`core/identity_trace.mojo`) carries the INPUT bytes first, then the
untransformed parameters, then the Jones-TRANSFORMED parameters (read those
second when two cards disagree: a different `t_params` is a different model
and everything downstream of it is noise), then every stage of the filter in
the order the device writes them --  `Z`, `R`, `T`, `RQ`, `RQR`, `P0`,
`alpha0`, `pred`, `vs`, `Fs`, `loglike`, `fc`, `P_final` -- and last the DECISION stages
`piv`, `info_init`, `info_loop` and `guards`, for each of the ten orders in
`arima/original/fixtures.mojo`'s table.

THE DECISION STAGES ARE NOT DECORATION. Every float stage records a VALUE;
these four record a CHOICE, and a choice can differ between vendors while
every value stays bit-identical. `piv` is the LU permutation and therefore
DEVIATION 674's tie rule; `guards` says which of the two rare rewrites fired
per series; `info_init` and `info_loop` are all zero on a healthy run and
prove the refusal paths were NOT taken. The holtwinters lane established the
shape of this argument: its `CRIT_ORDER` sabotage moves zero of 2800 float
cells and is caught only because the criterion is a recorded stage. Tags are unique and carry no launch
parameter, so a card is a function of the fixture and the mode alone.

READ THE STAGES IN ORDER. `T` before `P0` before `alpha0` before `pred`: the
first stage that differs is the one to diagnose, and every later one inherits
it. `P0` is the stage most likely to move between vendors, because it is
DEVIATION 674's hand-written LU where theirs is a closed cuBLAS batched
`getrf`/`getri`.

Not a port: cuML ships one backend and needs no card.

STATUS: RUN ON ONE APPLE M4, BOTH MODES. This file emits a 229-stage card
(138 when it first ran on 2026-08-23; the 2026-08-24 round added the four
DECISION stages, `Fs` and `P_final`). See `arima/README.md`'s status block,
`arima/DERIVATION_MAP.tsv` and IDENTITY_PATHS row 58.

NO SECOND VENDOR. `tools/e1_bootstrap.sh` phase 8 carries gemm, cd, kde,
linkage, svm, mamba and metrics, and not this lane, so no leg has ever
produced an arima card on an NVIDIA or AMD box and the cross-vendor claim
this card exists to make is UNTESTED. See `arima/README.md`'s OWED list.
"""

from max.gpu.host import DeviceContext

from core.identity_trace import IdentityTrace
from original.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL, numeric_mode_name

from arima.original.fixtures import (
    arima_fixture,
    arima_params_fixture,
    bits32,
    download_f32,
    order_table,
    upload_f32,
    upload_params,
)
from arima.derived.arima.batched_arima import batched_loglike
from arima.derived.linalg.batched.matrix import LYAP_R2_MAX


comptime IDENTICAL = GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
comptime N_OBS = 24
comptime SALT = 7
comptime FC_STEPS = 3


def _mode_name() -> String:
    """The build's tier, from the ONE definition of it.

    Delegates to `numeric_mode_name()` since 2026-08-29; see the note
    on that function. A local two-way IDENTICAL-or-FAST answers "FAST"
    for a DETERMINISTIC build, which mislabels every line the driver
    prints.
    """
    return numeric_mode_name()


def main() raises:
    print("== arima/arima_main.mojo [" + _mode_name() + "] ==")
    var ctx = DeviceContext()
    var trace = IdentityTrace()
    trace.header(
        "arima_main mode=" + _mode_name() + " n_obs=" + String(N_OBS)
        + " batch=6 salt=" + String(SALT) + " fc_steps=" + String(FC_STEPS)
    )

    var f = arima_fixture(N_OBS, SALT)
    var y = upload_f32(ctx, f.y)
    trace.record_device[DType.float32](ctx, "arima.input", y, f.batch_size * f.n_obs)

    var table = order_table()
    for oc in table:
        var order = oc.order
        var tag = "arima." + oc.name
        var rd = order.rd()
        var rd2 = rd * rd
        var b = f.batch_size

        var ph = arima_params_fixture(order, b, SALT, oc.plant)
        var params = upload_params(ctx, ph, order, b)
        if order.k != 0:
            trace.record_device[DType.float32](ctx, tag + ".param.mu", params.mu, b)
        if order.p != 0:
            trace.record_device[DType.float32](ctx, tag + ".param.ar", params.ar, order.p * b)
        if order.q != 0:
            trace.record_device[DType.float32](ctx, tag + ".param.ma", params.ma, order.q * b)
        if order.P != 0:
            trace.record_device[DType.float32](ctx, tag + ".param.sar", params.sar, order.P * b)
        if order.Q != 0:
            trace.record_device[DType.float32](ctx, tag + ".param.sma", params.sma, order.Q * b)
        trace.record_device[DType.float32](ctx, tag + ".param.sigma2", params.sigma2, b)

        var res = batched_loglike(ctx, y, b, f.n_obs, order, params, True, FC_STEPS)

        # the TRANSFORMED parameters: read these second when cards disagree
        if order.p != 0:
            trace.record_device[DType.float32](ctx, tag + ".jones.ar", res.t_params.ar, order.p * b)
        if order.q != 0:
            trace.record_device[DType.float32](ctx, tag + ".jones.ma", res.t_params.ma, order.q * b)
        if order.P != 0:
            trace.record_device[DType.float32](ctx, tag + ".jones.sar", res.t_params.sar, order.P * b)
        if order.Q != 0:
            trace.record_device[DType.float32](ctx, tag + ".jones.sma", res.t_params.sma, order.Q * b)
        trace.record_device[DType.float32](ctx, tag + ".jones.sigma2", res.t_params.sigma2, b)

        # the filter, in the order the device writes them
        trace.record_device[DType.float32](ctx, tag + ".Z", res.ws.Z, rd * b)
        trace.record_device[DType.float32](ctx, tag + ".R", res.ws.R, rd * b)
        trace.record_device[DType.float32](ctx, tag + ".T", res.ws.T, rd2 * b)
        trace.record_device[DType.float32](ctx, tag + ".RQ", res.ws.RQ, rd * b)
        trace.record_device[DType.float32](ctx, tag + ".RQR", res.ws.RQR, rd2 * b)
        trace.record_device[DType.float32](ctx, tag + ".P0", res.ws.P0, rd2 * b)
        trace.record_device[DType.float32](ctx, tag + ".alpha0", res.ws.alpha0, rd * b)
        trace.record_device[DType.float32](ctx, tag + ".pred", res.ws.pred, f.n_obs * b)
        trace.record_device[DType.float32](ctx, tag + ".vs", res.ws.vs, f.n_obs * b)
        trace.record_device[DType.float32](ctx, tag + ".Fs", res.ws.Fs, f.n_obs * b)
        trace.record_device[DType.float32](ctx, tag + ".loglike", res.ws.loglike, b)
        trace.record_device[DType.float32](ctx, tag + ".fc", res.ws.fc, FC_STEPS * b)
        # `P0` is the covariance BEFORE the filter; `P_final` is after it. Only
        # the first was ever recorded, so every vendor difference the filter
        # introduced into the covariance and then carried forward was invisible.
        trace.record_device[DType.float32](ctx, tag + ".P_final", res.ws.P, rd2 * b)

        # ------------------------------------------------------------------
        # DECISION STAGES (added 2026-08-24, Andrew's "are there more hashes
        # we can take"). Every stage above is a float BUFFER. These three are
        # the CHOICES the pipeline made, and until now they were recorded
        # nowhere, on any card, in this lane.
        #
        # `piv` is the important one. It is the LU permutation: which row was
        # selected as pivot in each column of `I - T (x) T` and of `I - T*`.
        # That sequence IS DEVIATION 674's decision -- the tie rule this lane
        # CHOSE, because cuBLAS's is not readable -- and it is not derivable
        # from any float stage, because `P0` is the product of the solve and
        # not of the permutation that produced it. Sabotage (e) perturbs
        # exactly this: flipping `>` to `>=` on a tied column changes the
        # permutation, and it can perfectly well leave `P0` bit-identical by
        # luck, in which case `piv` is the ONLY stage that moves. That is the
        # holtwinters CRIT_ORDER case in this lane: a decision-only sabotage
        # that no float comparison can see.
        #
        # `info_init` and `info_loop` are the refusal codes. On a healthy run
        # they are all zero, and recording a buffer of zeros sounds useless
        # until you ask what it proves: that the REFUSAL PATH WAS NOT TAKEN.
        # A vendor whose `P0` solve went singular where ours did not, or
        # whose `F` went non-positive at a step where ours did not, differs
        # in `info` before it differs anywhere a human would look.
        trace.record_device[DType.int32](ctx, tag + ".piv", res.ws.piv, LYAP_R2_MAX * b)
        trace.record_device[DType.int32](ctx, tag + ".info_init", res.ws.info_init, b)
        trace.record_device[DType.int32](ctx, tag + ".info_loop", res.ws.info_loop, b)
        trace.record_device[DType.uint8](ctx, tag + ".guards", res.ws.guards, b)

        var ll = download_f32(ctx, res.ws.loglike, b)
        print("  " + oc.name + " rd=" + String(rd) + " r=" + String(order.r())
              + ": loglike[0] = " + String(ll[0]) + " " + bits32(ll[0]))
        _ = res^
        _ = params^

    print("arima_main done [" + _mode_name() + "]"
          + (" card: " + trace.path if trace.enabled else " (no card: set MOJOLEARN_IDENTITY_TRACE)"))
    _ = y^
