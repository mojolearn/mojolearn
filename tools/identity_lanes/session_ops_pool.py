# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SUB-LANE session-ops-pool's IDENTITY LANES (2026-11-25): Sequential's pools as ops.Session operations
# (autograd/session_ops_pool.mojo, session codes 140..148), forward and backward, in a captured Program and in a
# scan cell.
#
# Two lanes, one per implementation: ops-pool-window (the rank-N window pool node: max / avg pool 1-D, 2-D with
# PyTorch's options, 3-D, and adaptive_avg_pool3d's equal tiles) and ops-pool-adaptive (x_cnn's adaptive window
# bodies: adaptive avg / max 1-D and 2-D). Each op is its own part, so a fault names the op it moved.
#
# Verify through `device=` (docs/plans/VERIFY_THROUGH_DEVICE.md): the lanes are device-agnostic, they open their
# sessions with no `device=` and the harness's view gives every Session the side, so the same lane runs the host
# column (device="cpu") and the device column (device="gpu") and the two digests are compared with the reference.
#
# Loaded by tools/identity_break.py `_load_lane_fragments` in the harness namespace: np, _h, _fit, _hw,
# _same_bytes, _batch_decl and `lane` are the harness's; helpers here are prefixed `_sop_pool_`. Inputs come from
# the hashed stream (`_hw`), nothing shared with another lane. Reference OWED (recorded once, at the release's one
# verification). Session and Program have no digest_: the parts hash the selected outputs and gradients directly.


def _sop_pool_first(r):
    """A Program's one output (run returns the outputs; a single output may come back bare)."""
    if isinstance(r, (list, tuple)):
        return np.asarray(r[0])
    return np.asarray(r)


def _sop_pool_mixed(shape, key):
    """Values whose magnitudes mix 1e2 and 1e-2 along the last axis, so the order of a window's additions shows
    in the last bits of an average."""
    v = _hw(shape, key, -1.0, 1.0)
    scale = np.where(np.arange(shape[-1]) % 2 == 0, np.float32(1.0e2), np.float32(1.0e-2)).astype(np.float32)
    return np.ascontiguousarray((v * scale).astype(np.float32))


def _sop_pool_loss(s, ys, key):
    """sum_k sum(y_k * c_k) with fixed weights c_k; the per-op results as numpy arrays."""
    loss = None
    values = []
    for k, y in enumerate(ys):  # driver-loop: over the lane's op results
        values.append(np.asarray(s.numpy(y)))
        c = s.tensor(_hw(values[-1].shape, f"{key}:c{k}", -1.0, 1.0))
        term = s.sum(s.mul(y, c))
        loss = term if loss is None else s.add(loss, term)
    return loss, values


_SOP_POOL_WINDOW_NAMES = ("max_pool1d", "avg_pool1d", "max_pool2d", "avg_pool2d", "avg_pool2d_divisor",
                          "max_pool3d", "avg_pool3d", "adaptive_avg_pool3d")


@lane("ops-pool-window")
def _(ml, X, yc, yr, Xh=None):
    """ops.Session's window pools (sub-lane session-ops-pool, 2026-11-25), the rank-N pool node of Sequential's
    max / avg pool layers, through the option words the Session adds: max_pool1d with dilation 2 and ceil_mode on
    x1 (2, 2, 7); avg_pool1d with ceil_mode and count_include_pad=False; max_pool2d on x2 (1, 2, 5, 6) with a
    (3, 2) window, stride (2, 1), padding 1, dilation (1, 2) and ceil_mode; avg_pool2d with padding, ceil_mode and
    count_include_pad=False, and another with a (2, 3) window and divisor_override 4; max_pool3d on x3
    (1, 2, 3, 4, 4) with stride (1, 2, 2), padding (1, 1, 0), dilation (1, 1, 2) and ceil_mode; avg_pool3d with
    padding and count_include_pad=False; adaptive_avg_pool3d of x4 (1, 2, 2, 4, 4) to (None, 2, 2). The loss is
    the sum of each result times a fixed weight; backward. Parts: each op's forward, the loss, each input's
    gradient. Then max_pool1d and avg_pool2d captured as a Program, run on the captured arrays (the same bytes as
    the session's results, checked) and on new arrays. Sabotage arms coverage-autograd-session_ops_pool_incpad_cpu
    (the host divides by the padded window when count_include_pad is False: avg_pool1d, avg_pool2d, avg_pool3d and
    the program move on the CPU side) and coverage-autograd-session_ops_pool_dilation_gpu (the device max pool
    reads every dilation as 1: max_pool1d, max_pool2d, max_pool3d and the program move on the GPU side)."""
    x1 = _sop_pool_mixed((2, 2, 7), "sop-pool:x1")
    x2 = _sop_pool_mixed((1, 2, 5, 6), "sop-pool:x2")
    x3 = _sop_pool_mixed((1, 2, 3, 4, 4), "sop-pool:x3")
    x4 = _sop_pool_mixed((1, 2, 2, 4, 4), "sop-pool:x4")
    parts = {}
    with ml.ops.Session(record=True, numeric_mode="identical") as s:
        h1, h2, h3, h4 = (s.tensor(a, requires_grad=True) for a in (x1, x2, x3, x4))  # glue: four inputs
        ys = [
            s.max_pool1d(h1, 2, stride=2, padding=1, dilation=2, ceil_mode=True),
            s.avg_pool1d(h1, 3, stride=2, padding=1, ceil_mode=True, count_include_pad=False),
            s.max_pool2d(h2, (3, 2), stride=(2, 1), padding=1, dilation=(1, 2), ceil_mode=True),
            s.avg_pool2d(h2, 3, stride=2, padding=1, ceil_mode=True, count_include_pad=False),
            s.avg_pool2d(h2, (2, 3), stride=(1, 2), padding=1, divisor_override=4),
            s.max_pool3d(h3, 2, stride=(1, 2, 2), padding=(1, 1, 0), dilation=(1, 1, 2), ceil_mode=True),
            s.avg_pool3d(h3, 2, stride=2, padding=1, count_include_pad=False),
            s.adaptive_avg_pool3d(h4, (None, 2, 2)),
        ]
        loss, values = _sop_pool_loss(s, ys, "sop-pool:w")
        for name, v in zip(_SOP_POOL_WINDOW_NAMES, values):  # driver-loop: one part per op
            parts["forward-" + name] = _h(v)
        parts["loss"] = _h(np.asarray(s.numpy(loss)))
        s.backward(loss)
        for name, h in zip(("x1", "x2", "x3", "x4"), (h1, h2, h3, h4)):  # driver-loop: one part per input
            parts["grad-" + name] = _h(np.asarray(s.grad(h)))
    with ml.ops.Session(numeric_mode="identical") as s:
        a = s.tensor(x1)
        b = s.tensor(x2)
        y1 = s.max_pool1d(a, 2, stride=2, padding=1, dilation=2, ceil_mode=True)
        y2 = s.avg_pool2d(b, 3, stride=2, padding=1, ceil_mode=True, count_include_pad=False)
        eager = [np.asarray(s.numpy(y1)), np.asarray(s.numpy(y2))]
        prog = s.capture(inputs=[a, b], outputs=[y1, y2])
        again = prog.run(x1, x2)
        _same_bytes("Program max_pool1d", np.asarray(again[0]), "Session.max_pool1d", eager[0])
        _same_bytes("Program avg_pool2d", np.asarray(again[1]), "Session.avg_pool2d", eager[1])
        other = prog.run(np.ascontiguousarray(x1[:, ::-1] * np.float32(-0.5)),
                         np.ascontiguousarray(x2[..., ::-1] * np.float32(2.0)))
        parts["program"] = _h(*eager, *(np.asarray(o) for o in other))
    return _fit(parts)


_SOP_POOL_ADAPTIVE_NAMES = ("adaptive_avg_pool1d", "adaptive_avg_pool1d_global", "adaptive_max_pool1d",
                            "adaptive_avg_pool2d", "adaptive_avg_pool2d_keep", "adaptive_max_pool2d",
                            "adaptive_max_pool2d_global")


@lane("ops-pool-adaptive")
def _(ml, X, yc, yr, Xh=None):
    """ops.Session's adaptive pools (sub-lane session-ops-pool, 2026-11-25), x_cnn's adaptive window bodies that
    Sequential's adaptive layers run: adaptive_avg_pool1d of x1 (2, 2, 7) to 3 (uneven, overlapping windows) and to
    1 (the global mean); adaptive_max_pool1d to 3; adaptive_avg_pool2d of x2 (1, 2, 5, 7) to (3, 4) and to
    (None, 2); adaptive_max_pool2d to (2, 3) and to 1 (the global max pool). The loss is the sum of each result
    times a fixed weight; backward. Parts: each op's forward, the loss, each input's gradient. Then a scan whose
    cell pools each step (state + adaptive_avg_pool1d(x_t, 3)) over xs (4, 2, 2, 7), forward and backward through
    time (the part "scan": outputs, final state, the gradient of xs). Sabotage arms
    coverage-autograd-session_ops_pool_adapt_avg_cpu (the host's adaptive average scaled by 7/8: every average part
    and the scan move on the CPU side) and coverage-autograd-session_ops_pool_adapt_max_vjp_gpu (the device's
    adaptive max gradient halved: the gradient parts move on the GPU side)."""
    x1 = _sop_pool_mixed((2, 2, 7), "sop-adapt:x1")
    x2 = _sop_pool_mixed((1, 2, 5, 7), "sop-adapt:x2")
    xs0 = _sop_pool_mixed((4, 2, 2, 7), "sop-adapt:xs")
    parts = {}
    with ml.ops.Session(record=True, numeric_mode="identical") as s:
        h1 = s.tensor(x1, requires_grad=True)
        h2 = s.tensor(x2, requires_grad=True)
        ys = [
            s.adaptive_avg_pool1d(h1, 3),
            s.adaptive_avg_pool1d(h1, 1),
            s.adaptive_max_pool1d(h1, 3),
            s.adaptive_avg_pool2d(h2, (3, 4)),
            s.adaptive_avg_pool2d(h2, (None, 2)),
            s.adaptive_max_pool2d(h2, (2, 3)),
            s.adaptive_max_pool2d(h2, 1),
        ]
        loss, values = _sop_pool_loss(s, ys, "sop-adapt:w")
        for name, v in zip(_SOP_POOL_ADAPTIVE_NAMES, values):  # driver-loop: one part per op
            parts["forward-" + name] = _h(v)
        parts["loss"] = _h(np.asarray(s.numpy(loss)))
        s.backward(loss)
        parts["grad-x1"] = _h(np.asarray(s.grad(h1)))
        parts["grad-x2"] = _h(np.asarray(s.grad(h2)))
    with ml.ops.Session(numeric_mode="identical") as c:
        state = c.tensor(np.zeros((2, 2, 3), dtype=np.float32))
        x_t = c.tensor(xs0[0])
        pooled = c.adaptive_avg_pool1d(x_t, 3)
        cell = c.capture(inputs=[state, x_t], outputs=[c.add(state, pooled), pooled])
        with ml.ops.Session(record=True, numeric_mode="identical") as s:
            xs = s.tensor(xs0, requires_grad=True)
            init = s.tensor(np.zeros((2, 2, 3), dtype=np.float32))
            out, final = s.scan(cell, xs, init)
            w = s.tensor(_hw((4, 2, 2, 3), "sop-adapt:ws", -1.0, 1.0))
            scan_loss = s.add(s.sum(s.mul(out, w)), s.sum(final))
            s.backward(scan_loss)
            parts["scan"] = _h(np.asarray(s.numpy(out)), np.asarray(s.numpy(final)), np.asarray(s.grad(xs)))
    return _fit(parts)


_batch_decl("n/a:training-step (one seeded forward and backward through the ops session, a captured program and a "
            "scan cell; the forward outputs are parts, digested whole)", "ops-pool-window", "ops-pool-adaptive")
