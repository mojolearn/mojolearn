# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# LANE session-ops-conv's IDENTITY LANE (2026-11-25): ops.Session.conv1d, conv2d with dilation and per-axis words,
# conv3d, conv_transpose1d and conv_transpose2d (autograd/session_ops_conv.mojo), forward and backward, and a
# captured Program of two of them.
#
# Verify through `device=` (docs/plans/VERIFY_THROUGH_DEVICE.md): the lane opens its sessions with no `device=`
# and the harness's view gives every Session the side, so the same lane runs the host column (device="cpu") and
# the device column (device="gpu") and the two digests are compared with the reference.
#
# Loaded by tools/identity_break.py `_load_lane_fragments` in the harness namespace: np, _h, _fit, _hw,
# _same_bytes, _batch_decl and `lane` are the harness's; helpers here are prefixed `_session_ops_conv_`. Inputs
# come from the hashed stream (`_hw`), nothing shared with another lane. Reference OWED (recorded once, at the
# release's one verification). The shapes are the smallest that reach each edge: a stride that skips input,
# padding wider than a tap, dilation on one axis only, groups, output_padding, a bias.


def _session_ops_conv_first(r):
    """A Program's one output (run returns the outputs; a single output may come back bare)."""
    if isinstance(r, (list, tuple)):
        return np.asarray(r[0])
    return np.asarray(r)


@lane("ops-conv-layers")
def _(ml, X, yc, yr, Xh=None):
    """ops.Session's convolutions (lane session-ops-conv, 2026-11-25), each the node of Sequential's layer. In one
    recording session: conv1d (2, 4, 7) by (4, 2, 3), stride 2, padding 2, dilation 2, groups 2, bias; conv2d
    (1, 2, 5, 6) by (3, 2, 2, 3) with stride (1, 2), padding (2, 0), dilation (2, 1), bias; conv3d (1, 2, 3, 4, 4)
    by (2, 1, 2, 2, 2) with stride (1, 2, 1), padding (1, 0, 1), dilation (1, 1, 2), groups 2, bias;
    conv_transpose1d (1, 2, 4) by (2, 3, 3), stride 2, padding 1, output_padding 1, bias; conv_transpose2d
    (1, 2, 3, 3) by (2, 2, 3, 2), stride (2, 1), padding (1, 0), output_padding (1, 0), dilation (1, 2), groups 2,
    bias. The loss is the sum of each output times a fixed weight; backward. Parts, apart where a fault differs:
    `conv` (the three convolutions' outputs and the gradients of their inputs, weights and biases), `transpose`
    (the two transposed convolutions' outputs and gradients), `loss`. Then conv1d and conv_transpose2d captured
    as a Program in an inference session: run on the captured arrays (the same bytes as the session's own results,
    checked) and on new arrays: part `program`. Sabotage arms: coverage-autograd-session_ops_conv_dilation_cpu
    (the host im2col ignores the width dilation: `conv`, `transpose` gradients and `loss` move on the CPU side),
    coverage-autograd-session_ops_conv_ct_bias_gpu (the device transposed output adds no bias: `transpose`,
    `loss` and `program` move on the GPU side)."""
    conv_specs = [  # glue: (method, x shape, weight shape, bias length, keyword arguments)
        ("conv1d", (2, 4, 7), (4, 2, 3), 4, dict(stride=2, padding=2, dilation=2, groups=2)),
        ("conv2d", (1, 2, 5, 6), (3, 2, 2, 3), 3, dict(stride=(1, 2), padding=(2, 0), dilation=(2, 1))),
        ("conv3d", (1, 2, 3, 4, 4), (2, 1, 2, 2, 2), 2,
         dict(stride=(1, 2, 1), padding=(1, 0, 1), dilation=(1, 1, 2), groups=2)),
    ]
    transpose_specs = [
        ("conv_transpose1d", (1, 2, 4), (2, 3, 3), 3, dict(stride=2, padding=1, output_padding=1)),
        ("conv_transpose2d", (1, 2, 3, 3), (2, 2, 3, 2), 4,
         dict(stride=(2, 1), padding=(1, 0), output_padding=(1, 0), dilation=(1, 2), groups=2)),
    ]
    arrays = {}
    for name, xs, ws, nb, _kw in conv_specs + transpose_specs:  # driver-loop: over the five operations' inputs
        arrays[name] = (np.ascontiguousarray(_hw(xs, f"ops-conv:{name}:x", -1.0, 1.0)),
                        np.ascontiguousarray(_hw(ws, f"ops-conv:{name}:w", -0.5, 0.5)),
                        np.ascontiguousarray(_hw((nb,), f"ops-conv:{name}:b", -0.25, 0.25)))
    parts = {}
    with ml.ops.Session(record=True, numeric_mode="identical") as s:
        groups = {}
        for label, specs in (("conv", conv_specs), ("transpose", transpose_specs)):  # driver-loop: two groups
            outs, leaves, terms = [], [], []
            for name, xs, ws, nb, kw in specs:  # driver-loop: over the group's operations
                x0, w0, b0 = arrays[name]
                x = s.tensor(x0, requires_grad=True)
                w = s.tensor(w0, requires_grad=True)
                b = s.tensor(b0, requires_grad=True)
                y = getattr(s, name)(x, w, b, **kw)
                c = _hw(s.shape(y), f"ops-conv:{name}:c", -1.0, 1.0)
                terms.append(s.sum(s.mul(y, s.tensor(np.ascontiguousarray(c)))))
                outs.append(y)
                leaves += [x, w, b]
            groups[label] = (outs, leaves, terms)
        all_terms = groups["conv"][2] + groups["transpose"][2]
        loss = all_terms[0]
        for t in all_terms[1:]:  # driver-loop: over the five loss terms
            loss = s.add(loss, t)
        parts["loss"] = _h(np.asarray(s.numpy(loss)))
        s.backward(loss)
        for label in ("conv", "transpose"):  # driver-loop: the two parts
            outs, leaves, _terms = groups[label]
            parts[label] = _h(*[np.asarray(s.numpy(y)) for y in outs],  # glue: one array per output
                              *[np.asarray(s.grad(t)) for t in leaves])  # glue: one array per input
    with ml.ops.Session(numeric_mode="identical") as s:
        x1, w1, b1 = arrays["conv1d"]
        x2, w2, b2 = arrays["conv_transpose2d"]
        hx1, hw1, hb1 = s.tensor(x1), s.tensor(w1), s.tensor(b1)
        hx2, hw2, hb2 = s.tensor(x2), s.tensor(w2), s.tensor(b2)
        y1 = s.conv1d(hx1, hw1, hb1, **conv_specs[0][4])
        y2 = s.conv_transpose2d(hx2, hw2, hb2, **transpose_specs[1][4])
        eager = [np.asarray(s.numpy(y1)), np.asarray(s.numpy(y2))]
        prog1 = s.capture(inputs=[hx1, hw1, hb1], outputs=[y1])
        prog2 = s.capture(inputs=[hx2, hw2, hb2], outputs=[y2])
        again = [_session_ops_conv_first(prog1.run(x1, w1, b1)), _session_ops_conv_first(prog2.run(x2, w2, b2))]
        _same_bytes("Program conv1d", again[0], "Session.conv1d", eager[0])
        _same_bytes("Program conv_transpose2d", again[1], "Session.conv_transpose2d", eager[1])
        other = [_session_ops_conv_first(prog1.run(np.ascontiguousarray(x1[::-1] * np.float32(-0.5)), w1, b1)),
                 _session_ops_conv_first(prog2.run(np.ascontiguousarray(x2 * np.float32(2.0)), w2, -b2))]
        parts["program"] = _h(*eager, *other)
    return _fit(parts)


_batch_decl("n/a:training-step (one seeded forward and backward through the ops session and a captured program; "
            "the forward outputs are parts, digested whole)", "ops-conv-layers")
