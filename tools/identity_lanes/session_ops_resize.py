# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# LANE session-ops-resize's IDENTITY LANE (2026-11-25): ops.Session.interpolate, pixel_shuffle, pixel_unshuffle and
# pad(mode='reflect' | 'replicate') (autograd/session_ops_resize.mojo, Sequential's own forward functions of
# autograd/nn_conv.mojo), forward and backward, and a captured Program of them.
#
# Verify through `device=` (docs/plans/VERIFY_THROUGH_DEVICE.md): the lane is device-agnostic, it opens its
# sessions with no `device=` and the harness's view gives every Session the side, so the same lane runs the host
# column (device="cpu") and the device column (device="gpu") and the two digests are compared with the reference.
#
# Loaded by tools/identity_break.py `_load_lane_fragments` in the harness namespace: np, _h, _fit, _hw,
# _same_bytes, _batch_decl and `lane` are the harness's; helpers here are prefixed `_session_ops_resize_`. The
# inputs are built from the hashed stream (`_hw`), nothing shared with another lane. Reference OWED (recorded
# once, at the release's one verification). The four operations share one family (nn_conv.mojo's bodies and
# NnConvRules), so they are one lane; each operation is its own part, so a fault names the operation it moves.


def _session_ops_resize_first(r):
    """A Program's one output (run returns the outputs; a single output may come back bare)."""
    if isinstance(r, (list, tuple)):
        return np.asarray(r[0])
    return np.asarray(r)


@lane("ops-resize")
def _(ml, X, yc, yr, Xh=None):
    """ops.Session's resampling and rearranging (lane session-ops-resize, 2026-11-25), the smallest shapes that
    reach each edge, in one recording session. x (1, 2, 3, 4) and a 1-D signal z (1, 2, 5). Parts, one per
    operation, each its forward words and its input's gradient of sum(output * fixed weight):
      interp_bilinear  interpolate x by 1.5 'bilinear' ((1, 2, 4, 6): unequal height weights) and to size (5, 7)
                       with align_corners;
      interp_nearest   interpolate x to size (2, 6) 'nearest' (a downsampled height, a doubled width) and z by 2
                       'linear' (Sequential's (N, C, 1, L) view);
      shuffle          pixel_unshuffle of a (1, 2, 4, 2) tensor by 2, then pixel_shuffle by 2 of the result
                       times a weight (both directions, not an identity);
      pad_reflect      pad x reflect (left 3 = W - 1, the largest count; top 1, bottom 2 = H - 1) and z (2, 1);
      pad_replicate    pad x replicate (left 5 > W, bottom 1) and z (0, 3).
    Then interpolate (bilinear x1.5) and pad (reflect) captured as one Program in an inference session and run on
    the captured array (the same bytes as the session's own result, checked) and on a new array: part program.
    Sabotage arms coverage-autograd-session_ops_resize_up_gpu (the bilinear height weights swapped, device only:
    interp_bilinear and program move on the device side) and coverage-autograd-session_ops_resize_pad_cpu
    (reflection's left column +0.0, host only: pad_reflect and program move on the host side)."""
    x0 = _hw((1, 2, 3, 4), "ops-rsz:x", -1.0, 1.0)
    z0 = _hw((1, 2, 5), "ops-rsz:z", -1.0, 1.0)
    u0 = _hw((1, 2, 4, 2), "ops-rsz:u", -1.0, 1.0)
    parts = {}

    def part(name, build):
        # one recording session per part: its own graph and one backward
        with ml.ops.Session(record=True, numeric_mode="identical") as s:
            inputs, outputs = build(s)
            loss = None
            for k, y in enumerate(outputs):  # driver-loop: over a part's outputs
                w = _hw(tuple(np.asarray(s.numpy(y)).shape), f"ops-rsz:w{name}{k}", -1.0, 1.0)
                term = s.sum(s.mul(y, s.tensor(w)))
                loss = term if loss is None else s.add(loss, term)
            values = [np.asarray(s.numpy(y)) for y in outputs]  # glue: one array per output
            s.backward(loss)
            parts[name] = _h(*values, *[np.asarray(s.grad(t)) for t in inputs])  # glue: one array per input

    def interp_bilinear(s):
        x = s.tensor(x0, requires_grad=True)
        return [x], [s.interpolate(x, scale_factor=1.5, mode="bilinear"),
                     s.interpolate(x, size=(5, 7), mode="bilinear", align_corners=True)]

    def interp_nearest(s):
        x = s.tensor(x0, requires_grad=True)
        z = s.tensor(z0, requires_grad=True)
        return [x, z], [s.interpolate(x, size=(2, 6)), s.interpolate(z, scale_factor=2, mode="linear")]

    def shuffle(s):
        u = s.tensor(u0, requires_grad=True)
        down = s.pixel_unshuffle(u, 2)
        mid = s.mul(down, s.tensor(_hw((1, 8, 2, 1), "ops-rsz:m", -1.0, 1.0)))
        return [u], [down, s.pixel_shuffle(mid, 2)]

    def pad_reflect(s):
        x = s.tensor(x0, requires_grad=True)
        z = s.tensor(z0, requires_grad=True)
        return [x, z], [s.pad(x, [0, 0, 1, 3, 0, 0, 2, 0], mode="reflect"),
                        s.pad(z, [0, 0, 2, 0, 0, 1], mode="reflect")]

    def pad_replicate(s):
        x = s.tensor(x0, requires_grad=True)
        z = s.tensor(z0, requires_grad=True)
        return [x, z], [s.pad(x, [0, 0, 0, 5, 0, 0, 1, 0], mode="replicate"),
                        s.pad(z, [0, 0, 0, 0, 0, 3], mode="replicate")]

    for name, build in (("interp_bilinear", interp_bilinear), ("interp_nearest", interp_nearest),
                        ("shuffle", shuffle), ("pad_reflect", pad_reflect),
                        ("pad_replicate", pad_replicate)):  # driver-loop: over the five operation parts
        part(name, build)
    with ml.ops.Session(numeric_mode="identical") as s:
        xi = s.tensor(x0)
        y = s.pad(s.interpolate(xi, scale_factor=1.5, mode="bilinear"), [0, 0, 1, 2, 0, 0, 0, 1], mode="reflect")
        eager = np.asarray(s.numpy(y))
        prog = s.capture(inputs=[xi], outputs=[y])
        again = _session_ops_resize_first(prog.run(x0))
        _same_bytes("Program interpolate + pad", again, "Session interpolate + pad", eager)
        x2 = np.ascontiguousarray((x0[..., ::-1] * np.float32(-0.5)).astype(np.float32))
        parts["program"] = _h(eager, _session_ops_resize_first(prog.run(x2)))
    return _fit(parts)


_batch_decl("n/a:training-step (one seeded forward and backward per operation through the ops session and a "
            "captured program; the forward outputs are parts, digested whole)", "ops-resize")
