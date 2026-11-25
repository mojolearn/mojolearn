# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SUB-LANE nn-blocks-cbw's IDENTITY LANE (2026-11-25): a user-defined backward rule (Session.custom_op /
# Session.custom, autograd/nn_cbw.mojo), Andrew's four deciding tests on each side.
#
# Device-agnostic (docs/plans/VERIFY_THROUGH_DEVICE.md contract 1): no `device=`; the harness's view gives
# `ops.Session` the side, so the same lane drives the host column and the device column. Loaded by
# tools/identity_break.py `_load_lane_fragments`; np, _h, _fit, _same_bytes and `lane` are the harness's. Helpers
# here are prefixed `_nn_cbw_`. Every reference is OWED (a new lane).
#
# Fault injection it must catch (test builds only): MOJOLEARN_NN_CBW_SABOTAGE_DROP_PATH (the rule overwrites the
# input's gradient, dropping the other path's term: the two-path check refuses and the gradient part moves) and
# MOJOLEARN_NN_CBW_SABOTAGE_SEED_ONE (the rule assumes an incoming gradient of 1: MyTanh's gradient differs from the
# built-in tanh's, refused). The defines are in the shared session code, so the same arm faults either side.


def _nn_cbw_f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def _nn_cbw_my_tanh(cap, shape):
    """MyTanh: forward y = tanh(x) saving y; backward grad_y * (1 - y * y)."""
    zeros = np.zeros(shape, np.float32)
    x = cap.tensor(zeros)
    forward = cap.capture(inputs=[x], outputs=[cap.tanh(x)])
    y, g = cap.tensor(zeros), cap.tensor(zeros)
    one = cap.full((1, 1), 1.0)
    backward = cap.capture(inputs=[y, g], outputs=[cap.mul(g, cap.sub(one, cap.mul(y, y)))])
    return forward, backward


@lane("nn-cbw-custom-vjp")
def _(ml, X, yc, yr, Xh=None):
    """Session.custom_op and Session.custom with MyTanh on fixture data: (1) loss 3 * sum(my_tanh(w)) against the
    built-in tanh's, values and gradient bit for bit; (2) the same input along two paths, 3 * (sum(my_tanh(x)) +
    sum(x * x)), the custom node recorded first, against the analytic 3 (1 - tanh(x)^2) + 6 x; (3) a program holding
    the operation saved, loaded on the same side and trained one SGD step (Program.fit) from the same start: the
    same parameter bits; (4) a backward with the wrong input count is refused by name. Parts: the gradients, the
    trained parameters."""
    import tempfile
    import os
    w0 = _nn_cbw_f32(X[:16, :3])
    x0 = _nn_cbw_f32(X[16:32, :3])
    t0 = _nn_cbw_f32(X[32:48, :3])
    m0 = _nn_cbw_f32(X[48:51, :3])
    ops = ml.ops
    with ops.Session() as cap:
        forward, backward = _nn_cbw_my_tanh(cap, (16, 3))
        g = cap.tensor(np.zeros((16, 3), np.float32))
        one_input = cap.capture(inputs=[g], outputs=[cap.tanh(g)])
        with ops.Session(record=True) as s:
            op = s.custom_op(forward, backward, saved=[0])
            w = s.tensor(w0, requires_grad=True)
            y = s.custom(op, w)
            s.backward(s.scale(s.sum(y), 3.0))
            g_custom, y_custom = s.grad(w), s.numpy(y)
            s.reset()
            x = s.tensor(x0, requires_grad=True)
            t = s.custom(op, x)
            z = s.mul(x, x)
            s.backward(s.scale(s.add(s.sum(t), s.sum(z)), 3.0))
            g_two = s.grad(x)
            try:
                s.custom_op(forward, one_input, saved=[0])
                refused = ""
            except ValueError as e:
                refused = str(e)
    if "the backward program takes 1 inputs; it must take 2" not in refused:
        raise AssertionError(f"a backward with the wrong input count was not refused by name: {refused!r}")
    with ops.Session(record=True) as r:
        w = r.tensor(w0, requires_grad=True)
        y = r.tanh(w)
        r.backward(r.scale(r.sum(y), 3.0))
        g_builtin, y_builtin = r.grad(w), r.numpy(y)
    _same_bytes("custom MyTanh values", y_custom, "built-in tanh values", y_builtin)
    _same_bytes("custom MyTanh gradient (seed 3)", g_custom, "built-in tanh gradient (seed 3)", g_builtin)
    x64 = x0.astype(np.float64)
    want = 3.0 * (1.0 - np.tanh(x64) ** 2) + 6.0 * x64
    err = np.abs(np.asarray(g_two, dtype=np.float64) - want)
    if not np.all(err <= 1e-5 + 1e-5 * np.abs(want)):
        raise AssertionError(f"two paths, seed 3: the gradient is off the analytic one by up to {err.max():.3g} "
                             "(a rule dropped a path or assumed an incoming gradient of 1)")
    with tempfile.TemporaryDirectory(prefix="ib-nn-cbw-") as tmp:
        path = os.path.join(tmp, "custom.mlp")
        with ops.Session() as s:
            forward, backward = _nn_cbw_my_tanh(s, (16, 3))
            op = s.custom_op(forward, backward, saved=[0])
            xin = s.tensor(np.zeros((16, 3), np.float32))
            m = s.tensor(m0)
            model = s.capture(inputs=[xin], outputs=[s.custom(op, s.matmul(xin, m))])
            model.save(path, verify=False)
            model.fit(x0, targets=t0, optimizer=s.optimizer("sgd", lr=0.1), loss="mse", epochs=1, batch_size=16,
                      shuffle=False)
            trained = s.numpy(m)
        with ops.Session() as s:
            loaded = s.load_program(path)
            loaded.fit(x0, targets=t0, optimizer=s.optimizer("sgd", lr=0.1), loss="mse", epochs=1, batch_size=16,
                       shuffle=False)
            again = s.numpy(loaded.params[0])
    _same_bytes("trained after save", trained, "trained after load", again)
    return _fit(dict(tanh=_h(np.asarray(g_custom), np.asarray(y_custom)), two_paths=_h(np.asarray(g_two)),
                     trained=_h(np.asarray(trained))))
