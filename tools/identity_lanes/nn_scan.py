# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE SCAN OVER A CAPTURED CELL (lane nn-blocks-scan, 2026-11-25): `ops.Session.scan`.
#
# A fragment of tools/identity_break.py (loaded by `_load_lane_fragments` into its namespace; np, os, tempfile, _h,
# _fit, _same_bytes, _seq, _hw, lane and _batch_decl are that namespace's, read when the lane runs). Device-agnostic
# (docs/plans/VERIFY_THROUGH_DEVICE.md contract 1): no `device=`; the harness's view gives `ops.Session` the side.
# Helpers here are prefixed `_nn_scan_`. Every reference is OWED.

_NN_SCAN_T, _NN_SCAN_B, _NN_SCAN_F, _NN_SCAN_H, _NN_SCAN_O = 8, 4, 3, 5, 2


def _nn_scan_cell(s):
    """A two-state cell (h, c, x) -> (h', c', y) on inference session s: g = tanh(x W + h U + b),
    c' = 0.9 c + g, h' = tanh(c'), y = h' V. Its parameters, in Program.params order: W, U, b, V."""
    B, F, H, O = _NN_SCAN_B, _NN_SCAN_F, _NN_SCAN_H, _NN_SCAN_O
    h = s.tensor(np.zeros((B, H), np.float32))
    c = s.tensor(np.zeros((B, H), np.float32))
    x = s.tensor(np.zeros((B, F), np.float32))
    w = s.tensor(_hw((F, H), "ops-scan:w", -0.5, 0.5))
    u = s.tensor(_hw((H, H), "ops-scan:u", -0.5, 0.5))
    b = s.tensor(_hw((1, H), "ops-scan:b", -0.1, 0.1))
    v = s.tensor(_hw((H, O), "ops-scan:v", -0.5, 0.5))
    g = s.tanh(s.add(s.add(s.matmul(x, w), s.matmul(h, u)), b))
    c2 = s.add(s.scale(c, 0.9), g)
    h2 = s.tanh(c2)
    y = s.matmul(h2, v)
    return s.capture(inputs=[h, c, x], outputs=[h2, c2, y])


@lane("ops-scan-cell")
def _(ml, X, yc, yr, Xh=None):
    """ops.Session.scan (lane nn-blocks-scan, 2026-11-25): a two-state cell (h, c) captured in an inference session
    and scanned in a recording session over (8, 4, 3) fixture inputs from given h_0 and c_0, with lengths 8 5 3 6;
    mse of the outputs against fixture targets plus mean(h_T^2), backward and one Adam step. Parts: the loss and
    outputs, the gradients (the four cell parameters, h_0, c_0 and xs), the updated parameters. Then in one
    inference session: the reverse scan (outputs and final states); the scan captured with its inputs into a
    Program, run (the eager bytes), saved and loaded (the same bytes). Sabotage arms
    coverage-autograd-nn_scan_state_lag_gpu / _cpu (step t fed the state of step t - 2) move every part on one side;
    coverage-autograd-nn_scan_init_grad_gpu / _cpu (the initial state detached) move the h_0 / c_0 gradient parts
    on one side."""
    T, B, F, H, O = _NN_SCAN_T, _NN_SCAN_B, _NN_SCAN_F, _NN_SCAN_H, _NN_SCAN_O
    xs = _seq(X, T, B, F)
    target = _seq(X, T, B, O, skip=4096)
    lengths = np.array([8, 5, 3, 6], dtype=np.float32)
    h0v = _hw((B, H), "ops-scan:h0", -0.5, 0.5)
    c0v = _hw((B, H), "ops-scan:c0", -0.5, 0.5)
    names = ("w", "u", "b", "v")
    init = {"w": _hw((F, H), "ops-scan:w", -0.5, 0.5), "u": _hw((H, H), "ops-scan:u", -0.5, 0.5),
            "b": _hw((1, H), "ops-scan:b", -0.1, 0.1), "v": _hw((H, O), "ops-scan:v", -0.5, 0.5)}
    parts = {}
    with ml.ops.Session(numeric_mode="identical") as c:
        cell = _nn_scan_cell(c)
        with ml.ops.Session(record=True, numeric_mode="identical") as s:
            p = {k: s.tensor(init[k], requires_grad=True) for k in names}  # glue: four parameters
            h0 = s.tensor(h0v, requires_grad=True)
            c0 = s.tensor(c0v, requires_grad=True)
            x = s.tensor(xs, requires_grad=True)
            ys, (h_t, c_t) = s.scan(cell, x, [h0, c0], params=[p[k] for k in names], lengths=s.tensor(lengths))
            loss = s.add(s.mse(ys, s.tensor(target)), s.mean(s.mul(h_t, h_t)))
            parts["loss"] = _h(np.asarray(s.numpy(loss)), np.asarray(s.numpy(ys)), np.asarray(s.numpy(c_t)))
            s.backward(loss)
            parts["grad"] = _h(*[np.asarray(s.grad(p[k])) for k in names],  # glue: one array per parameter
                               np.asarray(s.grad(x)))
            parts["grad_init"] = _h(np.asarray(s.grad(h0)), np.asarray(s.grad(c0)))
            s.step(s.optimizer("adam", lr=0.01), [p[k] for k in names])
            parts["param"] = _h(*[np.asarray(s.numpy(p[k])) for k in names])  # glue: one array per parameter
    with ml.ops.Session(numeric_mode="identical") as c:
        cell = _nn_scan_cell(c)
        x = c.tensor(xs)
        h0 = c.tensor(h0v)
        c0 = c.tensor(c0v)
        ry, (rh, rc) = c.scan(cell, x, [h0, c0], reverse=True)
        eager = [np.asarray(c.numpy(ry)), np.asarray(c.numpy(rh)), np.asarray(c.numpy(rc))]
        parts["reverse"] = _h(*eager)
        prog = c.capture(inputs=[x, h0, c0], outputs=[ry, rh, rc])
        got = prog.run(xs, h0v, c0v)
        for k, name in enumerate(("ys", "h_T", "c_T")):  # driver-loop: over the three outputs
            _same_bytes("Session.scan " + name, eager[k], "Program.run " + name, np.asarray(got[k]))
        with tempfile.TemporaryDirectory(prefix="ib-ops-scan-") as tmp:
            saved = os.path.join(tmp, "scan.mlp")
            prog.save(saved)
            with ml.ops.Session(numeric_mode="identical") as d:
                again = d.load_program(saved).run(xs, h0v, c0v)
                for k, name in enumerate(("ys", "h_T", "c_T")):  # driver-loop: over the three outputs
                    _same_bytes("Session.scan " + name, eager[k], "load_program(save).run " + name,
                                np.asarray(again[k]))
    return _fit(parts)


_batch_decl("n/a:training-step (one seeded training step through the ops session; its forward outputs are "
            "train parts, digested whole)", "ops-scan-cell")
