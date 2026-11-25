# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE OPTIMIZER PARAMETER GROUPS' LANE (lane nn-blocks-pg, 2026-11-25; autograd/nn_pg.mojo).
#
# Verify through `device=` (docs/plans/VERIFY_THROUGH_DEVICE.md): the lane is device-agnostic (no `device=`);
# the harness's view gives `ops.Session` its side. It covers `Session.param_group_optimizer`, `step` of a grouped
# optimizer (one global clip over every group), `scheduler_step` / `learning_rates` (one schedule scales every
# group), `optimizer_state_dict` / `load_optimizer_state_dict` (a resume gives the same bits as no save).
#
# Sabotage arms (tools/identity_lanes/coverage_sabotage.tsv, coverage-autograd-pg_*): a clip per group instead of
# one global clip, and every group stepped at group 0's rate, each on one side; both move this lane's weights.
#
# Loaded by tools/identity_break.py `_load_lane_fragments` in the harness namespace: np, _h, _fit, _same_bytes and
# `lane` are the harness's; helpers here are prefixed `_nn_pg_`. Every reference is OWED.


def _nn_pg_f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


def _nn_pg_model(s, X, y, kind, clip):
    """A two-layer model (6 -> 4 tanh -> 1) on 32 fixture rows: three groups (the body at a low rate with weight
    decay, the head at a higher rate, the bias at the head's rate without decay), stepped five times with the
    schedule advanced once after step three. Returns the session handles."""
    body = s.tensor(_nn_pg_f32(X[32:38, :4]) * np.float32(0.5), requires_grad=True)
    head = s.tensor(_nn_pg_f32(X[40:44, :1]) * np.float32(0.5), requires_grad=True)
    bias = s.tensor(np.zeros((1, 1), np.float32), requires_grad=True)
    x, t = s.tensor(_nn_pg_f32(X[:32, :6])), s.tensor(_nn_pg_f32(y[:32]).reshape(32, 1))
    groups = [{"params": [body], "lr": 0.002, "weight_decay": 0.01},
              {"params": [head], "lr": 0.02},
              {"params": [bias], "lr": 0.02, "weight_decay": 0.0}]
    if kind == "sgd":
        groups[1]["momentum"] = 0.5
        groups[2]["nesterov"] = True
        opt = s.param_group_optimizer(groups, "sgd", momentum=0.9, weight_decay=0.001, schedule="step",
                                      step_size=1, gamma=0.5, clip_grad_norm=clip)
    else:
        groups[1]["betas"] = (0.8, 0.99)
        opt = s.param_group_optimizer(groups, kind, weight_decay=0.001, schedule="step", step_size=1, gamma=0.5,
                                      clip_grad_norm=clip)
    return body, head, bias, x, t, opt


def _nn_pg_step(s, body, head, bias, x, t, opt):
    s.backward(s.mse(s.linear(s.linear(x, body, activation="tanh"), head, bias), t))
    s.step(opt, [body, head, bias])
    s.reset()


@lane("ops-param-groups")
def _(ml, X, yc, yr, Xh=None):
    """ops.Session optimizer parameter groups: AdamW and SGD (momentum, Nesterov in one group) with three groups
    (body, head, bias without decay), clip_grad_norm 0.05 over every group's gradients (it binds at these
    values: the coefficient is below one, so the global clip moves the weights), a step schedule advanced after
    step three; then `optimizer_state_dict`, a fresh session loading it (`load_optimizer_state_dict`) and one more
    step in each, which must give the same bytes. Parts: the weights after five steps, every group's rate, the
    first moments of the state dict and the weights after the sixth step."""
    parts = {}
    for kind in ("adamw", "sgd"):
        with ml.ops.Session(record=True) as s:
            body, head, bias, x, t, opt = _nn_pg_model(s, X, yr, kind, 0.05)
            for i in range(5):
                _nn_pg_step(s, body, head, bias, x, t, opt)
                if i == 2:
                    s.scheduler_step(opt)
            weights = [s.numpy(h) for h in (body, head, bias)]
            rates = np.asarray(s.learning_rates(opt), np.float64)
            sd = s.optimizer_state_dict(opt)
            names = ("exp_avg", "exp_avg_sq", "momentum_buffer")
            saved = {k: {n: (s.numpy(v) if n in names else v) for n, v in st.items()} for k, st in sd["state"].items()}
            groups = sd["param_groups"]
            epoch = sd["epoch"]
            _nn_pg_step(s, body, head, bias, x, t, opt)
            straight = [s.numpy(h) for h in (body, head, bias)]
        with ml.ops.Session(record=True) as r:
            rb = r.tensor(weights[0], requires_grad=True)
            rh = r.tensor(weights[1], requires_grad=True)
            rc = r.tensor(weights[2], requires_grad=True)
            rx, rt = r.tensor(_nn_pg_f32(X[:32, :6])), r.tensor(_nn_pg_f32(yr[:32]).reshape(32, 1))
            ropt = r.param_group_optimizer([{"params": [rb]}, {"params": [rh]}, {"params": [rc]}], kind,
                                           schedule="step", step_size=1, gamma=0.5, clip_grad_norm=0.05,
                                           momentum=0.9 if kind == "sgd" else 0.0)
            r.load_optimizer_state_dict(ropt, {"epoch": epoch, "param_groups": groups,
                                               "state": {k: {n: (r.tensor(v) if n in names else v)
                                                             for n, v in st.items()} for k, st in saved.items()}})
            _nn_pg_step(r, rb, rh, rc, rx, rt, ropt)
            resumed = [r.numpy(h) for h in (rb, rh, rc)]
        for k in range(3):
            _same_bytes(f"{kind} grouped step", straight[k], f"{kind} step after load_optimizer_state_dict", resumed[k])
        moments = [saved[k][n] for k in sorted(saved) for n in names if n in saved[k]]
        parts[f"{kind}_weights"] = _h(*weights)
        parts[f"{kind}_rates"] = _h(rates)
        parts[f"{kind}_state"] = _h(*moments)
        parts[f"{kind}_resumed"] = _h(*resumed)
    return _fit(parts)
