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


def _nn_pg_program(ml, X, y, groups_of):
    """Program.fit of a captured two-layer graph (inference session, captured parameters) with the optimizer
    `groups_of(s, w)` builds; returns the weights after two epochs and the loss curve."""
    x = _nn_pg_f32(X[:64, :6])
    t = _nn_pg_f32(y[:64]).reshape(64, 1)
    init = [_nn_pg_f32(X[64:70, :4]) * np.float32(0.5), np.zeros((1, 4), np.float32),
            _nn_pg_f32(X[70:74, :1]) * np.float32(0.5), np.zeros((1, 1), np.float32)]
    with ml.ops.Session() as s:
        w = [s.tensor(a) for a in init]
        xin = s.tensor(np.zeros((64, 6), dtype=np.float32))
        program = s.capture([xin], [s.linear(s.linear(xin, w[0], w[1], activation="tanh"), w[2], w[3])])
        opt = groups_of(s, w)
        program.fit(x, targets=t, optimizer=opt, epochs=2, batch_size=16, shuffle=False, seed=0)
        return [s.numpy(h) for h in w], np.asarray(program.loss_curve_, dtype=np.float32)


@lane("ops-param-groups-program-fit")
def _(ml, X, yc, yr, Xh=None):
    """Program.fit with an optimizer that has parameter groups: (1) two groups with the SAME options as a one-group
    optimizer give the same bytes as it (the groups change nothing but the table); (2) the body at a low rate with
    weight decay, the head at a higher rate, biases without decay, clip_grad_norm 0.05 over every group. Parts: the
    weights and loss curve of (2)."""
    def one(s, w):
        return s.optimizer("adamw", lr=0.01, weight_decay=0.01, clip_grad_norm=0.05)

    def uniform(s, w):
        return s.param_group_optimizer([{"params": w[:2]}, {"params": w[2:]}], "adamw", lr=0.01, weight_decay=0.01,
                                       clip_grad_norm=0.05)

    def grouped(s, w):
        return s.param_group_optimizer([{"params": [w[0], w[2]], "lr": 0.002},
                                        {"params": [w[1], w[3]], "lr": 0.02, "weight_decay": 0.0}],
                                       "adamw", weight_decay=0.01, clip_grad_norm=0.05)

    a = _nn_pg_program(ml, X, yr, one)
    b = _nn_pg_program(ml, X, yr, uniform)
    for k in range(4):
        _same_bytes("Program.fit one group", a[0][k], "Program.fit two identical groups", b[0][k])
    c = _nn_pg_program(ml, X, yr, grouped)
    return _fit(dict(weights=_h(*c[0]), loss=_h(c[1])))


def _nn_pg_train_arrays(X):
    return [_nn_pg_f32(X[:8, :6]) * np.float32(0.25), _nn_pg_f32(X[8:12, :6]) * np.float32(0.25),
            _nn_pg_f32(X[12:13, :4]) * np.float32(0.25)]


def _nn_pg_train_grads(X, step):
    return [_nn_pg_f32(X[16 + step:24 + step, :6]), _nn_pg_f32(X[24 + step:28 + step, :6]),
            _nn_pg_f32(X[28 + step:29 + step, :4])]


@lane("training-param-groups")
def _(ml, X, yc, yr, Xh=None):
    """mojolearn.SGD / AdamW with parameter groups (params as a list of dicts, torch's form): three steps with
    max_norm 0.5 must give the same bytes as clip_grad_norm_ ONCE over every gradient followed by one optimizer per
    group stepped without a clip (the one-clip contract, 3.5); then state_dict, a fresh grouped optimizer at the
    same weights loading it, and one more step in each: the same bytes. Parts: the weights, the total norms and
    the state of each kind."""
    parts = {}
    for name in ("AdamW", "SGD"):
        cls = getattr(ml, name)
        if name == "SGD":
            opts = ({"lr": 0.01, "momentum": 0.9}, {"lr": 0.05, "momentum": 0.5, "nesterov": True, "weight_decay": 0.0})
            base = {"lr": 0.01, "momentum": 0.9, "weight_decay": 0.001}
        else:
            opts = ({"lr": 0.001}, {"lr": 0.01, "betas": (0.8, 0.99), "weight_decay": 0.0})
            base = {"lr": 0.001, "weight_decay": 0.01}
        w = _nn_pg_train_arrays(X)
        r = [a.copy() for a in w]
        grouped = cls([dict(params=[w[0]], **opts[0]), dict(params=[w[1], w[2]], **opts[1])], **base)
        first = cls([r[0]], **dict(base, **opts[0]))
        second = cls([r[1], r[2]], **dict(base, **opts[1]))
        norms = []
        for step in range(3):
            g = _nn_pg_train_grads(X, step)
            norms.append(grouped.step([a.copy() for a in g], max_norm=0.5))
            h = [a.copy() for a in g]
            ml.clip_grad_norm_(h, 0.5)
            first.step([h[0]])
            second.step([h[1], h[2]])
        for k in range(3):
            _same_bytes(f"{name} param groups", w[k], f"{name} clip once then one optimizer a group", r[k])
        sd = grouped.state_dict()
        again = [a.copy() for a in w]
        loaded = cls([dict(params=[again[0]]), dict(params=[again[1], again[2]])], **base)
        loaded.load_state_dict(sd)
        g = _nn_pg_train_grads(X, 3)
        grouped.step([a.copy() for a in g], max_norm=0.5)
        loaded.step([a.copy() for a in g], max_norm=0.5)
        for k in range(3):
            _same_bytes(f"{name} grouped step", w[k], f"{name} step after load_state_dict", again[k])
        state = [np.asarray(sd["state"][gi][key]) for gi in sorted(sd["state"]) for key in ("exp_avg", "exp_avg_sq")]
        parts[f"{name}_weights"] = _h(*w)
        parts[f"{name}_norms"] = _h(np.asarray(norms, np.float64))
        parts[f"{name}_state"] = _h(*state)
    return _fit(parts)
