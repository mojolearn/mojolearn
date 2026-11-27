# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# SEQUENTIAL PARAMETER GROUPS' LANES (lane nn-blocks-pgseq, 2026-11-25; autograd/nn_pgseq.mojo).
#
# Verify through `device=` (docs/plans/VERIFY_THROUGH_DEVICE.md): the lanes are device-agnostic (no `device=`);
# the harness's view gives Sequential and the training module their side.
#
# sequential-param-groups: `Sequential.compile(param_groups=..., no_decay=...)` with momentum, AdamW and LAMB (its
# global clip at 0.05, which binds). Checks in the cell: groups with the ungrouped settings give the ungrouped
# fit's bytes; save_checkpoint after 2 epochs, from_checkpoint and 2 resumed epochs give the bytes of 4 epochs.
# Parts: the grouped fit's loss curve, parameters, first moments and predictions per optimizer.
# Sabotage arms (tools/identity_lanes/coverage_sabotage.tsv, coverage-autograd-pgseq_*): each run at the
# neighbouring group's lr, and LAMB's clip per group, each on one side; both move this lane's parts.
#
# training-param-groups-device: the device-gradient part owed by lane nn-blocks-pg: SGD / AdamW with parameter
# groups stepped from TrainDeviceArray gradients (clipped on the device, max_norm 0.5) give the bytes of the same
# steps from host gradients. GPU side only (the CPU training binding has no device array pool).
#
# Loaded by tools/identity_break.py `_load_lane_fragments` in the harness namespace: np, _h, _fit, _same_bytes,
# _one_sided and `lane` are the harness's; helpers here are prefixed `_nn_pgseq_`. Every reference is OWED.
import os
import tempfile


def _nn_pgseq_model(ml):
    return ml.Sequential(numeric_mode="identical").linear(12).tanh().linear(1)


def _nn_pgseq_fit(model, x, y, epochs, **kw):
    return model.fit(x, y, epochs=epochs, batch_size=16, seed=0, **kw)


@lane("sequential-param-groups")
def _(ml, X, yc, yr, Xh=None):
    """Sequential.compile with param_groups: the body (layer 0) at 0.002, the head (layer 2) at 0.02 with its own
    betas or momentum, no_decay='bias', a step schedule (gamma 0.5 every 2 epochs) scaling every group; for
    'momentum', 'adamw' and 'lamb' (max_grad_norm 0.05: the global clip binds). In the cell: (1) groups with the
    compile settings give the ungrouped bytes; (2) 2 epochs, save_checkpoint, from_checkpoint, 2 resumed epochs
    give the bytes of 4 epochs. Parts: loss curve, parameters, first moments and held-out predictions of the
    4-epoch grouped fit, per optimizer."""
    x = np.ascontiguousarray(X[:96, :6], dtype=np.float32)
    y = np.ascontiguousarray(yr[:96], dtype=np.float32)
    xh = np.ascontiguousarray((X if Xh is None else Xh)[:64, :6], dtype=np.float32)
    parts = {}
    for optimizer in ("momentum", "adamw", "lamb"):
        base = dict(optimizer=optimizer, lr=0.002, weight_decay=0.01, schedule="step", step_size=2, gamma=0.5)
        if optimizer == "lamb":
            base["max_grad_norm"] = 0.05
        head = {"layers": [2], "lr": 0.02}
        head.update({"momentum": 0.5} if optimizer == "momentum" else {"betas": (0.8, 0.99)})
        # (1) groups with the compile settings: the ungrouped bytes
        plain = _nn_pgseq_fit(_nn_pgseq_model(ml).compile(**base), x, y, 2, shuffle=False)
        same = _nn_pgseq_fit(_nn_pgseq_model(ml).compile(param_groups=[{"layers": [0]}, {"params": ["2.weight"]}],
                                                          **base), x, y, 2, shuffle=False)
        _same_bytes(f"{optimizer} ungrouped", plain.params_, f"{optimizer} groups with the compile settings",
                    same.params_)
        # (2) the grouped fit, straight and through a checkpoint
        kw = dict(base, param_groups=[head], no_decay="bias")
        straight = _nn_pgseq_fit(_nn_pgseq_model(ml).compile(**kw), x, y, 4)
        first = _nn_pgseq_fit(_nn_pgseq_model(ml).compile(**kw), x, y, 2)
        with tempfile.TemporaryDirectory(prefix="ib-nn-pgseq-") as tmp:
            path = os.path.join(tmp, "grouped.ckpt")
            first.save_checkpoint(path)
            loaded = ml.Sequential.from_checkpoint(path, numeric_mode="identical")
        loaded.fit(x, y, epochs=2, resume=True)
        _same_bytes(f"{optimizer} grouped 4 epochs", straight.params_, f"{optimizer} grouped 2 + 2 from a checkpoint",
                    loaded.params_)
        _same_bytes(f"{optimizer} grouped 4 epochs (moments)", straight.optimizer_m_,
                    f"{optimizer} grouped 2 + 2 from a checkpoint (moments)", loaded.optimizer_m_)
        parts[f"{optimizer}_loss"] = _h(np.asarray(straight.loss_curve_, dtype=np.float32))
        parts[f"{optimizer}_params"] = _h(straight.params_)
        parts[f"{optimizer}_m"] = _h(straight.optimizer_m_)
        parts[f"{optimizer}_held"] = _h(np.asarray(straight.predict(xh), dtype=np.float32))
    return _fit(parts)


def _nn_pgseq_arrays(X):
    return [np.ascontiguousarray(X[:8, :6], dtype=np.float32) * np.float32(0.25),
            np.ascontiguousarray(X[8:12, :6], dtype=np.float32) * np.float32(0.25),
            np.ascontiguousarray(X[12:13, :4], dtype=np.float32) * np.float32(0.25)]


def _nn_pgseq_grads(X, step):
    return [np.ascontiguousarray(X[16 + step:24 + step, :6], dtype=np.float32),
            np.ascontiguousarray(X[24 + step:28 + step, :6], dtype=np.float32),
            np.ascontiguousarray(X[28 + step:29 + step, :4], dtype=np.float32)]


@lane("sequence-optim-adafactor")
def _(ml, X, yc, yr, Xh=None):
    """One matrix and one bias, two batches per epoch. Resume with separate rates for the two tensors."""
    x = np.ascontiguousarray(X[:8, :3], dtype=np.float32)
    y = np.ascontiguousarray(yr[:8], dtype=np.float32)
    held = np.ascontiguousarray((X if Xh is None else Xh)[:4, :3], dtype=np.float32)
    def fit(epochs, groups=None):
        model = ml.Sequential(numeric_mode="identical").linear(1)
        model.compile(loss="mse", optimizer="adafactor", lr=0.01, param_groups=groups)
        return model.fit(x, y, epochs=epochs, batch_size=4, shuffle=False, seed=7)

    # At x=(1,2) the loaded model predicts (2.5,-0.5). Mean squared error
    # against (1.5,1.5) has output gradient (1,-2). Use those exact gradients
    # to step the standalone optimizer on the matrix in native (in,out) layout.
    weight = np.array([[1, -1], [0.5, 0.5]], dtype=np.float32)
    bias = np.array([0.5, -0.5], dtype=np.float32)
    initial = {"0.weight": weight.T.copy(), "0.bias": bias.copy()}
    probe = ml.Sequential(input_shape=(2,), numeric_mode="identical").linear(2).compile(
        loss="mse", optimizer="adafactor", lr=0.03125)
    probe.load_state_dict(initial)
    probe.fit(np.array([[1, 2]], dtype=np.float32), np.array([[1.5, 1.5]], dtype=np.float32),
              epochs=1, batch_size=1, shuffle=False)
    standalone = ml.Adafactor([weight, bias], lr=0.03125)
    standalone.step([np.array([[1, -2], [2, -4]], dtype=np.float32), np.array([1, -2], dtype=np.float32)])
    state = probe.state_dict()
    _same_bytes("matrix after standalone step", weight.T.copy(), "Sequential weight", state["0.weight"])
    _same_bytes("vector after standalone step", bias, "Sequential bias", state["0.bias"])

    plain = fit(2)
    equal = fit(2, [{"params": ["0.bias"]}])
    for name in ("params_", "optimizer_m_", "optimizer_v_"):
        _same_bytes(name + " ungrouped", getattr(plain, name), name + " equal groups", getattr(equal, name))
    groups = [{"params": ["0.bias"], "lr": 0.025}]
    straight = fit(2, groups)
    first = fit(1, groups)
    with tempfile.TemporaryDirectory(prefix="ib-adafactor-") as tmp:
        path = os.path.join(tmp, "model.ckpt")
        first.save_checkpoint(path)
        loaded = ml.Sequential.from_checkpoint(path, numeric_mode="identical")
    loaded.fit(x, y, epochs=1, resume=True)
    parts = {"plain": _h(plain.params_), "prediction": _h(straight.predict(held))}
    for name in ("params_", "optimizer_m_", "optimizer_v_", "loss_curve_"):
        expected = np.asarray(getattr(straight, name))
        _same_bytes(name + " uninterrupted", expected, name + " resumed", np.asarray(getattr(loaded, name)))
        parts[name] = _h(expected)
    return _fit(parts)


_batch_decl("n/a:training-trajectory (the lane compares full optimizer updates and checkpoint resume)",
            "sequence-optim-adafactor")


@lane("training-param-groups-device")
def _(ml, X, yc, yr, Xh=None):
    """mojolearn.SGD / AdamW with parameter groups stepped three times from device gradients (TrainDeviceArray,
    clipped on the device with max_norm 0.5) give the bytes of the same grouped optimizer stepped from host
    gradients, and the same total norms. Parts: the weights and norms of each kind."""
    from mojolearn._training_impl import train_to_device
    parts = {}
    for name in ("AdamW", "SGD"):
        cls = getattr(ml, name)
        if name == "SGD":
            opts = ({"lr": 0.01, "momentum": 0.9}, {"lr": 0.05, "momentum": 0.5, "nesterov": True, "weight_decay": 0.0})
            base = {"lr": 0.01, "momentum": 0.9, "weight_decay": 0.001}
        else:
            opts = ({"lr": 0.001}, {"lr": 0.01, "betas": (0.8, 0.99), "weight_decay": 0.0})
            base = {"lr": 0.001, "weight_decay": 0.01}
        w = _nn_pgseq_arrays(X)
        r = [a.copy() for a in w]
        on_device = cls([dict(params=[w[0]], **opts[0]), dict(params=[w[1], w[2]], **opts[1])], **base)
        # TrainDeviceArray gradients are the device-parameter step's (training/optimizer_public_common.mojo
        # GRADS_DEVICE; a one-group step refuses them before params_to_device()): the groups' parameters go to
        # the device first and come back before the compare (lane gs-fix-2: the lane stepped host parameters).
        on_device.params_to_device()
        on_host = cls([dict(params=[r[0]], **opts[0]), dict(params=[r[1], r[2]], **opts[1])], **base)
        dn, hn = [], []
        for step in range(3):
            g = _nn_pgseq_grads(X, step)
            dn.append(on_device.step([train_to_device(a.copy()) for a in g], max_norm=0.5))
            hn.append(on_host.step([a.copy() for a in g], max_norm=0.5))
        on_device.params_to_host()
        for k in range(3):
            _same_bytes(f"{name} groups, device gradients", w[k], f"{name} groups, host gradients", r[k])
        _same_bytes(f"{name} norms, device gradients", np.asarray(dn, np.float64), f"{name} norms, host gradients",
                    np.asarray(hn, np.float64))
        parts[f"{name}_weights"] = _h(*w)
        parts[f"{name}_norms"] = _h(np.asarray(dn, np.float64))
    return _fit(parts)


_one_sided("training-param-groups-device", missing="cpu",
           reason="the CPU training binding has no device array pool (TrainDeviceArray gradients exist on the GPU "
                  "side only)")
