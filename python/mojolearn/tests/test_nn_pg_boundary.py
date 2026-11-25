# SPDX-License-Identifier: Apache-2.0
"""The Python boundary of ops.Session's optimizer parameter groups (lane nn-blocks-pg, 2026-11-25): admission
of `param_groups`, its refusals in words, and the state-dict round trip (save, load into a fresh session, one
step: the same bits as no save). The update itself is checked natively (checks/native_tests/nn_pg.mojo) and by
the verify lane ops-param-groups."""
import inspect

import numpy as np
import pytest

from mojolearn import _backend


def _has_entries():
    try:
        return hasattr(_backend.binding("_mojolearn_x_sequence"), "ops_session_call")
    except Exception:
        return False


needs_binding = pytest.mark.skipif(not _has_entries(),
                                   reason="the sequence binding has no ops_session_call entry (rebuild it)")


def test_signature():
    from mojolearn import ops
    params = list(inspect.signature(ops.Session.param_group_optimizer).parameters)
    assert params[:4] == ["self", "param_groups", "kind", "lr"]
    assert params[-2:] == ["clip_grad_norm", "clip_grad_value"]
    for name in ("learning_rates", "optimizer_state_dict", "load_optimizer_state_dict"):
        assert callable(getattr(ops.Session, name))


_P = np.array([[1, -2], [0.5, -0.25]], np.float32)
_Q = np.array([[0.75, 0.125], [-1.5, 2]], np.float32)
_A = np.array([[0.5, -1], [2, 0.25]], np.float32)
_B = np.array([[-0.75, 1.5], [0.5, 1]], np.float32)


def _loss(s, p, q, a, b):
    return s.sum(s.add(s.matmul(a, p), s.matmul(b, q)))


def _groups(p, q):
    return [{"params": [p], "lr": 0.01, "weight_decay": 0.1},
            {"params": [q], "lr": 0.05, "weight_decay": 0.0, "betas": (0.8, 0.999)}]


@needs_binding
@pytest.mark.parametrize("groups, words", [
    (lambda p, q: [{"params": [p]}, {"params": [p, q]}], "is also in param_groups[0]"),
    (lambda p, q: [{"params": [p]}, {"params": []}], "param_groups[1] has no parameters"),
    (lambda p, q: [{"params": [p]}, {"lr": 0.1}], "param_groups[1] has no 'params'"),
    (lambda p, q: [{"params": [p], "lrr": 0.1}], "unknown parameter group key 'lrr'"),
    (lambda p, q: [{"params": [p], "momentum": 0.9}], "the adam optimizer does not take 'momentum'"),
    (lambda p, q: [{"params": [p], "clip_grad_norm": 1.0}], "'clip_grad_norm' is optimizer-wide"),
    (lambda p, q: [{"params": [p], "schedule": "step"}], "'schedule' is optimizer-wide"),
    (lambda p, q: [{"params": [p], "betas": (0.9,)}], "must be a pair (beta1, beta2)"),
    (lambda p, q: [[p]], "param_groups[0] must be a dict"),
])
def test_refusals(groups, words):
    from mojolearn import ops
    with ops.Session(record=True) as s:
        p = s.tensor(_P, requires_grad=True)
        q = s.tensor(_Q, requires_grad=True)
        with pytest.raises((ValueError, TypeError), match=None) as info:
            s.param_group_optimizer(groups(p, q), "adam")
        assert words in str(info.value)


@needs_binding
def test_step_outside_groups_refused():
    from mojolearn import ops
    with ops.Session(record=True) as s:
        p = s.tensor(_P, requires_grad=True)
        q = s.tensor(_Q, requires_grad=True)
        r = s.tensor(_A, requires_grad=True)
        opt = s.param_group_optimizer([{"params": [p]}, {"params": [q]}], "sgd", lr=0.1)
        s.backward(s.sum(s.add(s.matmul(r, p), q)))
        with pytest.raises(ValueError) as info:
            s.step(opt, [p, r])
        assert "in no parameter group" in str(info.value)


@needs_binding
def test_learning_rates_and_schedule():
    from mojolearn import ops
    with ops.Session(record=True) as s:
        p = s.tensor(_P, requires_grad=True)
        q = s.tensor(_Q, requires_grad=True)
        opt = s.param_group_optimizer(_groups(p, q), "adamw", schedule="step", step_size=1, gamma=0.5)
        assert s.learning_rates(opt) == [float(np.float32(0.01)), float(np.float32(0.05))]
        s.scheduler_step(opt)
        assert s.learning_rates(opt) == [float(np.float32(0.01) * np.float32(0.5)),
                                         float(np.float32(0.05) * np.float32(0.5))]


@needs_binding
def test_state_dict_round_trip_same_bits():
    from mojolearn import ops
    with ops.Session(record=True) as s:
        p = s.tensor(_P, requires_grad=True)
        q = s.tensor(_Q, requires_grad=True)
        a, b = s.tensor(_A), s.tensor(_B)
        opt = s.param_group_optimizer(_groups(p, q), "adamw", clip_grad_norm=1.0)
        for _ in range(2):
            s.backward(_loss(s, p, q, a, b))
            s.step(opt, [p, q])
            s.reset()
        sd = s.optimizer_state_dict(opt)
        assert sorted(sd) == ["epoch", "param_groups", "state"]
        assert [g["params"] for g in sd["param_groups"]] == [[0], [1]]
        assert sd["param_groups"][1]["betas"] == (np.float32(0.8).item(), np.float32(0.999).item())
        saved_state = {k: {n: (s.numpy(v) if n in ("exp_avg", "exp_avg_sq") else v) for n, v in st.items()}
                       for k, st in sd["state"].items()}
        weights = (s.numpy(p), s.numpy(q))
        s.backward(_loss(s, p, q, a, b))
        s.step(opt, [p, q])
        straight = (s.numpy(p), s.numpy(q))
    with ops.Session(record=True) as t:
        tp = t.tensor(weights[0], requires_grad=True)
        tq = t.tensor(weights[1], requires_grad=True)
        ta, tb = t.tensor(_A), t.tensor(_B)
        topt = t.param_group_optimizer([{"params": [tp]}, {"params": [tq]}], "adamw", clip_grad_norm=1.0)
        loaded = {"epoch": sd["epoch"], "param_groups": sd["param_groups"],
                  "state": {k: {n: (t.tensor(v) if isinstance(v, np.ndarray) else v) for n, v in st.items()}
                            for k, st in saved_state.items()}}
        t.load_optimizer_state_dict(topt, loaded)
        assert t.learning_rates(topt) == [float(np.float32(0.01)), float(np.float32(0.05))]
        t.backward(_loss(t, tp, tq, ta, tb))
        t.step(topt, [tp, tq])
        resumed = (t.numpy(tp), t.numpy(tq))
    for x, y in zip(straight, resumed):
        assert x.tobytes() == y.tobytes()


@needs_binding
def test_load_refuses_other_groups():
    from mojolearn import ops
    with ops.Session(record=True) as s:
        p = s.tensor(_P, requires_grad=True)
        q = s.tensor(_Q, requires_grad=True)
        opt = s.param_group_optimizer(_groups(p, q), "adam")
        sd = s.optimizer_state_dict(opt)
        sd["param_groups"] = sd["param_groups"][:1]
        with pytest.raises(ValueError) as info:
            s.load_optimizer_state_dict(opt, sd)
        assert "the state has 1 parameter groups and this optimizer has 2" in str(info.value)


# ---------------------------------------------------------------- Program.fit
@needs_binding
def test_program_fit_parameter_outside_groups_refused():
    from mojolearn import ops
    with ops.Session() as s:
        w = s.tensor(_P)
        v = s.tensor(_Q)
        xin = s.tensor(np.zeros((4, 2), np.float32))
        program = s.capture([xin], [s.matmul(s.matmul(xin, w), v)])
        opt = s.param_group_optimizer([{"params": [w]}], "sgd", lr=0.1)
        with pytest.raises(ValueError) as info:
            program.fit(np.ones((4, 2), np.float32), targets=np.ones((4, 2), np.float32), optimizer=opt, epochs=1)
        assert "Program.fit: the parameter is in no parameter group" in str(info.value)


# ---------------------------------------------------------------- SGD / Adam / AdamW
def _has_training():
    try:
        import mojolearn
        mojolearn.SGD([np.zeros(2, np.float32)])
        return True
    except Exception:
        return False


needs_training = pytest.mark.skipif(not _has_training(), reason="the training binding is not built")


@needs_training
@pytest.mark.parametrize("cls, groups, words", [
    ("SGD", lambda a, b: [{"params": [a]}, {"params": [a, b]}], "is also in param_groups[0]"),
    ("SGD", lambda a, b: [{"params": [a]}, {"params": []}], "param_groups[1] has no parameters"),
    ("SGD", lambda a, b: [{"params": [a]}, {"lr": 0.1}], "param_groups[1] has no 'params'"),
    ("SGD", lambda a, b: [{"params": [a]}, [b]], "param_groups[1] must be a dict"),
    ("Adam", lambda a, b: [{"params": [a], "momentum": 0.9}], "the adam optimizer does not take 'momentum'"),
    ("AdamW", lambda a, b: [{"params": [a], "lrr": 0.1}], "unknown parameter group key 'lrr'"),
    ("AdamW", lambda a, b: [{"params": [a], "beta1": 0.8}], "give 'betas' as a pair"),
])
def test_training_refusals(cls, groups, words):
    import mojolearn
    a, b = np.zeros(4, np.float32), np.zeros(3, np.float32)
    with pytest.raises((ValueError, TypeError)) as info:
        getattr(mojolearn, cls)(groups(a, b))
    assert words in str(info.value)


@needs_training
def test_training_param_groups_keyword_names_the_form():
    import mojolearn
    with pytest.raises(TypeError) as info:
        mojolearn.SGD([np.zeros(2, np.float32)], param_groups=[])
    assert "pass the parameter groups as `params`" in str(info.value)


@needs_training
def test_training_groups_round_trip_same_bits():
    import mojolearn
    w = [np.full(4, 0.5, np.float32), np.full(3, -0.25, np.float32)]
    opt = mojolearn.AdamW([{"params": [w[0]], "lr": 1e-3}, {"params": [w[1]], "lr": 1e-2, "weight_decay": 0.0}],
                          weight_decay=0.01)
    g = [np.linspace(-1, 1, 4, dtype=np.float32), np.linspace(2, 3, 3, dtype=np.float32)]
    for _ in range(2):
        opt.step([x.copy() for x in g], max_norm=0.5)
    sd = opt.state_dict()
    assert sorted(sd) == ["param_groups", "state"]
    assert [p["params"] for p in sd["param_groups"]] == [[0], [1]]
    again = [x.copy() for x in w]
    loaded = mojolearn.AdamW([{"params": [again[0]]}, {"params": [again[1]]}])
    loaded.load_state_dict(sd)
    opt.step([x.copy() for x in g], max_norm=0.5)
    loaded.step([x.copy() for x in g], max_norm=0.5)
    for x, y in zip(w, again):
        assert x.tobytes() == y.tobytes()
    with pytest.raises(ValueError) as info:
        opt.step([g[0].copy()])
    assert "1 gradients for 2 parameters across the groups" in str(info.value)
