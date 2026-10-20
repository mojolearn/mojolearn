# SPDX-License-Identifier: Apache-2.0
"""mojolearn.ops.Session optimizers (autograd/ops_api.mojo optimizer / step):
parameters updated in place on the executor with the optimizer state held in
the session.

The central check: a two-layer MLP trained through the session (forward,
mse, backward, step, scheduler_step, reset) ends on the same float32 words
as Sequential.fit of the same description from the same starting weights
(full batch, no shuffle), for SGD, SGD with Nesterov momentum, Adam and AdamW
with weight decay, and the step and exponential schedules. Both run the
same OP_OPT element body with the same host scalars. Also: an optimizer
state saved and loaded into a new session continues bit for bit; zero_grad
followed by an accumulating backward equals a plain backward; refusals.
Tiny data. OPS_OPT_DIGEST lines diff across CPU, Metal, NVIDIA and AMD."""
import hashlib

import numpy as np
import pytest

from mojolearn import _backend


def _has_step_entry():
    try:
        return hasattr(_backend.binding("_mojolearn_x_sequence"), "ops_step")
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_step_entry(),
                                reason="the sequence binding has no ops_step entry (rebuild it)")

N, D, H = 24, 3, 5
EPOCHS = 6


def _ops():
    from mojolearn import ops
    return ops


def _digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


def _data():
    rng = np.random.default_rng(3)
    X = rng.standard_normal((N, D)).astype(np.float32)
    Y = rng.standard_normal((N, 1)).astype(np.float32)
    # Sequential's flat layout: W1 (D, H), b1 (1, H), W2 (H, 1), b2 (1, 1)
    P0 = (0.5 * rng.standard_normal(D * H + H + H + 1)).astype(np.float32)
    return X, Y, P0


def _split(P):
    a = D * H
    return [P[:a].reshape(D, H), P[a:a + H].reshape(1, H), P[a + H:a + 2 * H].reshape(H, 1),
            P[a + 2 * H:].reshape(1, 1)]


def _sequential_fit(seq_optimizer, options):
    """Sequential.fit's trainer (model_fit) from the starting weights P0
    (init off), full batch, no shuffle."""
    import mojolearn
    X, Y, P0 = _data()
    m = mojolearn.Sequential().linear(H).relu().linear(1)
    m = m.compile(loss="mse", optimizer=seq_optimizer, **options)
    o = m._options
    prog, table = m._program(D), m._table()
    b = m._binding()
    _, yd, target = m._targets(Y, N, 1, "mse")
    from mojolearn._cast_in import describe
    xd = describe(X, "X")
    params = P0.copy()
    curve = np.zeros(EPOCHS, dtype=np.float32)
    ip = [N, 0, target, o["opt"], o["flags"], o["sched"], o["step_size"], N, EPOCHS, 0, 0, 0,
          o["n_iter_no_change"], 0]
    fp = [o["lr"], o["f1"], o["f2"], o["eps"], o["weight_decay"], 0.0, o["tol"], o["gamma"],
          o["huber_delta"]] + table
    n_iter = int(b.model_fit([xd, yd, params.ctypes.data, curve.ctypes.data], prog, ip, fp))
    assert n_iter == EPOCHS
    return params


def _session_fit(kind, options, epochs=EPOCHS):
    X, Y, P0 = _data()
    with _ops().Session(record=True) as s:
        ps = [s.tensor(p, requires_grad=True) for p in _split(P0)]
        x, y = s.tensor(X), s.tensor(Y)
        opt = s.optimizer(kind, **options)
        for _ in range(epochs):
            h = s.linear(x, ps[0], ps[1], activation="relu")
            s.backward(s.mse(s.linear(h, ps[2], ps[3]), y))
            s.step(opt, ps)
            s.scheduler_step(opt)
            s.reset()
        return np.concatenate([s.numpy(p).ravel() for p in ps])


# (session kind, session options, Sequential optimizer, Sequential options)
CASES = {
    "sgd_step_schedule": ("sgd", dict(lr=0.05, schedule="step", step_size=2, gamma=0.5),
                          "sgd", dict(lr=0.05, schedule="step", step_size=2, gamma=0.5)),
    "sgd_nesterov": ("sgd", dict(lr=0.05, momentum=0.9, nesterov=True, weight_decay=0.01),
                     "momentum", dict(lr=0.05, momentum=0.9, nesterov=True, weight_decay=0.01)),
    "adam": ("adam", dict(lr=0.01), "adam", dict(lr=0.01)),
    "adam_exponential": ("adam", dict(lr=0.02, schedule="exponential", gamma=0.9, weight_decay=0.001),
                         "adam", dict(lr=0.02, schedule="exponential", gamma=0.9, weight_decay=0.001)),
    "adamw": ("adamw", dict(lr=0.01, weight_decay=0.05, beta1=0.8, beta2=0.99),
              "adamw", dict(lr=0.01, weight_decay=0.05, beta1=0.8, beta2=0.99)),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_session_step_equals_sequential_fit(case):
    kind, opts, seq_kind, seq_opts = CASES[case]
    ours = _session_fit(kind, opts)
    ref = _sequential_fit(seq_kind, seq_opts)
    assert np.isfinite(ours).all()
    assert not np.array_equal(ours, _data()[2]), "the parameters did not move"
    assert ours.tobytes() == ref.tobytes(), f"{case}: session {_digest(ours)} != Sequential.fit {_digest(ref)}"
    _emit(case, _digest(ours))


def _emit(name, digest):
    print(f"OPS_OPT_DIGEST {name} {digest}")


def test_learning_rate_schedule():
    with _ops().Session(record=True) as s:
        opt = s.optimizer("sgd", lr=0.1, schedule="step", step_size=2, gamma=0.5)
        rates = []
        for _ in range(5):
            rates.append(s.learning_rate(opt))
            s.scheduler_step(opt)
    assert rates[0] == rates[1] == np.float32(0.1)
    assert rates[2] == rates[3] == np.float32(0.05)
    assert rates[4] == np.float32(0.025)


def _run_steps(s, opt, ps, x, y, k):
    for _ in range(k):
        h = s.linear(x, ps[0], ps[1], activation="relu")
        s.backward(s.mse(s.linear(h, ps[2], ps[3]), y))
        s.step(opt, ps)
        s.reset()


@pytest.mark.parametrize("kind,opts", [("adam", dict(lr=0.01)), ("sgd", dict(lr=0.05, momentum=0.9))])
def test_optimizer_state_save_and_load(kind, opts):
    X, Y, P0 = _data()
    ops = _ops()
    # one run of 5 steps
    with ops.Session(record=True) as s:
        ps = [s.tensor(p, requires_grad=True) for p in _split(P0)]
        x, y = s.tensor(X), s.tensor(Y)
        opt = s.optimizer(kind, **opts)
        _run_steps(s, opt, ps, x, y, 5)
        straight = np.concatenate([s.numpy(p).ravel() for p in ps])
    # 3 steps, a checkpoint read back, then 2 steps in a new session
    with ops.Session(record=True) as s:
        ps = [s.tensor(p, requires_grad=True) for p in _split(P0)]
        x, y = s.tensor(X), s.tensor(Y)
        opt = s.optimizer(kind, **opts)
        _run_steps(s, opt, ps, x, y, 3)
        weights = [s.numpy(p) for p in ps]
        saved = []
        for p in ps:
            st = s.optimizer_state(opt, p)
            assert st["step"] == 3
            saved.append({k: (s.numpy(v) if k in ("exp_avg", "exp_avg_sq", "momentum_buffer") else v)
                          for k, v in st.items()})
    with ops.Session(record=True) as s:
        ps = [s.tensor(w, requires_grad=True) for w in weights]
        x, y = s.tensor(X), s.tensor(Y)
        opt = s.optimizer(kind, **opts)
        for p, st in zip(ps, saved):
            s.load_optimizer_state(opt, p, {k: (s.tensor(v) if isinstance(v, np.ndarray) else v)
                                            for k, v in st.items()})
        _run_steps(s, opt, ps, x, y, 2)
        resumed = np.concatenate([s.numpy(p).ravel() for p in ps])
        assert s.optimizer_state(opt, ps[0])["step"] == 5
    assert resumed.tobytes() == straight.tobytes()
    _emit("resume_" + kind, _digest(resumed))


def test_zero_grad_then_accumulate_equals_backward():
    X, Y, P0 = _data()
    with _ops().Session(record=True) as s:
        w = s.tensor(_split(P0)[0], requires_grad=True)
        x, y = s.tensor(X), s.tensor(np.repeat(Y, H, axis=1))
        s.backward(s.mse(s.matmul(x, w), y))
        plain = s.grad(w)
        # the same loss recorded again on the same graph: without zero_grad
        # an accumulating backward would add to `plain`
        s.zero_grad()
        with pytest.raises(Exception):
            s.grad(w)
        s.backward(s.mse(s.matmul(x, w), y), accumulate=True)
        again = s.grad(w)
    assert plain.tobytes() == again.tobytes()


def test_step_refusals():
    ops = _ops()
    with ops.Session(record=True) as s:
        w = s.tensor(np.ones((2, 2), dtype=np.float32), requires_grad=True)
        c = s.tensor(np.ones((2, 2), dtype=np.float32))
        opt = s.optimizer("adam")
        with pytest.raises(Exception):
            s.step(opt, [w])               # no backward yet
        s.backward(s.sum(s.matmul(c, w)))
        with pytest.raises(Exception):
            s.step(opt, [c])               # not trainable
        with pytest.raises(Exception):
            s.step(opt, [w, w])            # listed twice
        with pytest.raises(ValueError):
            s.step(opt + 1, [w])           # not an optimizer of this session
        with pytest.raises(Exception):
            s.optimizer("sgd", nesterov=True)   # Nesterov without momentum
        with pytest.raises(Exception):
            s.optimizer("adam", lr=0.0)
        with pytest.raises(ValueError):
            s.optimizer("lion")
        s.step(opt, w)
        assert s.optimizer_state(opt, w)["step"] == 1
    with ops.Session() as s:
        with pytest.raises(Exception):
            s.zero_grad()                  # not a recording session
