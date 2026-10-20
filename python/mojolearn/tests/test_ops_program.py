# SPDX-License-Identifier: Apache-2.0
"""Captured programs (Session.capture / Program.run): a program's outputs
equal the eager session's word for word for the guide's compositions (the
session quickstart, the mixture classifier's predict path, the Shapes
section's image classifier and attention product), a second run gives the
same words, a saved program loads into a new session and gives them again,
and the refusals hold. Prints PROGRAM_DIGEST lines to diff across CPU,
Metal, NVIDIA and AMD."""
import hashlib
import pathlib
import re

import numpy as np
import pytest

from mojolearn import _backend

GUIDE = pathlib.Path(__file__).resolve().parents[3] / "docs-site" / "user_guide" / "autograd.md"


def _has_entries():
    try:
        b = _backend.binding("_mojolearn_x_sequence")
        return hasattr(b, "ops_open") and hasattr(b, "ops_capture") and hasattr(b, "ops_program_run")
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_entries(),
                                reason="the sequence binding has no ops_capture/ops_program_run entries (rebuild it)")


def _words(a):
    return np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)


def _digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


def _emit(name, *arrays):
    print(f"PROGRAM_DIGEST {name} {_digest(*arrays)}")


def _check(name, build, arrays, others):
    """build(s) -> (input handles, output handles) on a session whose inputs
    hold `arrays`; the program run on `arrays` equals the eager outputs, a
    second run equals the first, and a run on `others` equals an eager
    session built on `others`."""
    from mojolearn import ops
    with ops.Session() as s:
        ins, outs = build(s, arrays)
        eager = [s.numpy(o) for o in outs]
        prog = s.capture(inputs=ins, outputs=outs)
        assert prog.input_shapes == tuple(s.shape(h) for h in ins)
        assert prog.output_shapes == tuple(e.shape for e in eager)
        s.reset()
        got = prog.run(*arrays)
        got = list(got) if isinstance(got, tuple) else [got]
        again = prog.run(*arrays)
        again = list(again) if isinstance(again, tuple) else [again]
        moved = prog.run(*others)
        moved = list(moved) if isinstance(moved, tuple) else [moved]
    for e, g, a in zip(eager, got, again):
        assert g.shape == e.shape
        assert np.array_equal(_words(g), _words(e)), f"{name}: the program differs from the eager session"
        assert np.array_equal(_words(a), _words(g)), f"{name}: a second run differs"
    with ops.Session() as s:
        _, outs2 = build(s, others)
        eager2 = [s.numpy(o) for o in outs2]
    for e, m in zip(eager2, moved):
        assert np.array_equal(_words(m), _words(e)), f"{name}: the run on new inputs differs from the eager session"
    _emit(name, *got)
    _emit(name + "_new_inputs", *moved)


def _quickstart(s, arrays):
    X, W = arrays
    x, w = s.tensor(X), s.tensor(W)
    h = s.linear(x, w, activation="relu")
    return [x, w], [s.softmax(h), s.sum(h)]


def test_program_session_quickstart():
    rng = np.random.default_rng(0)
    a = [rng.standard_normal((64, 4), dtype=np.float32), rng.standard_normal((4, 3), dtype=np.float32)]
    b = [rng.standard_normal((64, 4), dtype=np.float32), rng.standard_normal((4, 3), dtype=np.float32)]
    _check("quickstart", _quickstart, a, b)


def test_program_mixture_predict():
    rng = np.random.default_rng(0)
    params = {k: 0.1 * rng.standard_normal(sh, dtype=np.float32) for k, sh in
              (("W0", (4, 2)), ("b0", (1, 2)), ("W1", (4, 2)), ("b1", (1, 2)), ("Wg", (4, 2)), ("bg", (1, 2)))}
    first = np.array([[1.0], [0.0]], np.float32)
    second = np.array([[0.0], [1.0]], np.float32)

    def build(s, arrays):
        (X,) = arrays
        x = s.tensor(X)
        p = {k: s.tensor(v) for k, v in params.items()}
        z0 = s.linear(x, p["W0"], p["b0"])
        z1 = s.linear(x, p["W1"], p["b1"])
        gate = s.linear(x, p["Wg"], p["bg"], activation="softmax")
        logits = s.add(s.mul(s.matmul(gate, s.tensor(first)), z0), s.mul(s.matmul(gate, s.tensor(second)), z1))
        return [x], [s.softmax(logits), gate]

    _check("mixture", build, [rng.standard_normal((128, 4), dtype=np.float32)],
           [rng.standard_normal((128, 4), dtype=np.float32)])


def test_program_shapes_conv():
    rng = np.random.default_rng(0)
    K = 0.3 * rng.standard_normal((4, 1, 3, 3), dtype=np.float32)
    KB = 0.1 * rng.standard_normal((1, 4), dtype=np.float32)
    W = 0.1 * rng.standard_normal((64, 2), dtype=np.float32)

    def build(s, arrays):
        (images,) = arrays
        x = s.tensor(images)
        h = s.relu(s.conv2d(x, s.tensor(K), s.tensor(KB), padding=1))
        h = s.max_pool2d(h, 2)
        logits = s.linear(s.flatten(h), s.tensor(W))
        return [x], [s.log_softmax(logits), s.avg_pool2d(h, 2)]

    _check("shapes_conv", build, [rng.standard_normal((8, 1, 8, 8), dtype=np.float32)],
           [rng.standard_normal((8, 1, 8, 8), dtype=np.float32)])


def test_program_shapes_attention():
    rng = np.random.default_rng(0)

    def build(s, arrays):
        Q, Kv = arrays
        q, kv = s.tensor(Q), s.tensor(Kv)
        scores = s.scale(s.matmul(q, s.transpose(kv)), 0.5)
        out = s.matmul(s.softmax(scores), kv)
        merged = s.reshape(s.permute(out, (0, 2, 1, 3)), (2, 6, 16))
        pieces = s.split(merged, 2, axis=2)
        return [q, kv], [merged, s.concat([pieces[1], pieces[0]], axis=2), s.max(merged, axis=1)]

    a = [rng.standard_normal((2, 4, 6, 4), dtype=np.float32), rng.standard_normal((2, 4, 6, 4), dtype=np.float32)]
    b = [rng.standard_normal((2, 4, 6, 4), dtype=np.float32), rng.standard_normal((2, 4, 6, 4), dtype=np.float32)]
    _check("shapes_attention", build, a, b)


def test_program_save_and_load(tmp_path):
    from mojolearn import ops
    rng = np.random.default_rng(1)
    X = rng.standard_normal((32, 4), dtype=np.float32)
    W1 = rng.standard_normal((4, 8), dtype=np.float32)
    W2 = rng.standard_normal((8, 2), dtype=np.float32)
    path = tmp_path / "mlp.program"
    with ops.Session() as s:
        x = s.tensor(X)
        h = s.layer_norm(s.linear(x, s.tensor(W1), activation="tanh"),
                         s.tensor(np.ones((1, 8), np.float32)), s.tensor(np.zeros((1, 8), np.float32)))
        y = s.softmax(s.linear(h, s.tensor(W2)))
        eager = s.numpy(y)
        prog = s.capture(x, y)
        assert len(prog.params) == 4
        prog.save(path)
    with ops.Session() as s:
        served = s.load_program(path)
        assert served.input_shapes == ((32, 4),) and served.output_shapes == ((32, 2),)
        got = served.run(X)
    assert np.array_equal(_words(got), _words(eager))
    _emit("saved_mlp", got)


def test_program_guide_block(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    if not GUIDE.is_file():
        pytest.skip(f"no guide at {GUIDE} (run from the repository tree)")
    blocks = dict(re.findall(r"<!-- block: ([a-z0-9-]+) -->\n```python\n(.*?)```", GUIDE.read_text(), flags=re.S))
    ns = {"__name__": "autograd_guide"}
    exec(compile(blocks["program"], "autograd.md[program]", "exec"), ns)
    assert np.array_equal(_words(ns["first"]), _words(ns["eager"]))
    assert np.array_equal(_words(ns["served_second"]), _words(ns["second"]))
    _emit("guide_program_first", ns["first"])
    _emit("guide_program_second", ns["second"])


def test_program_refusals():
    from mojolearn import ops
    rng = np.random.default_rng(2)
    X = rng.standard_normal((8, 3), dtype=np.float32)
    with ops.Session(record=True) as s:
        x = s.tensor(X)
        y = s.relu(x)
        with pytest.raises(Exception, match="inference session"):
            s.capture(x, y)
    with ops.Session() as s:
        x = s.tensor(X)
        table = s.tensor(rng.standard_normal((5, 2), dtype=np.float32))
        e = s.embedding(table, np.array([0, 3, 4], np.int64))
        with pytest.raises(Exception, match="embedding and take"):
            s.capture(table, e)
        w = s.tensor(rng.standard_normal((3, 2), dtype=np.float32))
        prog = s.capture(x, s.matmul(x, w))
        with pytest.raises(ValueError, match="shape"):
            prog.run(np.zeros((4, 3), np.float32))
        with pytest.raises(ValueError, match="input arrays"):
            prog.run(X, X)
        s.free(w)
        with pytest.raises(Exception, match="freed"):
            prog.run(X)
