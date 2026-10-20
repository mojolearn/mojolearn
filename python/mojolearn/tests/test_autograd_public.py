# SPDX-License-Identifier: Apache-2.0
"""The public autograd surface as its documentation shows it: every Python
block of docs-site/user_guide/autograd.md runs verbatim (the Sequential
quickstart, the operation-session quickstart and the two-expert mixture
classifier), and every ``>>>`` example of the Sequential and ops.Session
docstrings runs. Small sizes. Each case checks shapes and finite values,
runs twice for the same digest, and prints PUBLIC_DIGEST lines to diff
across CPU, Metal, NVIDIA and AMD."""
import doctest
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
        return hasattr(b, "ops_open") and hasattr(b, "model_fit")
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _has_entries(),
                                reason="the sequence binding has no ops_open/model_fit entries (rebuild it)")


def _blocks():
    if not GUIDE.is_file():
        pytest.skip(f"no guide at {GUIDE} (run from the repository tree)")
    blocks = re.findall(r"```python\n(.*?)```", GUIDE.read_text(), flags=re.S)
    # 0 Sequential quickstart; 1-2 session quickstart (2 continues 1); 3 session training loop (continues 1);
    # 4 mixture classifier
    assert len(blocks) == 5, f"expected 5 python blocks in {GUIDE.name}, found {len(blocks)}"
    return blocks


def _run(*sources):
    ns = {"__name__": "autograd_guide"}
    for k, src in enumerate(sources):
        exec(compile(src, f"{GUIDE.name}[block {k}]", "exec"), ns)
    return ns


def _digest(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
    return h.hexdigest()[:16]


def _emit(name, digest):
    print(f"PUBLIC_DIGEST {name} {digest}")


def test_guide_sequential_quickstart():
    b = _blocks()
    ns = _run(b[0])
    clf, proba, labels = ns["clf"], ns["proba"], ns["labels"]
    assert proba.shape == (200, 2) and labels.shape == (200,)
    assert np.isfinite(proba).all() and np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert set(np.unique(labels)) <= set(clf.classes_.tolist())
    assert clf.loss_curve_[-1] < clf.loss_curve_[0]
    ns2 = _run(b[0])
    assert _digest(clf.params_, proba) == _digest(ns2["clf"].params_, ns2["proba"])
    _emit("guide_sequential_params", _digest(clf.params_))
    _emit("guide_sequential_proba", _digest(proba))


def test_guide_session_quickstart():
    b = _blocks()
    ns = _run(b[1], b[2])
    probs, total, grad_w = ns["probs"], ns["total"], ns["grad_w"]
    X, W, Y = ns["X"], ns["W"], ns["Y"]
    assert probs.shape == (64, 3) and np.isfinite(probs).all()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert np.isfinite(total) and total >= 0
    assert grad_w.shape == (4, 3) and np.isfinite(grad_w).all()
    reference = (2.0 / Y.size) * X.T @ (X @ W - Y)
    assert np.allclose(grad_w, reference, rtol=1e-4, atol=1e-5)
    ns2 = _run(b[1], b[2])
    assert _digest(probs, [total], grad_w) == _digest(ns2["probs"], [ns2["total"]], ns2["grad_w"])
    _emit("guide_session_probs", _digest(probs))
    _emit("guide_session_total", _digest([total]))
    _emit("guide_session_grad", _digest(grad_w))


def test_guide_session_training_loop():
    blocks = _blocks()
    ns = _run(blocks[1], blocks[3])
    trained = ns["trained"]
    assert trained.shape == ns["W"].shape and np.isfinite(trained).all()
    assert not np.array_equal(trained, ns["W"])
    ns2 = _run(blocks[1], blocks[3])
    assert _digest(trained) == _digest(ns2["trained"])
    _emit("guide_session_trained", _digest(trained))


def test_guide_mixture_classifier(capsys):
    b = _blocks()
    ns = _run(b[3])
    proba, hard, params, accuracy = ns["proba"], ns["hard"], ns["params"], ns["accuracy"]
    assert proba.shape == (128, 2) and hard.shape == (128, 2)
    assert np.isfinite(proba).all() and np.isfinite(hard).all()
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert 0.0 <= accuracy <= 1.0
    assert ns["expert"] in ("0", "1")
    printed = capsys.readouterr().out
    assert hashlib.sha256(proba.tobytes()).hexdigest()[:16] in printed
    flat = [params[k] for k in sorted(params)]
    ns2 = _run(b[3])
    capsys.readouterr()
    assert _digest(proba, hard, *flat) == _digest(ns2["proba"], ns2["hard"],
                                                  *[ns2["params"][k] for k in sorted(ns2["params"])])
    with capsys.disabled():
        _emit("guide_mixture_proba", _digest(proba))
        _emit("guide_mixture_hard", _digest(hard))
        _emit("guide_mixture_params", _digest(*flat))
        _emit("guide_mixture_accuracy", _digest([accuracy]))


def _docstring_examples():
    import mojolearn
    from mojolearn import ops
    return [("ops", ops), ("ops.Session", ops.Session), ("Sequential", mojolearn.Sequential),
            ("Sequential.add", mojolearn.Sequential.add), ("Sequential.compile", mojolearn.Sequential.compile)]


@pytest.mark.parametrize("name", ["ops", "ops.Session", "Sequential", "Sequential.add", "Sequential.compile"])
def test_docstring_examples_run(name):
    obj = dict(_docstring_examples())[name]
    finder = doctest.DocTestFinder(recurse=False)
    runner = doctest.DocTestRunner(optionflags=doctest.ELLIPSIS)
    tests = [t for t in finder.find(obj, name) if t.examples]
    assert tests, f"{name} has no examples"
    for t in tests:
        runner.run(t, clear_globs=False)
    result = runner.summarize(verbose=False)
    assert result.failed == 0, f"{name}: {result.failed} of {result.attempted} examples failed"
    ns = tests[0].globs
    for key in ("probs", "pred", "proba", "total", "grad_w"):
        if key in ns:
            _emit(f"doc_{name.replace('.', '_')}_{key}", _digest(np.atleast_1d(ns[key])))
