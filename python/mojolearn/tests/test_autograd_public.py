# SPDX-License-Identifier: Apache-2.0
"""The public autograd surface as its documentation shows it: every Python
block of docs-site/user_guide/autograd.md runs verbatim (the Sequential
quickstart, the operation-session quickstart, the two-expert mixture
classifier, the shapes section and the weight import), and every ``>>>`` example of the Sequential and ops.Session
docstrings runs (load_state_dict and load_weights included). Small sizes. Each case checks shapes and finite values,
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
    """The guide's python blocks by name: each fence is preceded by `<!-- block: name -->`."""
    if not GUIDE.is_file():
        pytest.skip(f"no guide at {GUIDE} (run from the repository tree)")
    found = re.findall(r"<!-- block: ([a-z0-9-]+) -->\n```python\n(.*?)```", GUIDE.read_text(), flags=re.S)
    blocks = dict(found)
    assert len(found) == len(blocks) >= 11, f"named python blocks in {GUIDE.name}: {sorted(blocks)}"
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
    ns = _run(b["seq-quickstart"])
    clf, proba, labels = ns["clf"], ns["proba"], ns["labels"]
    assert proba.shape == (200, 2) and labels.shape == (200,)
    assert np.isfinite(proba).all() and np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert set(np.unique(labels)) <= set(clf.classes_.tolist())
    assert clf.loss_curve_[-1] < clf.loss_curve_[0]
    ns2 = _run(b["seq-quickstart"])
    assert _digest(clf.params_, proba) == _digest(ns2["clf"].params_, ns2["proba"])
    _emit("guide_sequential_params", _digest(clf.params_))
    _emit("guide_sequential_proba", _digest(proba))


def test_guide_session_quickstart():
    b = _blocks()
    ns = _run(b["sess-setup"], b["sess-loss"])
    probs, total, grad_w = ns["probs"], ns["total"], ns["grad_w"]
    X, W, Y = ns["X"], ns["W"], ns["Y"]
    assert probs.shape == (64, 3) and np.isfinite(probs).all()
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5)
    assert np.isfinite(total) and total >= 0
    assert grad_w.shape == (4, 3) and np.isfinite(grad_w).all()
    reference = (2.0 / Y.size) * X.T @ (X @ W - Y)
    assert np.allclose(grad_w, reference, rtol=1e-4, atol=1e-5)
    ns2 = _run(b["sess-setup"], b["sess-loss"])
    assert _digest(probs, [total], grad_w) == _digest(ns2["probs"], [ns2["total"]], ns2["grad_w"])
    _emit("guide_session_probs", _digest(probs))
    _emit("guide_session_total", _digest([total]))
    _emit("guide_session_grad", _digest(grad_w))


def test_guide_session_training_loop():
    blocks = _blocks()
    ns = _run(blocks["sess-setup"], blocks["sess-loss"], blocks["sess-train"])
    trained = ns["trained"]
    assert trained.shape == ns["W"].shape and np.isfinite(trained).all()
    assert not np.array_equal(trained, ns["W"])
    ns2 = _run(blocks["sess-setup"], blocks["sess-loss"], blocks["sess-train"])
    assert _digest(trained) == _digest(ns2["trained"])
    _emit("guide_session_trained", _digest(trained))


def test_guide_export_and_load(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    blocks = _blocks()
    ns = _run(blocks["seq-quickstart"], blocks["export"])
    assert np.array_equal(ns["labels_served"], ns["clf"].predict(ns["X"]))
    assert np.array_equal(ns["served"].predict_proba(ns["X"]), ns["clf"].predict_proba(ns["X"]))
    _emit("guide_export_labels", _digest(ns["labels_served"].astype(np.float32)))


def test_guide_mixture_classifier(capsys):
    b = _blocks()
    ns = _run(b["mixture"])
    proba, hard, params, accuracy = ns["proba"], ns["hard"], ns["params"], ns["accuracy"]
    assert proba.shape == (128, 2) and hard.shape == (128, 2)
    assert np.isfinite(proba).all() and np.isfinite(hard).all()
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert 0.0 <= accuracy <= 1.0
    assert ns["expert"] in ("0", "1")
    printed = capsys.readouterr().out
    assert hashlib.sha256(proba.tobytes()).hexdigest()[:16] in printed
    flat = [params[k] for k in sorted(params)]
    ns2 = _run(b["mixture"])
    capsys.readouterr()
    assert _digest(proba, hard, *flat) == _digest(ns2["proba"], ns2["hard"],
                                                  *[ns2["params"][k] for k in sorted(ns2["params"])])
    with capsys.disabled():
        _emit("guide_mixture_proba", _digest(proba))
        _emit("guide_mixture_hard", _digest(hard))
        _emit("guide_mixture_params", _digest(*flat))
        _emit("guide_mixture_accuracy", _digest([accuracy]))



def test_guide_shapes(capsys):
    b = _blocks()
    ns = _run(b["shapes"])
    grads = {"grad_k": (4, 1, 3, 3), "grad_kb": (1, 4), "grad_w": (64, 2),
             "attention": (2, 6, 16), "grad_q": (2, 4, 6, 4)}
    for name, shape in grads.items():
        assert ns[name].shape == shape and np.isfinite(ns[name]).all(), name
    assert np.any(ns["grad_k"] != 0) and np.any(ns["grad_q"] != 0)
    printed = capsys.readouterr().out
    assert ns["digest"].hexdigest()[:16] in printed
    ns2 = _run(b["shapes"])
    capsys.readouterr()
    assert _digest(*[ns[k] for k in grads]) == _digest(*[ns2[k] for k in grads])
    with capsys.disabled():
        _emit("guide_shapes_conv_grads", _digest(ns["grad_k"], ns["grad_kb"], ns["grad_w"]))
        _emit("guide_shapes_attention", _digest(ns["attention"]))
        _emit("guide_shapes_attention_grad", _digest(ns["grad_q"]))


def test_guide_shapes_sequential(capsys):
    b = _blocks()
    ns = _run(b["shapes-seq"])
    cnn, proba, proba_flat = ns["cnn"], ns["proba"], ns["proba_flat"]
    assert proba.shape == (64, 2) and np.isfinite(proba).all()
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert np.array_equal(proba, proba_flat)
    assert cnn.params_.size == 4 * 1 * 3 * 3 + 4 + 64 * 2 + 2 and np.isfinite(cnn.params_).all()
    assert cnn.loss_curve_[-1] < cnn.loss_curve_[0]
    printed = capsys.readouterr().out
    assert hashlib.sha256(cnn.params_.tobytes() + proba.tobytes()).hexdigest()[:16] in printed
    ns2 = _run(b["shapes-seq"])
    capsys.readouterr()
    assert _digest(cnn.params_, proba) == _digest(ns2["cnn"].params_, ns2["proba"])
    with capsys.disabled():
        _emit("guide_shapes_seq_params", _digest(cnn.params_))
        _emit("guide_shapes_seq_proba", _digest(proba))

def test_guide_program(tmp_path, monkeypatch):
    b = _blocks()
    if "program" not in b or not hasattr(_backend.binding("_mojolearn_x_sequence"), "ops_capture"):
        pytest.skip("no captured-program entries in this binding (rebuild it)")
    monkeypatch.chdir(tmp_path)
    ns = _run(b["program"])
    first, second, eager = ns["first"], ns["second"], ns["eager"]
    assert first.shape == (8, 2) and np.isfinite(first).all()
    assert np.array_equal(first.view(np.uint32), eager.view(np.uint32))
    assert np.array_equal(ns["served_second"].view(np.uint32), second.view(np.uint32))
    ns2 = _run(b["program"])
    assert _digest(first, second) == _digest(ns2["first"], ns2["second"])
    _emit("guide_program_first", _digest(first))
    _emit("guide_program_second", _digest(second))
def test_guide_import_weights():
    b = _blocks()
    ns = _run(b["import-weights"])
    logits, logits_session = ns["logits"], ns["logits_session"]
    assert logits.shape == (32, 2) and np.isfinite(logits).all()
    assert np.allclose(logits, logits_session, atol=1e-5)
    state = ns["state"]
    assert np.array_equal(ns["imported"].params_[:64].reshape(4, 16), state["0.weight"].T)
    ns2 = _run(b["import-weights"])
    assert _digest(ns["imported"].params_, logits, logits_session) == \
        _digest(ns2["imported"].params_, ns2["logits"], ns2["logits_session"])
    _emit("guide_import_params", _digest(ns["imported"].params_))
    _emit("guide_import_logits", _digest(logits))
    _emit("guide_import_session", _digest(logits_session))


def _docstring_examples():
    import mojolearn
    from mojolearn import ops
    return [("ops", ops), ("ops.Session", ops.Session), ("Sequential", mojolearn.Sequential),
            ("Sequential.add", mojolearn.Sequential.add), ("Sequential.compile", mojolearn.Sequential.compile),
            ("ops.Program", ops.Program), ("ops.Session.capture", ops.Session.capture),
            ("Sequential.load_state_dict", mojolearn.Sequential.load_state_dict),
            ("ops.Session.load_weights", ops.Session.load_weights)]


@pytest.mark.parametrize("name", ["ops", "ops.Session", "Sequential", "Sequential.add", "Sequential.compile",
                                  "ops.Program", "ops.Session.capture",
                                  "Sequential.load_state_dict", "ops.Session.load_weights"])
def test_docstring_examples_run(name):
    if name.startswith("ops.Program") or name == "ops.Session.capture":
        if not hasattr(_backend.binding("_mojolearn_x_sequence"), "ops_capture"):
            pytest.skip("no captured-program entries in this binding (rebuild it)")
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
