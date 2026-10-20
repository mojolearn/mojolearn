# SPDX-License-Identifier: Apache-2.0
"""Save, load and resume for Sequential and the operation session
(autograd/model_checkpoint.mojo through training/checkpoint.mojo's container).

(a) Training 4 epochs straight equals training 2, saving, loading and
training 2 more, word for word: the parameters, the optimizer buffers and
the loss curve (Adam with a step schedule and shuffling, Nesterov momentum on
a classifier, AdamW with layer norm, early stopping).
(b) A checkpoint written on the CPU host column resumes on the GPU column
(and the reverse) to the same digests, and an inference export gives the
same predictions on either column: the cross-vendor resume claim. On a
machine with one GPU binding this compares the host column with that GPU;
the NVIDIA and AMD runs are the same test on those boxes.

Small models (scikit-learn-test sizes).
MOJOLEARN_TEST_INSTALLED=1 requires non-editable installed wheels. Run pytest
with python -I and --import-mode=importlib against this file; the harness may
live in the checkout while the package under test comes from site-packages.
"""
import hashlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import mojolearn as ml
from mojolearn import _backend

_REQUIRE_INSTALLED = os.environ.get("MOJOLEARN_TEST_INSTALLED") == "1"
_PACKAGE_EVIDENCE = Path(__file__).with_name("_package_evidence.py")
_package_evidence = runpy.run_path(str(_PACKAGE_EVIDENCE))["package_evidence"]
if _REQUIRE_INSTALLED:
    _package_evidence(ml, require_installed=True)


@pytest.fixture(scope="module", autouse=True)
def _record_package():
    yield
    print("PACKAGE_EVIDENCE", json.dumps(_package_evidence(ml, require_installed=_REQUIRE_INSTALLED),
                                         sort_keys=True))


def _has_entries():
    try:
        found = hasattr(_backend.binding("_mojolearn_x_sequence"), "model_save_checkpoint")
    except Exception:
        if _REQUIRE_INSTALLED:
            raise
        return False
    if _REQUIRE_INSTALLED and not found:
        raise RuntimeError("Installed sequence binding has no model_save_checkpoint entry")
    return found


pytestmark = pytest.mark.skipif(not _has_entries(),
                                reason="the sequence binding has no checkpoint entries (rebuild it)")


def _data():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 5)).astype(np.float32)
    yr = (X[:, 0] * 2 - X[:, 1] + 0.5 * X[:, 2]).astype(np.float32)
    y3 = (X[:, 0] > 0).astype(np.int64) + (X[:, 1] > 0.5)
    return X, yr, y3


def _h(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def _words(m):
    """Every resumable word of a fitted model."""
    return dict(params=_h(m.params_), m=_h(m._opt_m), v=_h(m._opt_v),
                curve=_h(np.asarray(m.loss_curve_, dtype=np.float32)), n_iter=m.n_iter_)


_CASES = {
    "adam_step_schedule": lambda: (ml.Sequential().linear(8).tanh().linear(1)
                                   .compile(loss="mse", optimizer="adam", lr=0.02, schedule="step",
                                            step_size=1, gamma=0.7), "yr", dict(batch_size=32)),
    "nesterov_classifier": lambda: (ml.Sequential().linear(8).relu().linear(3)
                                    .compile(loss="cross_entropy", optimizer="momentum", lr=0.05,
                                             nesterov=True), "y3", dict(batch_size=25)),
    "adamw_layer_norm": lambda: (ml.Sequential().linear(6).layer_norm().relu().linear(1)
                                 .compile(loss="huber", optimizer="adamw", lr=0.01, weight_decay=0.01,
                                          schedule="exponential", gamma=0.9), "yr", dict(batch_size=40)),
    "sgd_unshuffled": lambda: (ml.Sequential().linear(4).sigmoid().linear(1)
                               .compile(loss="mse", optimizer="sgd", lr=0.05), "yr",
                               dict(batch_size=30, shuffle=False)),
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_straight_equals_save_load_continue(case, tmp_path):
    X, yr, y3 = _data()
    model, target, kw = _CASES[case]()
    y = yr if target == "yr" else y3
    straight = model.fit(X, y, epochs=4, seed=7, **kw)
    ref = _words(straight)

    first, _, _ = _CASES[case]()
    first.fit(X, y, epochs=2, seed=7, **kw)
    path = tmp_path / f"{case}.ckpt"
    first.save_checkpoint(path)
    again = ml.Sequential.from_checkpoint(path)
    again.fit(X, y, epochs=2, resume=True)
    assert _words(again) == ref

    # the same continuation without the file
    inmem, _, _ = _CASES[case]()
    inmem.fit(X, y, epochs=2, seed=7, **kw)
    inmem.fit(X, y, epochs=2, resume=True)
    assert _words(inmem) == ref
    np.testing.assert_array_equal(again.predict(X), straight.predict(X))


def test_checkpoint_bytes_are_the_same_for_the_same_state(tmp_path):
    X, yr, _ = _data()
    paths = []
    for i in range(2):
        m, _, kw = _CASES["adam_step_schedule"]()
        m.fit(X, yr, epochs=2, seed=3, **kw)
        paths.append(tmp_path / f"run{i}.ckpt")
        m.save_checkpoint(paths[-1])
    assert paths[0].read_bytes() == paths[1].read_bytes()
    # load then save again: the same file
    back = tmp_path / "back.ckpt"
    ml.Sequential.from_checkpoint(paths[0]).save_checkpoint(back)
    assert back.read_bytes() == paths[0].read_bytes()


def test_early_stopping_state_resumes(tmp_path):
    X, yr, _ = _data()

    def build():
        return (ml.Sequential().linear(4).relu().linear(1)
                .compile(loss="mse", optimizer="adam", lr=0.05, tol=10.0, n_iter_no_change=2))

    straight = build().fit(X, yr, epochs=6, seed=1)
    part = build().fit(X, yr, epochs=1, seed=1)
    part.save_checkpoint(tmp_path / "es.ckpt")
    rest = ml.Sequential.from_checkpoint(tmp_path / "es.ckpt").fit(X, yr, epochs=5, resume=True)
    assert rest.n_iter_ == straight.n_iter_
    assert _h(rest.params_) == _h(straight.params_)


def test_export_predicts_the_same(tmp_path):
    X, _, y3 = _data()
    c, _, kw = _CASES["nesterov_classifier"]()
    c.fit(X, y3, epochs=3, seed=0, **kw)
    path = tmp_path / "clf.ckpt"
    c.export(path)
    served = ml.Sequential.load(path)
    np.testing.assert_array_equal(served.predict(X), c.predict(X))
    np.testing.assert_array_equal(served.predict_proba(X), c.predict_proba(X))
    assert served.classes_.dtype == c.classes_.dtype
    with pytest.raises(ValueError):
        served.fit(X, y3, epochs=1, resume=True)
    with pytest.raises(ValueError):
        served.save_checkpoint(tmp_path / "no_state.ckpt")
    # a fresh fit of an exported model is allowed
    served.fit(X, y3, epochs=1, seed=0)


def test_refuses_damaged_and_foreign_files(tmp_path):
    X, yr, _ = _data()
    m, _, kw = _CASES["adam_step_schedule"]()
    m.fit(X, yr, epochs=1, seed=0, **kw)
    good = tmp_path / "good.ckpt"
    m.save_checkpoint(good)
    raw = bytearray(good.read_bytes())
    raw[len(raw) // 2] ^= 0x01
    bad = tmp_path / "bad.ckpt"
    bad.write_bytes(bytes(raw))
    with pytest.raises(Exception, match="CKPT_REFUSE"):
        ml.Sequential.from_checkpoint(bad)
    short = tmp_path / "short.ckpt"
    short.write_bytes(good.read_bytes()[:-5])
    with pytest.raises(Exception, match="CKPT_REFUSE"):
        ml.Sequential.from_checkpoint(short)
    # a resume under another optimizer is refused
    other = ml.Sequential.from_checkpoint(good).compile(loss="mse", optimizer="sgd", lr=0.01)
    with pytest.raises(Exception, match="resume"):
        other.fit(X, yr, epochs=1, resume=True)


# ---------------------------------------------------------------- operation session
def _has_ops_state():
    try:
        return hasattr(_backend.binding("_mojolearn_x_sequence"), "ops_save_state")
    except Exception:
        return False


@pytest.mark.skipif(not _has_ops_state(), reason="the sequence binding has no ops_save_state entry")
def test_session_parameters_and_counters_round_trip(tmp_path):
    from mojolearn import ops
    rng = np.random.default_rng(0)
    w0 = rng.standard_normal((4, 3), dtype=np.float32)
    b0 = rng.standard_normal((1, 3), dtype=np.float32)
    path = tmp_path / "session.ckpt"
    with ops.Session(record=True) as s:
        w = s.tensor(w0, requires_grad=True)
        b = s.tensor(b0, requires_grad=True)
        s.save_state(path, [w, b], seed=2 ** 40 + 5, epoch=3, step=17)
    with ops.Session(record=True) as s:
        st = s.load_state(path)
        assert (st["seed"], st["epoch"], st["step"]) == (2 ** 40 + 5, 3, 17)
        w, b = st["params"]
        np.testing.assert_array_equal(s.numpy(w), w0)
        np.testing.assert_array_equal(s.numpy(b), b0)
        # trainable after loading
        x = s.tensor(rng.standard_normal((5, 4), dtype=np.float32))
        loss = s.mean(s.linear(x, w, b))
        s.backward(loss)
        assert s.grad(w).shape == (4, 3)


def _session_has_optimizer():
    from mojolearn import ops
    return hasattr(ops.Session, "optimizer") and hasattr(ops.Session, "step")


@pytest.mark.skipif(not (_has_ops_state() and _session_has_optimizer()),
                    reason="Session has no optimizer / step (lane ag-session-optim)")
def test_session_optimizer_resume_word_for_word(tmp_path):
    from mojolearn import ops
    rng = np.random.default_rng(1)
    X = rng.standard_normal((32, 4), dtype=np.float32)
    Y = rng.standard_normal((32, 2), dtype=np.float32)
    W0 = rng.standard_normal((4, 2), dtype=np.float32)

    def train(s, w, opt, steps):
        x = s.tensor(X)
        y = s.tensor(Y)
        for _ in range(steps):  # the user's training loop
            s.reset()
            s.backward(s.mse(s.matmul(x, w), y))
            s.step(opt, [w])

    with ops.Session(record=True) as s:
        w = s.tensor(W0, requires_grad=True)
        opt = s.optimizer("adam", lr=0.01)
        train(s, w, opt, 6)
        straight = s.numpy(w)
    path = tmp_path / "opt.ckpt"
    with ops.Session(record=True) as s:
        w = s.tensor(W0, requires_grad=True)
        opt = s.optimizer("adam", lr=0.01)
        train(s, w, opt, 3)
        s.save_state(path, [w], optimizer=opt, step=3)
    with ops.Session(record=True) as s:
        opt = s.optimizer("adam", lr=0.01)
        st = s.load_state(path, optimizer=opt)
        (w,) = st["params"]
        train(s, w, opt, 3)
        np.testing.assert_array_equal(s.numpy(w), straight)


# ---------------------------------------------------------------- across columns
_CHILD = textwrap.dedent('''
    import hashlib, json, runpy, sys
    from pathlib import Path
    mode, path, expected_package, helper, installed = sys.argv[1:]
    installed = installed == "1"
    if not installed:
        # Preserve the parent's actual package, which may itself be installed.
        # Never derive the package under test from this test file's location.
        sys.path.insert(0, str(Path(expected_package).parent.parent))
    import numpy as np
    import mojolearn as ml
    if Path(ml.__file__).resolve() != Path(expected_package):
        raise RuntimeError("Resume subprocess imported a different mojolearn package")
    package_evidence = runpy.run_path(helper)["package_evidence"]
    package_evidence(ml, require_installed=installed)

    def h(a):
        return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]

    rng = np.random.default_rng(0)
    X = rng.standard_normal((120, 5)).astype(np.float32)
    y3 = (X[:, 0] > 0).astype(np.int64) + (X[:, 1] > 0.5)

    def build():
        return (ml.Sequential().linear(8).relu().linear(3)
                .compile(loss="cross_entropy", optimizer="adam", lr=0.02, schedule="step", step_size=1,
                         gamma=0.8))

    if mode == "save":
        straight = build().fit(X, y3, epochs=4, batch_size=25, seed=11)
        print("DIGEST straight", h(straight.params_), h(straight._opt_m), h(straight._opt_v))
        part = build().fit(X, y3, epochs=2, batch_size=25, seed=11)
        part.save_checkpoint(path + ".train")
        part.export(path + ".export")
        print("DIGEST predict", h(part.predict_proba(X)), h(part.predict(X)))
    else:
        back = ml.Sequential.from_checkpoint(path + ".train").fit(X, y3, epochs=2, resume=True)
        print("DIGEST straight", h(back.params_), h(back._opt_m), h(back._opt_v))
        served = ml.Sequential.load(path + ".export")
        print("DIGEST predict", h(served.predict_proba(X)), h(served.predict(X)))
    print("COLUMN", ml.vendor())
    print("PACKAGE_EVIDENCE", json.dumps(package_evidence(ml, require_installed=installed), sort_keys=True))
''')


def _run(column, mode, path):
    env = dict(os.environ)
    if column == "cpu":
        env["MOJOLEARN_VENDOR"] = "cpu"
    else:
        env["MOJOLEARN_VENDOR"] = ml.vendor()
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    r = subprocess.run([sys.executable, "-I", "-c", _CHILD, mode, str(path),
                        str(Path(ml.__file__).resolve()), str(_PACKAGE_EVIDENCE),
                        "1" if _REQUIRE_INSTALLED else "0"], env=env, capture_output=True,
                       text=True, timeout=900)
    if r.returncode != 0 and column == "cpu" and "No module named" in r.stderr and not _REQUIRE_INSTALLED:
        pytest.skip("no CPU host binding in this install")
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-4000:]
    assert f"COLUMN {env['MOJOLEARN_VENDOR']}" in r.stdout.splitlines()
    for line in r.stdout.splitlines():
        if line.startswith("PACKAGE_EVIDENCE "):
            print(line)
    return {line.split()[1]: tuple(line.split()[2:]) for line in r.stdout.splitlines()
            if line.startswith("DIGEST ")}


def _gpu_present():
    try:
        # This suite needs the sequence binding, not the unrelated core
        # binding queried by ml.vendor() in a focused source build.
        return _backend.binding("_mojolearn_x_sequence").x_sequence_vendor() in ("metal", "cuda", "hip")
    except Exception:
        return False


@pytest.mark.skipif(not _gpu_present(), reason="no GPU binding: nothing to compare the host column with")
@pytest.mark.parametrize("writer, reader", [("cpu", "gpu"), ("gpu", "cpu")])
def test_resume_and_export_across_columns(writer, reader, tmp_path):
    path = tmp_path / f"{writer}_to_{reader}"
    wrote = _run(writer, "save", path)
    read = _run(reader, "load", path)
    # resuming on the other column gives the words of training straight
    # through on the writing column, and the export predicts the same
    assert read["straight"] == wrote["straight"]
    assert read["predict"] == wrote["predict"]
