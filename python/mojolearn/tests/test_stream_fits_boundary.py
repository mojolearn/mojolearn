# SPDX-License-Identifier: Apache-2.0
"""Public boundary of the streamed fits (lane stream-fits, 2026-11-25): Sequential.fit(source=, steps=) and the
recurrent estimators' fit(source=) / partial_fit(source=, steps=). The refusals in the user's words, the returned
fields, and in-process resumes equal to the uninterrupted run. The fresh-process resumes are the verify cases
sequential-stream-resume and gru-stream-resume (lane ops-composed-cases); the numerics the native check
checks/native_tests/stream_fits.mojo."""
import os
import pickle

import numpy as np
import pytest

import mojolearn
from mojolearn import ops

F = 2
LENGTHS = ((1, 3, 2, 4), (2, 1, 3))
LONGEST = 4


def _shards(root, targets="dense", width=1, seal=True, scale=1.0):
    """Two shards of 7 records (lengths 1 to 4, F = 2) with a target row each: `dense` (width floats), `classes`
    (one class index in [0, 3)) or `steps` (width floats a step, padded to the longest record)."""
    os.makedirs(root, exist_ok=True)
    lines = ["mojolearn-batch-shards v1"]
    for k, lengths in enumerate(LENGTHS):
        n, total = len(lengths), sum(lengths)
        steps = scale * (np.arange(total * F, dtype=np.float32).reshape(-1, F) % 7 - 3) / 4.0 + k
        np.save(os.path.join(root, f"p{k}.steps.npy"), np.ascontiguousarray(steps, np.float32))
        np.save(os.path.join(root, f"p{k}.offsets.npy"), np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64))
        if targets == "dense":
            t = (np.arange(n * width, dtype=np.float32).reshape(n, width) % 5) / 5.0
        elif targets == "classes":
            t = (np.arange(n, dtype=np.float32).reshape(n, 1) + k) % 3
        else:
            t = (np.arange(n * width * LONGEST, dtype=np.float32).reshape(n, -1) % 5) / 5.0
        np.save(os.path.join(root, f"p{k}.targets.npy"), np.ascontiguousarray(t, np.float32))
        lines.append(f"shard p{k}")
    with open(os.path.join(root, "index.txt"), "w", encoding="ascii") as fh:
        fh.write("\n".join(lines) + "\n")
    if seal:
        assert ops.BatchSource.seal(root) == len(LENGTHS)
    return root


def _source(root, **kw):
    kw = dict(dict(inputs=("steps",), target="targets", shuffle=True, seed=5, bucket_batches=2), **kw)
    return ops.BatchSource(root, 2, **kw)


def _regressor(width=1):
    return (mojolearn.Sequential().gru(3, output="final").flatten().linear(width)
            .compile(loss="mse", optimizer="adam", lr=0.05))


def test_sequential_streamed_fit_sets_the_fitted_fields(tmp_path):
    src = _source(_shards(str(tmp_path)))
    m = _regressor().fit(source=src, epochs=2)
    assert m.n_iter_ == 2 and len(m.loss_curve_) == 2 and np.all(np.isfinite(m.loss_curve_))
    assert m.n_features_in_ == LONGEST * F and m.optimizer_w_ is None
    assert src.state_dict()["epoch"] == 1 and src.state_dict()["consumed"] == src.n_batches
    assert m.predict(np.zeros((3, LONGEST, F), np.float32), lengths=np.array([1, 2, 4])).shape == (3,)


def test_sequential_streamed_resume_equals_the_uninterrupted_fit(tmp_path):
    root = _shards(str(tmp_path / "s"))
    whole = _regressor().fit(source=_source(root), epochs=2)
    part = _regressor().fit(source=_source(root), epochs=2, steps=3)
    path = str(tmp_path / "m.ckpt")
    part.save_checkpoint(path)
    back = mojolearn.Sequential.from_checkpoint(path)
    back.fit(source=_source(root), epochs=2, steps=5, resume=True)
    for name in ("params_", "optimizer_m_", "optimizer_v_"):
        assert np.array_equal(getattr(back, name), getattr(whole, name)), name


def test_sequential_streamed_classifier_reads_class_indices(tmp_path):
    src = _source(_shards(str(tmp_path), targets="classes"))
    m = (mojolearn.Sequential().gru(3, output="final").flatten().linear(3)
         .compile(loss="cross_entropy", optimizer="adam", lr=0.05).fit(source=src, epochs=1))
    assert list(m.classes_) == [0, 1, 2]


@pytest.mark.parametrize("kw, words", [
    (dict(batch_size=4), "no batch_size or shuffle"),
    (dict(validation_data=(np.zeros((2, 4, 2), np.float32), np.zeros(2, np.float32))), "no validation_data"),
    (dict(lengths=np.ones(7)), "no validation_data, restore_best_weights, lengths"),
])
def test_sequential_streamed_refusals(tmp_path, kw, words):
    src = _source(_shards(str(tmp_path)))
    with pytest.raises((TypeError, ValueError), match=words):
        _regressor().fit(source=src, **kw)


def test_sequential_source_with_arrays_or_other_fields_is_refused(tmp_path):
    root = _shards(str(tmp_path))
    with pytest.raises(TypeError, match="no X or y"):
        _regressor().fit(np.zeros((7, 4, 2), np.float32), np.zeros(7, np.float32), source=_source(root))
    with pytest.raises(ValueError, match="inputs=\\('steps',\\)"):
        _regressor().fit(source=_source(root, inputs=("steps", "lengths")))
    with pytest.raises(ValueError, match="outputs"):
        _regressor(width=2).fit(source=_source(root))


def test_sequential_resume_on_other_data_or_unsealed_shards_is_refused(tmp_path):
    root = _shards(str(tmp_path / "a"))
    m = _regressor().fit(source=_source(root), epochs=1, steps=2)
    twin = _shards(str(tmp_path / "b"), scale=2.0)
    with pytest.raises(ValueError, match="not the data the checkpoint was saved with"):
        m.fit(source=_source(twin), epochs=1, resume=True)
    loose = _shards(str(tmp_path / "c"), seal=False)
    with pytest.raises(ValueError, match="records no content digest"):
        m.fit(source=_source(loose), epochs=1, resume=True)


def _gru():
    return mojolearn.GRURegressor(hidden_size=4, output="sequence", learning_rate=0.05, random_state=0)


def test_recurrent_partial_fit_resume_after_pickling_equals_the_uninterrupted_run(tmp_path):
    root = _shards(str(tmp_path), targets="steps")
    whole = _gru().partial_fit(source=_source(root), steps=8)
    part = _gru().partial_fit(source=_source(root), steps=3)
    back = pickle.loads(pickle.dumps(part))
    back.partial_fit(source=_source(root), steps=5)
    assert np.array_equal(back.params_, whole.params_)
    assert np.array_equal(back._optimizer_state_, whole._optimizer_state_)
    assert len(whole.loss_curve_) == 8 and whole.n_iter_ == 8 and back.n_iter_ == 8


def test_recurrent_streamed_fit_and_refusals(tmp_path):
    root = _shards(str(tmp_path), targets="steps")
    est = mojolearn.GRURegressor(hidden_size=4, output="sequence", max_epochs=2).fit(source=_source(root))
    assert len(est.loss_curve_) == 2 * _source(root).n_batches and est.n_timesteps_ == LONGEST
    with pytest.raises(TypeError, match="no lengths, sample_weight"):
        _gru().fit(source=_source(root), sample_weight=np.ones(7))
    with pytest.raises(ValueError, match="one class index a record"):
        mojolearn.GRUClassifier(hidden_size=4).fit(source=_source(root))


def test_recurrent_streamed_classifier_classes_are_the_indices(tmp_path):
    root = _shards(str(tmp_path), targets="classes")
    est = mojolearn.GRUClassifier(hidden_size=4, max_epochs=1).fit(source=_source(root))
    assert list(est.classes_) == [0, 1, 2] and est.n_outputs_ == 3


def _classes_by_step(root):
    """The classes fixture with one class index a step (output='sequence'), padded to the longest record."""
    os.makedirs(root, exist_ok=True)
    lines = ["mojolearn-batch-shards v1"]
    for k, lengths in enumerate(LENGTHS):
        n, total = len(lengths), sum(lengths)
        steps = (np.arange(total * F, dtype=np.float32).reshape(-1, F) % 7 - 3) / 4.0 + k
        np.save(os.path.join(root, f"p{k}.steps.npy"), np.ascontiguousarray(steps, np.float32))
        np.save(os.path.join(root, f"p{k}.offsets.npy"), np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64))
        t = (np.arange(n * LONGEST, dtype=np.float32).reshape(n, LONGEST) * 5 + k) % 3
        np.save(os.path.join(root, f"p{k}.targets.npy"), np.ascontiguousarray(t, np.float32))
        lines.append(f"shard p{k}")
    with open(os.path.join(root, "index.txt"), "w", encoding="ascii") as fh:
        fh.write("\n".join(lines) + "\n")
    assert ops.BatchSource.seal(root) == len(LENGTHS)
    return root


def _tagger(**kw):
    return mojolearn.GRUClassifier(hidden_size=4, output="sequence", learning_rate=0.05, random_state=0,
                                   max_epochs=2, **kw)


def test_recurrent_streamed_class_weights_per_step(tmp_path):
    """class_weight as a dict weighs every valid step by its label; weights of 1 are the unweighted fit."""
    root = _classes_by_step(str(tmp_path))
    plain = _tagger().fit(source=_source(root))
    ones = _tagger(class_weight={0: 1.0, 1: 1.0, 2: 1.0}).fit(source=_source(root))
    assert np.array_equal(ones.params_, plain.params_)
    weighted = _tagger(class_weight={0: 1.0, 1: 3.0, 2: 0.5}).fit(source=_source(root))
    assert not np.array_equal(weighted.params_, plain.params_)
    assert np.all(np.isfinite(weighted.loss_curve_))
    with pytest.raises(ValueError, match="class_weight as a dict"):
        _tagger(class_weight="balanced").fit(source=_source(root))
    with pytest.raises(ValueError, match="class_weight is for classifiers"):
        mojolearn.GRURegressor(hidden_size=4, class_weight={0: 2.0}).fit(source=_source(_shards(str(tmp_path / "r"))))


def test_recurrent_streamed_validation_per_epoch(tmp_path):
    """validation_data arrays give one validation loss for each pass over the source and leave training as it is."""
    root = _classes_by_step(str(tmp_path))
    xv = (np.arange(3 * LONGEST * F, dtype=np.float32).reshape(3, LONGEST, F) % 5 - 2) / 3.0
    yv = (np.arange(3 * LONGEST).reshape(3, LONGEST) * 7) % 3
    lv = np.array([4, 2, 3])
    plain = _tagger().fit(source=_source(root))
    seen = _tagger().fit(source=_source(root), validation_data=(xv, yv, lv))
    assert np.array_equal(seen.params_, plain.params_)
    assert len(seen.validation_loss_) == 2 and np.all(np.isfinite(seen.validation_loss_))
    assert seen.best_epoch_ == 1 + int(np.argmin(seen.validation_loss_))
    with pytest.raises(ValueError, match="needs the validation sequences' lengths"):
        _tagger().fit(source=_source(root), validation_data=(xv, yv))
    with pytest.raises(ValueError, match="padded to the source's max_length"):
        _tagger().fit(source=_source(root), validation_data=(xv[:, :3], yv[:, :3], np.array([3, 2, 3])))
