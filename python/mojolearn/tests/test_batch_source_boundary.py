# SPDX-License-Identifier: Apache-2.0
"""Public boundary of ops.BatchSource, Program.fit(source=), Program.save_training and Session.load_training (lane
batch-source, 2026-11-25): the refusals in the user's words and the returned objects. The numerics (the plan, the
padding, a mid-epoch stop after the read-ahead and a resume equal to the uninterrupted run) are the native check
checks/native_tests/batch_source.mojo and the verify lane ops-batch-source-resume."""
import os

import numpy as np
import pytest

from mojolearn import ops

F = 2


def _shards(root, first_line="mojolearn-batch-shards v1", lengths=(1, 3, 2, 4)):
    os.makedirs(root, exist_ok=True)
    steps = np.arange(sum(lengths) * F, dtype=np.float32).reshape(-1, F) / 10.0
    np.save(os.path.join(root, "a.steps.npy"), steps)
    np.save(os.path.join(root, "a.offsets.npy"), np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64))
    with open(os.path.join(root, "index.txt"), "w", encoding="ascii") as fh:
        fh.write(first_line + "\nshard a\n")
    return root


def _program(s):
    w = s.tensor(np.full((F, 1), 0.25, np.float32))
    xs = s.tensor(np.zeros((2, 4, F), np.float32))
    xl = s.tensor(np.zeros((2, 1), np.float32))
    loss = s.add(s.sum(s.matmul(s.reshape(xs, (-1, F)), w)), s.sum(xl))
    return s.capture(inputs=[xs, xl], outputs=[loss]), w


def test_open_reports_the_stream(tmp_path):
    src = ops.BatchSource(_shards(str(tmp_path)), 2, bucket_batches=2, seed=3)
    assert (src.n_records, src.n_batches, src.n_features, src.max_length) == (4, 2, F, 4)
    state = src.state_dict()
    assert state["format"] == "mojolearn-batch-shards v1"
    assert (state["epoch"], state["consumed"]) == (0, 0)
    assert sorted(src.next_records()) == sorted(set(src.next_records()))


def test_bad_format_words_are_refused(tmp_path):
    with pytest.raises(ValueError, match="format words"):
        ops.BatchSource(_shards(str(tmp_path), first_line="mojolearn-batch-shards v9"), 2)


def test_bad_shard_line_and_unknown_field_are_refused(tmp_path):
    root = _shards(str(tmp_path))
    with open(os.path.join(root, "index.txt"), "w", encoding="ascii") as fh:
        fh.write("mojolearn-batch-shards v1\npart a\n")
    with pytest.raises(ValueError, match="shard <stem>"):
        ops.BatchSource(root, 2)
    with pytest.raises(ValueError, match="unknown field"):
        ops.BatchSource(_shards(str(tmp_path / "b")), 2, inputs=("steps", "tokens"))


def test_empty_record_is_refused(tmp_path):
    with pytest.raises(ValueError, match="no step"):
        ops.BatchSource(_shards(str(tmp_path), lengths=(1, 0, 2)), 2)


def test_fit_refuses_a_python_iterable(tmp_path):
    with ops.Session() as s:
        prog, w = _program(s)
        opt = s.optimizer("sgd", lr=0.1)
        batches = iter([(np.zeros((2, 4, F), np.float32), np.ones((2, 1), np.float32))])
        with pytest.raises(TypeError, match="never calls back into Python"):
            prog.fit(source=batches, optimizer=opt, parameters=[w], loss="output")

        class Loader(ops.BatchSource):
            pass

        loader = Loader.__new__(Loader)
        with pytest.raises(TypeError, match="mojolearn.ops.BatchSource"):
            prog.fit(source=loader, optimizer=opt, parameters=[w], loss="output")


def test_fit_with_a_source_takes_no_arrays_and_steps_needs_a_source(tmp_path):
    src = ops.BatchSource(_shards(str(tmp_path)), 2)
    with ops.Session() as s:
        prog, w = _program(s)
        opt = s.optimizer("sgd", lr=0.1)
        with pytest.raises(ValueError, match="takes no arrays"):
            prog.fit(np.zeros((2, 4, F), np.float32), source=src, optimizer=opt, parameters=[w], loss="output")
        with pytest.raises(ValueError, match="steps applies to a source"):
            prog.fit(np.zeros((2, 4, F), np.float32), np.zeros((2, 1), np.float32), optimizer=opt,
                     parameters=[w], loss="output", steps=1)
        with pytest.raises(ValueError, match="tol"):
            prog.fit(source=src, optimizer=opt, parameters=[w], loss="output", tol=1e-4)
        # the batches are the source's: a fit argument that says otherwise is refused by name, never ignored
        with pytest.raises(ValueError, match=r"omit batch_size or open the source with batch_size=64"):
            prog.fit(source=src, optimizer=opt, parameters=[w], loss="output", batch_size=64)
        with pytest.raises(ValueError, match="omit shuffle"):
            prog.fit(source=src, optimizer=opt, parameters=[w], loss="output", shuffle=False)
        with pytest.raises(ValueError, match="omit seed"):
            prog.fit(source=src, optimizer=opt, parameters=[w], loss="output", seed=7)
        assert src.state_dict()["consumed"] == 0
        # the source's own values are not a disagreement
        prog.fit(source=src, optimizer=opt, parameters=[w], loss="output", batch_size=2, steps=1)
        assert src.state_dict()["consumed"] == 1


def test_fit_moves_the_position_and_the_checkpoint_round_trip_returns_the_documented_objects(tmp_path):
    src = ops.BatchSource(_shards(str(tmp_path / "d")), 2)
    path = str(tmp_path / "run.ckpt")
    with ops.Session() as s:
        prog, w = _program(s)
        opt = s.optimizer("adam", lr=0.01)
        assert prog.fit(source=src, optimizer=opt, parameters=[w], loss="output", steps=1) is prog
        assert src.state_dict()["consumed"] == 1
        assert isinstance(prog.save_training(path, optimizer=opt, source=src), int)
    with ops.Session() as d:
        fresh = ops.BatchSource(str(tmp_path / "d"), 2)
        state = d.load_training(path, source=fresh)
        assert isinstance(state["program"], ops.Program)
        assert isinstance(state["optimizer"], int) and len(state["parameters"]) == 1
        assert fresh.state_dict()["consumed"] == 1
        other = ops.BatchSource(str(tmp_path / "d"), 2, seed=9)
        with pytest.raises(ValueError, match="not the one the checkpoint was saved"):
            d.load_training(path, source=other)
        # the refused load left nothing in the session: the next optimizer takes the next id
        assert other.state_dict()["consumed"] == 0
        assert d.optimizer("sgd", lr=0.1) == state["optimizer"] + 1


def test_training_checkpoint_carries_centered_rmsprop(tmp_path):
    """RMSprop(centered=True) keeps a third state buffer (the gradient average): the checkpoint carries it, so the
    step after a resume equals the uninterrupted one."""
    src = ops.BatchSource(_shards(str(tmp_path / "d")), 2, shuffle=False)
    path = str(tmp_path / "run.ckpt")
    with ops.Session() as s:
        prog, w = _program(s)
        opt = s.optimizer("rmsprop", lr=0.05, centered=True, momentum=0.5)
        prog.fit(source=src, optimizer=opt, parameters=[w], loss="output", steps=1)
        prog.save_training(path, optimizer=opt, source=src)
        prog.fit(source=src, optimizer=opt, parameters=[w], loss="output", steps=1)
        straight = np.asarray(s.numpy(w)).copy()
    with ops.Session() as d:
        fresh = ops.BatchSource(str(tmp_path / "d"), 2, shuffle=False)
        state = d.load_training(path, source=fresh)
        state["program"].fit(source=fresh, optimizer=state["optimizer"], parameters=state["parameters"],
                             loss="output", steps=1)
        resumed = np.asarray(d.numpy(state["parameters"][0]))
    assert resumed.tobytes() == straight.tobytes()


def test_load_state_dict_refuses_other_settings(tmp_path):
    src = ops.BatchSource(_shards(str(tmp_path)), 2, seed=1)
    state = src.state_dict()
    other = ops.BatchSource(str(tmp_path), 2, seed=2)
    with pytest.raises(ValueError, match="not the one"):
        other.load_state_dict(state)
    bad = dict(state, format="other")
    with pytest.raises(ValueError, match="format words"):
        src.load_state_dict(bad)
