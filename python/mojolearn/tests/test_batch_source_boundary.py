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


def _shards(root, first_line="mojolearn-batch-shards v1", lengths=(1, 3, 2, 4), scale=1.0, seal=None):
    """One shard under root. Sealed (BatchSource.seal) unless the fixture is one a test expects to be refused at
    open, or seal=False. `scale` changes the step values and nothing else (same-shaped replacement data)."""
    os.makedirs(root, exist_ok=True)
    steps = scale * np.arange(sum(lengths) * F, dtype=np.float32).reshape(-1, F) / 10.0
    np.save(os.path.join(root, "a.steps.npy"), steps)
    np.save(os.path.join(root, "a.offsets.npy"), np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64))
    with open(os.path.join(root, "index.txt"), "w", encoding="ascii") as fh:
        fh.write(first_line + "\nshard a\n")
    if seal is None:
        seal = first_line == "mojolearn-batch-shards v1" and min(lengths) > 0
    if seal:
        assert ops.BatchSource.seal(root) == 1
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


def test_each_batch_trains_at_its_own_padded_length(tmp_path):
    """A program captured at one length replays on batches of other lengths: its nodes run again on each batch's
    shape (a reshape with -1 is resolved from the batch in front of it). Records of lengths 1, 3, 2, 4 in order,
    two a batch: the first batch is padded to 3 steps, the second to 4, the capture was made at 4."""
    lengths = (1, 3, 2, 4)
    steps = np.arange(sum(lengths) * F, dtype=np.float32).reshape(-1, F) / 10.0
    src = ops.BatchSource(_shards(str(tmp_path / "d"), lengths=lengths), 2, shuffle=False)
    with ops.Session() as s:
        prog, w = _program(s)
        opt = s.optimizer("sgd", lr=0.1)
        # loss = sum(steps @ w) + sum(lengths): the gradient of w is the column sums of the batch's real steps
        prog.fit(source=src, optimizer=opt, parameters=[w], loss="output", steps=1)
        first = np.asarray(s.numpy(w), np.float64).reshape(-1)
        np.testing.assert_allclose(first, 0.25 - 0.1 * steps[0:4].astype(np.float64).sum(axis=0), rtol=1e-5)
        prog.fit(source=src, optimizer=opt, parameters=[w], loss="output", steps=1)
        second = np.asarray(s.numpy(w), np.float64).reshape(-1)
        np.testing.assert_allclose(second, first - 0.1 * steps[4:10].astype(np.float64).sum(axis=0), rtol=1e-5)


def test_a_program_fixed_to_one_length_is_refused_on_another(tmp_path):
    """A node that names the captured length (a reshape to (8, F) of a (2, 4, F) input) cannot take a batch
    padded to 3 steps: the fit refuses, it never reads past the batch."""
    src = ops.BatchSource(_shards(str(tmp_path / "d"), lengths=(1, 3, 2, 4)), 2, shuffle=False)
    with ops.Session() as s:
        w = s.tensor(np.full((F, 1), 0.25, np.float32))
        xs = s.tensor(np.zeros((2, 4, F), np.float32))
        xl = s.tensor(np.zeros((2, 1), np.float32))
        loss = s.add(s.sum(s.matmul(s.reshape(xs, (8, F)), w)), s.sum(xl))
        fixed = s.capture(inputs=[xs, xl], outputs=[loss])
        opt = s.optimizer("sgd", lr=0.1)
        with pytest.raises(Exception):
            fixed.fit(source=src, optimizer=opt, parameters=[w], loss="output", steps=1)


def test_one_step_fits_keep_one_opened_source_and_a_loaded_position_rebuilds_it(tmp_path):
    # lane fr2-followups: the handle keeps its opened source on a native owner across fit calls (no reopen of the
    # index and the epoch plan per step); a position set from outside a fit drops it and the next fit rebuilds it
    src = ops.BatchSource(_shards(str(tmp_path)), 2)
    assert src._owner is None
    with ops.Session() as s:
        prog, w = _program(s)
        opt = s.optimizer("adam", lr=0.01)
        prog.fit(source=src, optimizer=opt, parameters=[w], loss="output", steps=1)
        owner = src._owner
        assert owner is not None
        for _ in range(99):
            prog.fit(source=src, optimizer=opt, parameters=[w], loss="output", steps=1)
            assert src._owner is owner
        state = src.state_dict()
        assert (state["epoch"], state["consumed"]) == (49, 2)
        src.load_state_dict(dict(state, epoch=3, consumed=1))
        assert src._owner is None
        before = src.next_records()
        prog.fit(source=src, optimizer=opt, parameters=[w], loss="output", steps=1)
        assert src._owner is not None and src._owner is not owner
        assert (src.state_dict()["epoch"], src.state_dict()["consumed"]) == (3, 2)
        assert len(before) == 2


def test_seal_records_the_content_digest_in_the_index(tmp_path):
    root = _shards(str(tmp_path), seal=False)
    assert ops.BatchSource(root, 2).sealed is False
    assert ops.BatchSource.seal(root) == 1
    with open(os.path.join(root, "index.txt"), encoding="ascii") as fh:
        lines = fh.read().splitlines()
    assert lines[0] == "mojolearn-batch-shards v1"
    stem, word = lines[1].rsplit(" ", 1)
    assert stem == "shard a" and word.startswith("sha256:") and len(word) == 7 + 64
    src = ops.BatchSource(root, 2)
    assert src.sealed is True and src.state_dict()["sealed"] is True
    assert src.state_dict()["inputs"] == ("steps", "lengths") and src.state_dict()["target"] is None
    # sealing again gives the same words for the same bytes
    assert ops.BatchSource.seal(root) == 1
    with open(os.path.join(root, "index.txt"), encoding="ascii") as fh:
        assert fh.read().splitlines() == lines


def test_a_resume_needs_sealed_shards(tmp_path):
    """Without the content digest a position could be applied to other data of the same shape: every resume door
    refuses an unsealed source by name."""
    src = ops.BatchSource(_shards(str(tmp_path / "d"), seal=False), 2)
    with ops.Session() as s:
        prog, w = _program(s)
        opt = s.optimizer("adam", lr=0.01)
        prog.fit(source=src, optimizer=opt, parameters=[w], loss="output", steps=1)
        with pytest.raises(ValueError, match=r"records no content digest.*BatchSource\.seal"):
            prog.save_training(str(tmp_path / "run.ckpt"), optimizer=opt, source=src)
    with pytest.raises(ValueError, match="records no content digest"):
        src.load_state_dict(src.state_dict())


def test_same_shaped_replacement_data_is_refused_by_name(tmp_path):
    """Other step values under the same record lengths, widths and counts: the checkpoint and a saved position are
    bound to the sealed content, so neither resumes on it."""
    trained = ops.BatchSource(_shards(str(tmp_path / "d")), 2)
    replaced_root = _shards(str(tmp_path / "e"), scale=2.0)
    path = str(tmp_path / "run.ckpt")
    with ops.Session() as s:
        prog, w = _program(s)
        opt = s.optimizer("adam", lr=0.01)
        prog.fit(source=trained, optimizer=opt, parameters=[w], loss="output", steps=1)
        prog.save_training(path, optimizer=opt, source=trained)
    state = trained.state_dict()
    with ops.Session() as d:
        replaced = ops.BatchSource(replaced_root, 2)
        assert (replaced.n_records, replaced.n_batches, replaced.max_length) == (
            trained.n_records, trained.n_batches, trained.max_length)
        with pytest.raises(ValueError, match="not the data the checkpoint was saved with"):
            d.load_training(path, source=replaced)
        assert replaced.state_dict()["consumed"] == 0
        with pytest.raises(ValueError, match="not the data"):
            replaced.load_state_dict(state)
        # the trained shards still resume
        again = ops.BatchSource(str(tmp_path / "d"), 2)
        d.load_training(path, source=again)
        assert again.state_dict()["consumed"] == 1


def test_a_changed_field_mapping_is_refused_by_name(tmp_path):
    root = _shards(str(tmp_path / "d"))
    src = ops.BatchSource(root, 2)
    path = str(tmp_path / "run.ckpt")
    with ops.Session() as s:
        prog, w = _program(s)
        opt = s.optimizer("adam", lr=0.01)
        prog.fit(source=src, optimizer=opt, parameters=[w], loss="output", steps=1)
        prog.save_training(path, optimizer=opt, source=src)
    with ops.Session() as d:
        masked = ops.BatchSource(root, 2, inputs=("steps", "mask"))
        with pytest.raises(ValueError, match=r"inputs \(steps, mask\) are not the fields .* \(steps, lengths\)"):
            d.load_training(path, source=masked)
        swapped = ops.BatchSource(root, 2, inputs=("lengths", "steps"))
        with pytest.raises(ValueError, match="are not the fields"):
            d.load_training(path, source=swapped)
    with pytest.raises(ValueError, match="are not the fields"):
        ops.BatchSource(root, 2, inputs=("steps", "mask")).load_state_dict(src.state_dict())


def test_a_save_replaces_the_checkpoint_and_never_writes_into_the_previous_one(tmp_path):
    """save_training writes a new file and renames it over the name. A hard link taken on the first checkpoint
    still holds the first checkpoint after a second save to the same name, so a save that is interrupted at any
    point leaves the previous checkpoint loadable; no temporary is left beside it."""
    root = _shards(str(tmp_path / "d"))
    src = ops.BatchSource(root, 2)
    path = str(tmp_path / "run.ckpt")
    previous = str(tmp_path / "previous.ckpt")
    with ops.Session() as s:
        prog, w = _program(s)
        opt = s.optimizer("adam", lr=0.01)
        prog.fit(source=src, optimizer=opt, parameters=[w], loss="output", steps=1)
        prog.save_training(path, optimizer=opt, source=src)
        os.link(path, previous)
        with open(previous, "rb") as fh:
            first = fh.read()
        prog.fit(source=src, optimizer=opt, parameters=[w], loss="output", steps=1)
        prog.save_training(path, optimizer=opt, source=src)
    with open(previous, "rb") as fh:
        assert fh.read() == first
    with open(path, "rb") as fh:
        assert fh.read() != first
    assert sorted(os.listdir(tmp_path)) == ["d", "previous.ckpt", "run.ckpt"]
    with ops.Session() as d:
        fresh = ops.BatchSource(root, 2)
        d.load_training(previous, source=fresh)
        assert fresh.state_dict()["consumed"] == 1
        d.load_training(path, source=fresh)
        assert fresh.state_dict()["consumed"] == 2
