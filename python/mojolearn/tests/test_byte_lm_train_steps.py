# SPDX-License-Identifier: Apache-2.0
"""`train_steps` and the ragged greedy pick against the per-step entries.

Plan docs/plans/NATIVE_ENTRY_POINTS_AFTER_0866.md section 2C, "Proof": run
steps through the session entry and through the old per-step entry and
compare the loss words. Runs on whatever byte LM binding the install has
(GPU, or the host column on a CPU-only install); skipped without one.
"""
import numpy as np
import pytest

import mojolearn

SCHEDULE = {"corpus": "train-steps-test"}


def _trainer(seed=0):
    shape = mojolearn.LanguageModelConfig()
    rng = np.random.default_rng(seed)
    weights = (rng.standard_normal(shape.n_total) * 0.02).astype(np.float32)
    try:
        return shape, mojolearn.LanguageModelTrainer(weights, data_schedule=SCHEDULE)
    except (ImportError, RuntimeError) as error:
        pytest.skip(f"no byte LM binding: {error}")


def _batches(shape, n_steps, seed=1):
    rng = np.random.default_rng(seed)
    return rng.integers(0, shape.vocab_size, (n_steps, shape.batch, shape.length + 1), dtype=np.int32)


def _bits(values):
    return np.asarray(values, dtype=np.float32).view(np.uint32)


def _state_bytes(trainer):
    state = trainer.export_state()
    return [np.asarray(state[key]).tobytes() for key in ("parameters", "m", "v", "flags")]


@pytest.mark.parametrize("n_steps", [1, 2, 4])
def test_train_steps_equals_single_steps(n_steps):
    shape, many = _trainer()
    _, single = _trainer()
    ids = _batches(shape, n_steps)
    result = many.train_steps(ids)
    losses = [single.train_step(np.ascontiguousarray(ids[k]))["loss"] for k in range(n_steps)]
    assert tuple(result["losses"].shape) == (n_steps,)
    assert np.array_equal(_bits(result["losses"]), _bits(losses))
    assert _bits([result["loss"]])[0] == _bits([losses[-1]])[0]
    assert result["completed_steps"] == n_steps == result["step"] == result["next_batch_index"]
    assert _state_bytes(many) == _state_bytes(single)


def test_train_steps_on_an_open_session_runs_the_multi_step_operation():
    # The first call opens the session; the second runs every step in the
    # multi-step native operation. Both trainers end on the same bits.
    shape, many = _trainer()
    _, single = _trainer()
    first = _batches(shape, 1, seed=2)
    rest = _batches(shape, 3, seed=3)
    many.train_step(np.ascontiguousarray(first[0]))
    single.train_step(np.ascontiguousarray(first[0]))
    result = many.train_steps(rest)
    losses = [single.train_step(np.ascontiguousarray(rest[k]))["loss"] for k in range(3)]
    assert np.array_equal(_bits(result["losses"]), _bits(losses))
    assert result["completed_steps"] == 4
    assert _state_bytes(many) == _state_bytes(single)
    # The gradient of the last step stays exportable, as after train_step.
    assert np.array_equal(np.asarray(many.export_gradients()["flat_gradients"]),
                          np.asarray(single.export_gradients()["flat_gradients"]))


def test_train_steps_refuses_a_wrong_shape_and_out_of_vocabulary_ids():
    shape, trainer = _trainer()
    with pytest.raises(ValueError):
        trainer.train_steps(np.zeros((2, shape.batch, shape.length), dtype=np.int32))
    with pytest.raises(ValueError):
        trainer.train_steps(np.zeros((0, shape.batch, shape.length + 1), dtype=np.int32))
    bad = _batches(shape, 2)
    bad[1, 0, 0] = shape.vocab_size
    with pytest.raises(ValueError):
        trainer.train_steps(bad)
    with pytest.raises(TypeError):
        trainer.train_steps(np.zeros((2, shape.batch, shape.length + 1), dtype=np.int64))


def test_ragged_next_bytes_is_the_logits_argmax_at_each_last_real_position():
    shape, trainer = _trainer()
    trainer.train_step(np.ascontiguousarray(_batches(shape, 1)[0]))
    batch, length = 3, 9
    rng = np.random.default_rng(4)
    ids = rng.integers(0, shape.vocab_size, (batch, length), dtype=np.int32)
    lengths = [9, 1, 5]
    picked = trainer.next_bytes(ids, lengths=lengths)
    logits = np.asarray(trainer.logits(ids, lengths=lengths))
    expected = [int(np.argmax(logits[row, lengths[row] - 1])) for row in range(batch)]
    assert list(picked) == expected
