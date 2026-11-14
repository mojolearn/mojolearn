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


# test_train_steps_equals_single_steps and
# test_train_steps_on_an_open_session_runs_the_multi_step_operation moved to
# checks/native_tests/byte_lm_trainer_session_cases.mojo units 0-3 (lane
# tmig-lm-1): the loss words, completed counts, state and gradient bytes of
# k steps in one call against k single steps, through the session entry.


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
