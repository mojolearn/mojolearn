# SPDX-License-Identifier: Apache-2.0
"""Release-gate coverage of the byte-LM parallel trainers and their fold
helpers changed since 0.8.55 (docs/plans/TEST_COVERAGE_20261021.md).

Real path: the split step (shard_gradient + ordered_fold + apply_gradient)
and the device fold (fold_reset, shard_gradient_fold, fold_add, fold_export)
must commit the same bits as one train_step; the pooled and offloaded
schedules train, report ownership and replay from a checkpoint. Faults: a
closed trainer, wrong dtypes and shapes, and bad checkpoints are refused by
the native boundary with their message. One device, tiny model, no timing."""
import copy

import numpy as np
import pytest

from mojolearn import (OffloadedByteLanguageModelTrainer, ParallelByteLanguageModelTrainer,
                       ParallelNeuralTrainer, PooledByteLanguageModelTrainer, _backend)
from mojolearn import cross_vendor as cv
from mojolearn.parallel_training import ordered_sum_gradients
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_parallel_byte_native import native_failure, state_fixture

FIELDS = ("parameters", "m", "v", "flags")
SHARDS = [np.array([[0, 1, 2, 3, 4]], np.int32), np.array([[4, 3, 2, 1, 0]], np.int32)]
CLOSED = "parallel trainer is closed or lost"

gpu_only = pytest.mark.skipif(_backend.vendor() == "cpu",
                              reason="the cooperative byte-LM session requires a GPU")


def _state():
    state, shape = state_fixture()
    state["parameters"][:] = (np.random.default_rng(0).standard_normal(shape.n_total)
                              .astype(np.float32) * np.float32(0.1))
    return state, shape


def _make(cls=ParallelByteLanguageModelTrainer, **kwargs):
    owner = object.__new__(cls)
    audit_native_call(owner.__init__, copy.deepcopy(_state()[0]), **kwargs).assert_native()
    return owner


def _same_state(a, b):
    sa, sb = a.state_dict(), b.state_dict()
    assert sa["completed_steps"] == sb["completed_steps"]
    for key in FIELDS:
        assert np.asarray(sa[key]).tobytes() == np.asarray(sb[key]).tobytes(), key


@gpu_only
@pytest.mark.parametrize("form", ["bytes", "float32"])
def test_split_step_commits_the_bits_of_train_step(form):
    whole = _make(logical_shards=2, pool_optimizer=False)
    split = _make(logical_shards=2, pool_optimizer=False)
    try:
        losses = whole.train_step(SHARDS)["losses"]
        parts = [audit_native_call(split.shard_gradient, s).assert_native() for s in SHARDS]
        assert [p[0] for p in parts] == list(losses)
        total = cv.ordered_fold([p[1] for p in parts])
        if form == "float32":
            total = np.frombuffer(total, np.float32)
        assert audit_native_call(split.apply_gradient, total).assert_native() == 1
        _same_state(whole, split)
    finally:
        whole.close()
        split.close()


@gpu_only
def test_device_fold_matches_the_host_ordered_fold():
    worker = _make(logical_shards=2, pool_optimizer=False)
    helper = _make(logical_shards=2, pool_optimizer=False)
    try:
        assert audit_native_call(type(worker).has_device_fold.fget, worker).assert_native() is True
        expected = cv.ordered_fold([helper.shard_gradient(s)[1] for s in SHARDS])
        audit_native_call(worker.fold_reset).assert_native()
        loss = audit_native_call(worker.shard_gradient_fold, SHARDS[0]).assert_native()
        assert loss == helper.shard_gradient(SHARDS[0])[0]
        audit_native_call(worker.fold_add, helper.shard_gradient(SHARDS[1])[1]).assert_native()
        assert audit_native_call(worker.fold_export).assert_native() == expected
        into = np.empty(_state()[1].n_total, np.float32)
        assert audit_native_call(worker.fold_export, into=into).assert_native() is into
        assert into.tobytes() == expected
        # a prefix continues the fold: prefix(shard 0) + shard 1 == the full fold
        audit_native_call(worker.fold_reset, helper.shard_gradient(SHARDS[0])[1]).assert_native()
        worker.shard_gradient_fold(SHARDS[1])
        assert worker.fold_export() == expected
    finally:
        worker.close()
        helper.close()


@gpu_only
def test_split_and_fold_refuse_poisoned_inputs_by_message():
    owner = _make(logical_shards=2, pool_optimizer=False)
    n = _state()[1].n_total
    try:
        bad_ids = np.zeros((1, 5), np.float32)
        assert "float32 or int32" in native_failure(owner.shard_gradient, TypeError, bad_ids)
        assert "float32 or int32" in native_failure(owner.shard_gradient_fold, TypeError, bad_ids)
        assert "wrong shape" in native_failure(owner.apply_gradient, ValueError, np.zeros(3, np.float32))
        assert "wrong shape" in native_failure(owner.fold_add, ValueError, np.zeros(3, np.float32))
        assert "multiple of item size" in native_failure(owner.fold_reset, ValueError, b"abc")
        assert "exact contiguous one-dimensional" in native_failure(
            owner.fold_export, ValueError, into=np.empty(3, np.float32))
        assert "finite" in native_failure(owner.set_lr, ValueError, float("nan"))
        assert "into must be a dict" in native_failure(
            owner.export_raw, ValueError, into={"parameters": np.zeros(n, np.float32)})
        assert "invalid replica index" in native_failure(owner.state_dict, ValueError, rank=9)
    finally:
        owner.close()
    for operation, args in ((owner.train_step, (SHARDS,)), (owner.state_dict, ()),
                            (owner.export_gradients, ()), (owner.checkpoint, ()),
                            (owner.shard_gradient, (SHARDS[0],)), (owner.fold_export, ()),
                            (type(owner).has_device_fold.fget, (owner,))):
        assert CLOSED in native_failure(operation, RuntimeError, *args)


@gpu_only
def test_context_manager_closes_and_bad_checkpoint_is_refused():
    with _make(logical_shards=2) as owner:
        assert type(owner) is ParallelByteLanguageModelTrainer
        assert len(owner.train_step(SHARDS)["losses"]) == 2
    assert CLOSED in native_failure(owner.train_step, RuntimeError, SHARDS)
    for cls in (ParallelByteLanguageModelTrainer, PooledByteLanguageModelTrainer,
                OffloadedByteLanguageModelTrainer):
        assert "unsupported parallel checkpoint contract" in native_failure(
            cls.from_checkpoint, ValueError, {"schema": "not-a-checkpoint"})


@gpu_only
@pytest.mark.parametrize("cls", [PooledByteLanguageModelTrainer, OffloadedByteLanguageModelTrainer])
def test_pooled_and_offloaded_train_and_replay_from_checkpoint(cls):
    reference = _make(logical_shards=2)
    owner = _make(cls, logical_shards=2)
    replay = None
    try:
        expected = reference.train_step(SHARDS)["losses"]
        assert audit_native_call(owner.train_step, SHARDS).assert_native()["losses"] == expected
        _same_state(reference, owner)
        if cls is PooledByteLanguageModelTrainer:
            rows = audit_native_call(owner.model_ownership).assert_native()
            assert sum(row["count"] for row in rows) == _state()[1].n_total
            assert owner.optimizer_ownership() == rows
        else:
            with pytest.raises(NotImplementedError, match="offloaded state lives on the host"):
                owner.optimizer_ownership()
        checkpoint = owner.checkpoint()
        replay = audit_native_call(cls.from_checkpoint, checkpoint).assert_native()
        assert type(replay) is cls
        reference.train_step(SHARDS)
        replay.train_step(SHARDS)
        _same_state(reference, replay)
    finally:
        reference.close()
        owner.close()
        if replay is not None:
            replay.close()


@pytest.mark.parametrize("cls", [PooledByteLanguageModelTrainer, OffloadedByteLanguageModelTrainer])
def test_pooled_and_offloaded_admission_refusals(cls):
    state = copy.deepcopy(_state()[0])
    owner = object.__new__(cls)
    assert "logical_shards outside admitted range" in native_failure(
        owner.__init__, ValueError, state, logical_shards=0)
    if cls is OffloadedByteLanguageModelTrainer and _backend.vendor() != "cpu":
        message = native_failure(object.__new__(cls).__init__, ValueError, state, devices=(0, 1))
        assert "exactly one device" in message


def test_ordered_sum_gradients_and_fold_pair_refuse_bad_parts():
    assert "at least one shard" in native_failure(ordered_sum_gradients, ValueError, [])
    assert "wrong shape" in native_failure(
        ordered_sum_gradients, ValueError, [[np.zeros(3, np.float32)], [np.zeros(4, np.float32)]])
    assert "differ in length" in native_failure(cv.fold_pair, ValueError, b"\0" * 4, b"\0" * 8)
    assert "Python buffer hooks are not accepted" in native_failure(cv.fold_pair, ValueError, "ab", b"\0" * 4)


def test_coordinator_drain_events_is_empty_before_any_step():
    owner = object.__new__(cv.Coordinator)
    audit_native_call(owner.__init__, port=0, workers=1, logical_shards=1, steps=0).assert_native()
    assert audit_native_call(owner.drain_events).assert_native() == []


def test_parallel_neural_trainer_context_manager_closes():
    if _backend.default_mode() != "identical":
        pytest.skip("parallel neural training retains its identical-only contract")
    from mojolearn.tests.test_parallel_mlp_native import model_fixture
    x = np.zeros((3, 8), np.float32)
    x[:, 0] = [1.0, -0.25, 0.5]
    y = np.array([0, 1, 2], np.int64)
    with ParallelNeuralTrainer(model_fixture(), devices=(0,), logical_shards=1) as owner:
        assert type(owner) is ParallelNeuralTrainer
        assert len(owner.train_step([(x, y)])["losses"]) == 1
    with pytest.raises(Exception):
        owner.train_step([(x, y)])
