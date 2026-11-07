# SPDX-License-Identifier: Apache-2.0
"""The parallel byte-LM doors read caller buffers in place (pass 3 rows N102,
N105, N111, N157): C-contiguous shards and gradients are pinned, other layouts
take the copy route, the corpus-fed step gathers rows natively, and the device
refuses non-finite gradients. Every route must commit the bits of the plain
`train_step` / split step. One device, tiny model, no timing."""
import array
import copy
import hashlib
import json

import numpy as np
import pytest

from mojolearn import ParallelByteLanguageModelTrainer, _backend, lm_corpus
from mojolearn import cross_vendor as cv
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.tests.test_parallel_byte_native import native_failure, state_fixture

FIELDS = ("parameters", "m", "v", "flags")
SHARDS = [np.array([[0, 1, 2, 3, 4]], np.int32), np.array([[4, 3, 2, 1, 0]], np.int32)]

gpu_only = pytest.mark.skipif(_backend.vendor() == "cpu",
                              reason="the cooperative byte-LM session requires a GPU")


def _state():
    state, shape = state_fixture()
    state["parameters"][:] = (np.random.default_rng(0).standard_normal(shape.n_total)
                              .astype(np.float32) * np.float32(0.1))
    return state, shape


def _make(**kwargs):
    owner = object.__new__(ParallelByteLanguageModelTrainer)
    audit_native_call(owner.__init__, copy.deepcopy(_state()[0]), **kwargs).assert_native()
    return owner


def _raw(owner):
    state = owner.state_dict()
    return state["completed_steps"], [np.asarray(state[key]).tobytes() for key in FIELDS]


def _tokens(tmp_path, batch=1, length=4, count=64):
    # Ids 0..4 only: the fixture model's vocabulary is 16.
    directory = tmp_path / "tokens"
    directory.mkdir(parents=True)
    payload = array.array("i", [i % 5 for i in range(count)]).tobytes()
    (directory / "tokens.i32").write_bytes(payload)
    manifest = dict(schema=lm_corpus.TOKENS_SCHEMA, source=dict(sha256="c" * 64),
                    vocabulary=dict(schema=lm_corpus.VOCABULARY_SCHEMA, sha256="v" * 64, n_vocab=16),
                    sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload), tokens=count,
                    train_range=[3, count - 2])
    (directory / "manifest.json").write_text(json.dumps(manifest) + "\n")
    return lm_corpus.TokenBatches(directory, batch, length)


@gpu_only
def test_strided_fortran_and_array_shards_commit_the_bits_of_contiguous_shards():
    reference = _make(logical_shards=2, pool_optimizer=False)
    other = _make(logical_shards=2, pool_optimizer=False)
    try:
        expected = reference.train_step(SHARDS)["losses"]
        wide = np.zeros((1, 10), np.int32)
        wide[:, ::2] = SHARDS[0]                        # strided: the copy route
        fortran = np.asfortranarray(SHARDS[1])          # one row: also C-contiguous
        got = audit_native_call(other.train_step, [wide[:, ::2], fortran]).assert_native()
        assert got["losses"] == expected
        assert _raw(other) == _raw(reference)
        # The caller's buffers are read, never written.
        assert wide[:, ::2].tolist() == SHARDS[0].tolist()
        assert fortran.tolist() == SHARDS[1].tolist()
        # A read-only buffer and a shared buffer for both shards pin as well.
        frozen = SHARDS[0].copy()
        frozen.setflags(write=False)
        reference.train_step([frozen, frozen])
        audit_native_call(other.train_step, (frozen, frozen)).assert_native()
        assert _raw(other) == _raw(reference)
    finally:
        reference.close()
        other.close()


@gpu_only
@pytest.mark.parametrize("form", ["bytes", "bytearray", "numpy", "readonly", "memoryview"])
def test_apply_gradient_reads_every_admitted_form_in_place(form):
    whole = _make(logical_shards=2, pool_optimizer=False)
    split = _make(logical_shards=2, pool_optimizer=False)
    try:
        whole.train_step(SHARDS)
        total = cv.ordered_fold([split.shard_gradient(s)[1] for s in SHARDS])
        if form == "bytearray":
            total = bytearray(total)
        elif form == "numpy":
            total = np.frombuffer(total, np.float32).copy()
        elif form == "readonly":
            total = np.frombuffer(total, np.float32)
        elif form == "memoryview":
            total = memoryview(total).cast("f")
        assert audit_native_call(split.apply_gradient, total).assert_native() == 1
        assert _raw(split) == _raw(whole)
    finally:
        whole.close()
        split.close()


@gpu_only
def test_device_refuses_nonfinite_gradients_without_committing():
    owner = _make(logical_shards=2, pool_optimizer=False)
    n = _state()[1].n_total
    try:
        before = _raw(owner)
        poisoned = np.zeros(n, np.float32)
        poisoned[n // 2] = np.nan
        # The session's device scan refuses (its message carries no class).
        refused = (ValueError, RuntimeError)
        assert "nonfinite" in native_failure(owner.apply_gradient, refused, poisoned)
        assert _raw(owner) == before
        audit_native_call(owner.fold_reset).assert_native()
        assert "nonfinite" in native_failure(owner.fold_add, refused, poisoned.tobytes())
        assert "nonfinite" in native_failure(owner.fold_reset, refused, poisoned)
        # The fold keeps working after a refusal, and the trainer still steps.
        audit_native_call(owner.fold_reset, np.zeros(n, np.float32)).assert_native()
        assert owner.fold_export() == bytes(4 * n)
        assert owner.train_step(SHARDS)["completed_steps"] == 1
    finally:
        owner.close()


def test_copy_route_keeps_its_refusal_messages():
    owner = _make(logical_shards=2, pool_optimizer=False)
    n = _state()[1].n_total
    try:
        assert "float32 or int32" in native_failure(owner.train_step, TypeError,
                                                    [SHARDS[0].astype(np.int64), SHARDS[1]])
        assert "wrong shape" in native_failure(owner.train_step, ValueError,
                                               [SHARDS[0][:, :4], SHARDS[1]])
        assert "wrong rank" in native_failure(owner.train_step, ValueError, [SHARDS[0][0], SHARDS[1]])
        assert "float32 or int32" in native_failure(owner.apply_gradient, TypeError, np.zeros(n, np.float64))
        assert "wrong shape" in native_failure(owner.apply_gradient, ValueError, bytes(4 * n + 4))
        assert "multiple of item size" in native_failure(owner.apply_gradient, ValueError, bytes(4 * n + 1))
    finally:
        owner.close()


@gpu_only
def test_train_step_from_corpus_commits_the_bits_of_train_step_on_ids(tmp_path):
    batches = _tokens(tmp_path)
    reference = _make(logical_shards=2, pool_optimizer=False)
    owner = _make(logical_shards=2, pool_optimizer=False)
    try:
        for steps in ([0, 1], (5, 3)):
            expected = reference.train_step([batches.ids(s) for s in steps])
            got = audit_native_call(owner.train_step_from, batches, steps).assert_native()
            assert got == expected
            assert _raw(owner) == _raw(reference)
    finally:
        reference.close()
        owner.close()


def test_train_step_from_refuses_before_any_device_work(tmp_path):
    owner = _make(logical_shards=2, pool_optimizer=False)
    try:
        assert "logical shard count mismatch" in native_failure(
            owner.train_step_from, ValueError, _tokens(tmp_path), [0])
        assert "steps must be a list or tuple" in native_failure(
            owner.train_step_from, TypeError, _tokens(tmp_path / "a"), iter([0, 1]))
        assert "native integer" in native_failure(
            owner.train_step_from, TypeError, _tokens(tmp_path / "b"), [0, 1.0])
        assert "wrong shape" in native_failure(
            owner.train_step_from, ValueError, _tokens(tmp_path / "c", batch=2), [0, 1])
        assert "bad dims" in native_failure(
            owner.train_step_from, ValueError, _tokens(tmp_path / "d"), [0, -1])
        assert "corpus operation requires its native-connected owner class" in native_failure(
            owner.train_step_from, TypeError, object(), [0, 1])
        assert owner.step_ == 0
    finally:
        owner.close()
