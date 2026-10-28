# SPDX-License-Identifier: Apache-2.0
"""Bounded public generation probes; require rebuilt candidate native entries.

Run with MOJOLEARN_GENERATION_DEVICE=cpu or gpu. These are correctness/lifetime
fixtures, never timing evidence. Source-only lanes syntax-check but do not run.
"""
import gc
import pickle

import numpy as np
import pytest

from mojolearn import Array
from mojolearn.tests.test_generation_native import load_fixture, block_reference


@pytest.fixture(autouse=True)
def local_layer_route(monkeypatch):
    # Exercise the operation-local layer snapshot even when a GPU resident
    # session is available. Existing generation tests qualify that other route.
    monkeypatch.setenv('MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD', '0')


@pytest.mark.parametrize('family', ['llama', 'mamba', 'mamba2'])
def test_snapshot_multi_token_reads_equal_chunked_reads_and_resume(tmp_path, family):
    model = load_fixture(tmp_path, family)
    prompt = np.array([[1, 3], [2, 4]], dtype='i4')
    options = dict(greedy=False, seed=17, top_k=5)
    whole = np.asarray(model.generate(prompt, 5, **options))[:, 2:]
    stream = model.stream(prompt, 5, **options)
    first = np.asarray(stream.read(2)).copy()
    saved = pickle.loads(pickle.dumps(stream.state_dict()))
    rest = np.asarray(stream.read(3)).copy()
    resumed = np.asarray(model.resume_generation(saved).read(3))
    assert np.array_equal(rest, resumed)
    assert np.array_equal(np.concatenate((first, rest), axis=1), whole)
    assert stream.info()['usable'] is True
    assert stream.info()['generated_tokens'] == 5


@pytest.mark.parametrize('family', ['llama', 'mamba', 'mamba2'])
def test_new_input_and_state_metadata_are_refreshed_each_forward(tmp_path, family):
    model = load_fixture(tmp_path, family)
    for values in ([[1, 3, 5]], [[2], [4]]):
        ids = Array.from_list(values, '<i4')
        expected = block_reference(model, ids)
        state = model.allocate_state(len(values), 7)
        actual = np.asarray(model.forward(ids, state))
        assert np.array_equal(actual.view('u4'), expected.view('u4'))
        assert state.positions == len(values[0])
        model.step(np.ones((len(values),), dtype='i4'), state)
        assert state.positions == len(values[0]) + 1


@pytest.mark.parametrize('foreign', [False, True])
def test_head_owner_replacement_and_data_mutation_refresh_between_calls(tmp_path, foreign):
    model = load_fixture(tmp_path)
    prompt = np.array([[1, 3]], dtype='i4')
    original = model._head
    original_values = np.asarray(original).copy()
    baseline = np.asarray(model.forward(prompt)).copy()
    assert np.any(baseline != 0)
    stream = model.stream(prompt, 4)
    stream.read(1)
    # Replace the admitted owner/address between reads, then release caller refs.
    zeros = np.zeros_like(original_values)
    model._head = zeros if foreign else Array.from_buffer(zeros)
    del zeros
    gc.collect()
    assert np.asarray(stream.read(2)).tolist() == [[0, 0]]
    assert np.all(np.asarray(model.forward(prompt)) == 0)
    # Existing owner with changed bytes must also remain visible next operation.
    np.asarray(model._head)[:] = original_values
    restored = np.asarray(model.forward(prompt))
    assert np.array_equal(restored.view('u4'), baseline.view('u4'))
    model._head = original


def test_stream_retains_model_and_weight_owners_after_caller_release(tmp_path):
    model = load_fixture(tmp_path)
    prompt = np.array([[1, 3]], dtype='i4')
    expected = np.asarray(model.generate(prompt, 4))[:, 2:].copy()
    stream = model.stream(prompt, 4)
    del model, prompt
    gc.collect()
    assert np.array_equal(np.asarray(stream.read(4)), expected)


def test_failed_snapshot_admission_cleans_busy_and_marks_stream_unusable(tmp_path):
    model = load_fixture(tmp_path)
    stream = model.stream(np.array([[1, 3]], dtype='i4'), 3)
    # Exact Array metadata refusal occurs while constructing the lazy snapshot.
    # Never forge a pointer: only invalidate its documented contiguous order.
    model._head.order = 'F'
    with pytest.raises(Exception, match='contiguous|order|C-order'):
        stream.read(2)
    assert stream.info()['usable'] is False
    with pytest.raises(Exception, match='lost after a failed call'):
        stream.read(1)


def test_layer_weight_owner_replacement_is_seen_on_next_forward(tmp_path):
    model = load_fixture(tmp_path)
    ids = Array.from_list([[1, 3]], '<i4')
    before = np.asarray(model.forward(ids)).copy()
    block = model.blocks[0]
    original = block._w
    changed = list(original)
    # The first transformer slot is its input norm. Preserve shape/dtype while
    # replacing both object and address through the existing exposed field.
    changed[0] = Array.from_buffer(np.asarray(original[0]).copy() * np.float32(0.5))
    block._w = changed
    del changed
    gc.collect()
    expected = block_reference(model, ids)
    actual = np.asarray(model.forward(ids))
    assert np.array_equal(actual.view('u4'), expected.view('u4'))
    assert not np.array_equal(actual.view('u4'), before.view('u4'))
    block._w = original
    assert np.array_equal(np.asarray(model.forward(ids)).view('u4'), before.view('u4'))
