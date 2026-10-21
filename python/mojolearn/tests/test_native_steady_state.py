# SPDX-License-Identifier: Apache-2.0
"""Repeated public calls through the native shells keep nothing.

Mojo 1.0's PythonObject attribute assignment, item assignment and positional
calls each keep one extra reference on the value or argument
(docs/MODULAR_ASKS.md). The shared shell helpers use bindings/py_balanced.mojo
instead. Each test warms one public operation, then runs it CALLS more times:
the caller's inputs keep their reference counts and the traced Python
allocations after garbage collection stay flat (a retained returned array,
key string or input reference per call would add far more than the bound).

`python -m mojolearn.tests.test_native_steady_state` prints the measured bytes
per call for each operation.
"""
import gc
import sys
import tracemalloc

import numpy as np
import pytest

CALLS = 50
# Bytes per call that may remain after CALLS calls: free-list and dictionary
# resizing noise, far below one retained 64-row output or key object per call.
BOUND = 64


def _counts(values):
    return [sys.getrefcount(v) for v in values]


def growth(operation, inputs=(), calls=CALLS):
    """(traced bytes kept per call, input reference count changes)."""
    for _ in range(3):  # warm caches, lazy imports and pools
        operation()
    gc.collect()
    counts = _counts(inputs)
    tracemalloc.start()
    try:
        gc.collect()
        before = tracemalloc.get_traced_memory()[0]
        for _ in range(calls):  # repeated public operation under test
            operation()
        gc.collect()
        after = tracemalloc.get_traced_memory()[0]
    finally:
        tracemalloc.stop()
    changed = [a - b for a, b in zip(_counts(inputs), counts)]
    return (after - before) / calls, changed


def small_mlp():
    from mojolearn import SmallMLPTrainer
    weights = [((np.arange(np.prod(shape), dtype=np.float32) % 7 - 3) / 32).reshape(shape)
               for shape in ((16, 8), (16,), (3, 16), (3,))]
    model = SmallMLPTrainer(*weights, data_schedule={'dataset': 'steady-state', 'order': 'sequential'})
    x = ((np.arange(64 * 8, dtype=np.float32) % 13 - 6) / 64).reshape(64, 8)
    y = np.arange(64, dtype=np.int32) % 3
    return (lambda: model.train_step(x, y)), (x, y)


def byte_lm():
    from mojolearn import SmallByteLanguageModelTrainer
    model = SmallByteLanguageModelTrainer(np.zeros(34944, dtype=np.float32), data_schedule={
        'corpus_sha256': '1' * 64, 'token_schedule_sha256': '2' * 64,
        'batch_offsets': [0, 32], 'planned_steps': 1000})
    ids = (np.arange(66, dtype=np.int32) % 256).reshape(2, 33)
    return (lambda: model.train_step(ids)), (ids,)


def sequential():
    from mojolearn import Sequential
    model = Sequential().linear(1).compile(optimizer="sgd", lr=0.03)
    x = np.arange(12, dtype="f4").reshape(4, 3)
    y = np.arange(4, dtype="f4")
    return (lambda: model.fit(x, y, epochs=1)), (x, y)


def causal_lm(tmp_path):
    from mojolearn._causal_lm_fixtures import family_fixture, _write_checkpoint
    from mojolearn.models import CausalLM
    config, tensors = family_fixture("llama", False)
    _write_checkpoint(str(tmp_path), config, tensors)
    model = CausalLM.load(tmp_path, device="cpu", numeric_profile="fp32_v1")
    ids = np.array([[1, 2, 3]], dtype="i4")
    return (lambda: model.generate(ids, 4)), (ids,)


def sha256_tokenizer():
    from mojolearn.tokenizer import _binding
    entry = _binding().sha256_hex
    data = bytes(range(256)) * 16
    return (lambda: entry(data)), (data,)


def sha256_byte_lm():
    from mojolearn import _byte_lm_impl
    data = bytes(range(256)) * 16
    return (lambda: _byte_lm_impl._sha256_hex(data)), (data,)


OPERATIONS = {
    "SmallMLPTrainer.train_step": small_mlp,
    "SmallByteLanguageModelTrainer.train_step": byte_lm,
    "Sequential.fit": sequential,
    "CausalLM.generate": causal_lm,
    "tokenizer sha256_hex": sha256_tokenizer,
    "byte-LM sha256_hex": sha256_byte_lm,
}


def _build(name, tmp_path):
    factory = OPERATIONS[name]
    return factory(tmp_path) if factory is causal_lm else factory()


@pytest.mark.parametrize("name", list(OPERATIONS))
def test_repeated_calls_keep_nothing(name, tmp_path):
    operation, inputs = _build(name, tmp_path)
    per_call, changed = growth(operation, inputs)
    assert changed == [0] * len(inputs), (name, changed)
    assert per_call < BOUND, (name, per_call)


if __name__ == "__main__":
    import tempfile
    from pathlib import Path
    selected = sys.argv[1:] or list(OPERATIONS)
    for name in selected:  # report each selected operation
        with tempfile.TemporaryDirectory() as root:
            try:
                operation, inputs = _build(name, Path(root))
                per_call, changed = growth(operation, inputs)
                print(f"{name}: {per_call:.0f} bytes/call, input refcount change {changed}")
            except Exception as error:  # report and continue to the next operation
                print(f"{name}: ERROR {type(error).__name__}: {str(error)[:200]}")
