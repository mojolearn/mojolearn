# SPDX-License-Identifier: Apache-2.0
"""A saved Embedding answers on the CPU through its native owner check.

`HostEmbedding` inherits Embedding's native lifetime, so the host archive
connection gives it the host archive's native finalizer instead (its load
requires that descriptor). The Embedding owner check must admit that
connected descriptor for the host inference class; `verify --cross-check` on
the embedding lane was refused with "Embedding subclasses cannot install
Python finalizer callbacks" when it did not. A real Python finalizer stays
refused.
"""
import numpy as np
import pytest

import mojolearn
from mojolearn import _classical_host, embedding


def _saved_host(tmp_path):
    if isinstance(embedding._EMBEDDING_CONTEXT.get("embedding_host_binding"), str):
        pytest.skip("_mojolearn_embedding_infer_host is not built here")
    rng = np.random.default_rng(11)
    w = rng.uniform(-0.25, 0.25, (32, 8)).astype(np.float32)
    ids = rng.integers(0, 32, 64).astype(np.int32)
    model = mojolearn.Embedding(32, 8, padding_idx=3, weight=w)
    path = tmp_path / "embedding.npz"
    model.save(path)
    host = mojolearn.host_model(path)
    assert type(host) is _classical_host.HostEmbedding
    assert type(host).__del__ is not mojolearn.Embedding.__del__
    return model, host, ids


def _same(a, b):
    a, b = np.asarray(a), np.asarray(b)
    assert (a.dtype, a.shape, a.tobytes()) == (b.dtype, b.shape, b.tobytes())


def test_saved_embedding_forwards_on_the_cpu(tmp_path):
    model, host, ids = _saved_host(tmp_path)
    assert host.vendor_used() == "cpu"
    _same(model.forward(ids), host.forward(ids))


def test_python_finalizer_on_the_host_embedding_class_is_still_refused(tmp_path):
    model, host, ids = _saved_host(tmp_path)
    calls = []

    def finalize(self):
        calls.append("finalize")

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(_classical_host.HostEmbedding, "__del__", finalize)
        with pytest.raises(TypeError, match="cannot install Python finalizer"):
            host.forward(ids)
    # Restored: the connected native finalizer answers again.
    _same(model.forward(ids), host.forward(ids))
    assert calls == []
