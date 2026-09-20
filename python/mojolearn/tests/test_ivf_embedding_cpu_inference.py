# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Public CPU inference for a saved IVF-Flat index and a saved embedding
table (lane/inference-embedding-ivf-cholesky, 2026-09-15): the IVF build and
search split into two binding calls, the index and the table saved, and two
inference bindings that ship (`_mojolearn_ivf_search_host`,
`_mojolearn_embedding_infer_host`) with no build and no backward.

Source checks run on a box with nothing built. The runtime checks run where a
binding is built and say when they skip. The bit claim against the GPU
columns is identity_break's ivf, ivf-euclidean and embedding lanes and
tools/classical_host_gate.py over the Metal recording.

    cd python && python -m pytest mojolearn/tests/test_ivf_embedding_cpu_inference.py
"""
import os
import re
from pathlib import Path

import numpy as np
import pytest

from mojolearn import _backend, host_surface
from mojolearn._ivf_impl import IVFIndex, _IVF_FORMAT
from mojolearn.embedding import Embedding, _EMBEDDING_FORMAT

ROOT = Path(__file__).resolve().parents[3]
REGISTERED = re.compile(r'def_function\[\w+\]\(\s*"(\w+)"\s*\)')


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def _registered(rel):
    return set(REGISTERED.findall(_read(rel)))


# -- source -------------------------------------------------------------------

def test_inference_families_ship_and_carry_no_training():
    for fam, route, reference, forbidden, served in (
        ("ivf_search", "_mojolearn_ivf", "ivf", ("ivf_flat_build", "ivf_flat_build_and_search"), "ivf_flat_search"),
        ("embedding_infer", "_mojolearn_embedding", "embedding", ("embedding_backward",), "embedding_forward"),
    ):
        f = host_surface.family(fam)
        assert f["ships_in_wheel"] and f["routes"] is None and f["serves"] == (route,)
        # The reference binding ships too since lane/ship-cpu-host-families
        # (2026-09-16); what this family still guarantees is a binary with no
        # build or backward in it, asserted by name below.
        assert host_surface.family(reference)["ships_in_wheel"]
        names = _registered(host_surface.binding_source(fam))
        assert served in names
        for name in forbidden:
            assert name not in names, f"{fam} registers {name}"
        assert host_surface.inference_routes()[route] == f["binding"]


def test_gpu_and_reference_bindings_register_build_and_search():
    for rel in ("bindings/_mojolearn_ivf.mojo", "bindings/_mojolearn_ivf_host.mojo"):
        names = _registered(rel)
        assert {"ivf_flat_build", "ivf_flat_search", "ivf_flat_build_and_search"} <= names, rel
    shared = _read("bindings/ivf_index_arrays.mojo")
    assert "ivf_validate_index_arrays" in shared
    # Both bindings take their search from the ONE shared source, and since
    # lane/laneless-public-classes (2026-09-19) that source carries the
    # partial (disjoint-shard) search and the Euclidean root as well, so both
    # binaries answer a DistributedIVFIndex worker through the same file.
    # The import may be parenthesized across lines, which is why the names
    # are matched inside the import's own span rather than on one line.
    for rel in ("bindings/_mojolearn_ivf_host.mojo", "bindings/_mojolearn_ivf_search_host.mojo"):
        text = _read(rel)
        head = re.search(r"from bindings\.ivf_host_search import (\([^)]*\)|[^\n]*)", text)
        assert head, rel
        for name in ("ivf_flat_search_binding", "ivf_flat_partial_search_binding",
                     "ivf_finalize_distances_binding"):
            assert re.search(r"\b%s\b" % name, head.group(1)), f"{rel}: {name}"
        assert {"ivf_flat_partial_search", "ivf_finalize_distances"} <= _registered(rel), rel
    for rel in ("bindings/_mojolearn_embedding_host.mojo", "bindings/_mojolearn_embedding_infer_host.mojo"):
        assert "from bindings.embedding_host_forward import" in _read(rel), rel


def test_one_call_host_entry_is_build_then_search():
    text = _read("ivf/host/ivf_host.mojo")
    body = text[text.index("def host_ivf_build_and_search("):]
    assert "host_ivf_build(" in body and "host_ivf_search(" in body


def _fake_index(n=12, dim=3, n_lists=3):
    from mojolearn._buffer import as_f32_c, as_i32_c
    rng = np.random.default_rng(0)
    m = IVFIndex(n_lists=n_lists, n_probes=2, n_neighbors=4, metric="euclidean", random_state=5)
    m.numeric_mode = "identical"
    m.centers_ = as_f32_c(rng.standard_normal((n_lists, dim)).astype(np.float32), ndim=2, name="c")[0]
    m.center_norms_ = as_f32_c(rng.random(n_lists).astype(np.float32), ndim=1, name="n")[0]
    m.list_offsets_ = as_i32_c(np.array([0, 4, 8, 12], np.int32), ndim=1, name="o")[0]
    m.list_indices_ = as_i32_c(np.arange(n, dtype=np.int32), ndim=1, name="i")[0]
    m.list_data_ = as_f32_c(rng.standard_normal((n, dim)).astype(np.float32), ndim=2, name="d")[0]
    m.n_features_in_, m.n_rows_, m.n_lists_, m.metric_code_ = dim, n, n_lists, 1
    return m


def test_ivf_save_load_round_trips_every_array(tmp_path):
    m = _fake_index()
    path = str(tmp_path / "ivf.npz")
    m.save(path)
    back = IVFIndex.load(path)
    for name in ("centers_", "center_norms_", "list_offsets_", "list_indices_", "list_data_"):
        assert np.asarray(getattr(back, name)).tobytes() == np.asarray(getattr(m, name)).tobytes(), name
    assert (back.n_lists, back.n_probes, back.n_neighbors, back.metric, back.random_state) == (3, 2, 4, "euclidean", 5)
    assert back.metric_code_ == 1 and back.numeric_mode == "identical"


def test_ivf_load_refuses_a_cast_and_a_shape(tmp_path):
    from mojolearn import _serialize
    from mojolearn._array import Array
    m = _fake_index()
    path = str(tmp_path / "ivf.npz")
    m.save(path)
    arrays = _serialize.read_npz(path, _IVF_FORMAT)
    arrays["list_indices"] = Array.from_list(list(range(12)), "<i8")
    _serialize.write_npz(path, arrays)
    with pytest.raises(ValueError, match="refusing to cast"):
        IVFIndex.load(path)
    arrays["list_indices"] = Array.from_list(list(range(11)), "<i4")
    _serialize.write_npz(path, arrays)
    with pytest.raises(ValueError, match="not"):
        IVFIndex.load(path)


def test_ivf_search_refuses_a_metric_the_index_was_not_built_under():
    m = _fake_index()
    m.metric = "sqeuclidean"
    with pytest.raises(ValueError, match="a built index has one metric"):
        m.search(np.zeros((1, 3), np.float32))


def test_embedding_save_load_round_trips(tmp_path):
    w = np.random.default_rng(1).standard_normal((8, 3)).astype(np.float32)
    e = Embedding(8, 3, padding_idx=2, weight=w, plan="sort", numeric_mode="identical")
    path = str(tmp_path / "emb.npz")
    e.save(path)
    back = Embedding.load(path)
    assert np.asarray(back.weight).tobytes() == w.tobytes()
    assert (back.padding_idx, back.plan, back.numeric_mode) == (2, "sort", "identical")
    e2 = Embedding(8, 3, weight=w, numeric_mode="identical")
    e2.save(path)
    assert Embedding.load(path).padding_idx is None


# -- runtime (needs a binding) ------------------------------------------------

def _built(basename):
    return os.path.exists(_backend.host_module_path(basename))


@pytest.mark.skipif(not _built("_mojolearn_ivf_search_host"), reason="_mojolearn_ivf_search_host.so is not built here")
def test_the_search_binding_refuses_a_broken_index():
    from mojolearn._classical_host import HostIVFIndex
    m = _fake_index()
    h = HostIVFIndex(n_lists=3, n_probes=2, n_neighbors=4, metric="euclidean")
    h.__dict__.update({k: v for k, v in m.__dict__.items() if k.endswith("_") or k == "numeric_mode"})
    q = np.zeros((2, 3), np.float32)
    d, i = h.search(q)
    assert d.shape == (2, 4) and i.shape == (2, 4)
    from mojolearn._buffer import as_i32_c
    h.list_indices_ = as_i32_c(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 10], np.int32), ndim=1, name="i")[0]
    with pytest.raises(Exception, match="ascending"):
        h.search(q)
    h.list_indices_ = as_i32_c(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10], np.int32), ndim=1, name="i")[0]
    with pytest.raises(Exception, match="stored twice"):
        h.search(q)


@pytest.mark.skipif(not (_built("_mojolearn_ivf_host") and _built("_mojolearn_ivf_search_host")),
                    reason="the ivf reference and search host bindings are not both built here")
def test_a_saved_host_index_searches_the_same_bytes_on_the_shipped_binding(tmp_path):
    import mojolearn
    from mojolearn._cpu_reference import reference_training
    if _backend._CPU_ONLY is None:
        pytest.skip("a GPU set loaded; the reference fit would run on the GPU")
    x = np.random.default_rng(2).standard_normal((512, 8)).astype(np.float32)
    q = np.random.default_rng(3).standard_normal((16, 8)).astype(np.float32)
    with reference_training():
        m = IVFIndex(n_lists=8, n_probes=3, n_neighbors=5, random_state=1).fit(x)
    d0, i0 = m.search(q)
    one = IVFIndex(n_lists=8, n_probes=3, n_neighbors=5, random_state=1)
    path = str(tmp_path / "ivf.npz")
    m.save(path)
    h = mojolearn.host_model(path)
    assert type(h).__name__ == "HostIVFIndex"
    d1, i1 = h.search(q)
    assert np.asarray(d1).tobytes() == np.asarray(d0).tobytes()
    assert np.asarray(i1).tobytes() == np.asarray(i0).tobytes()
    assert np.asarray(h.n_candidates_).tobytes() == np.asarray(m.n_candidates_).tobytes()
    del one


@pytest.mark.skipif(not _built("_mojolearn_embedding_infer_host"), reason="_mojolearn_embedding_infer_host.so is not built here")
def test_a_saved_table_looks_up_on_the_shipped_binding(tmp_path):
    import mojolearn
    w = np.random.default_rng(4).standard_normal((16, 4)).astype(np.float32)
    ids = np.array([3, 0, 15, 3, 7], np.int32)
    path = str(tmp_path / "emb.npz")
    Embedding(16, 4, weight=w, numeric_mode="identical").save(path)
    h = mojolearn.host_model(path)
    assert type(h).__name__ == "HostEmbedding"
    y = np.asarray(h.forward(ids))
    sabotaged = bool(_backend.load_host_module("_mojolearn_embedding_infer_host").embedding_infer_host_sabotage())
    assert (y.tobytes() == w[ids].tobytes()) != sabotaged
    with pytest.raises(Exception, match="embedding_backward|internal bitwise verifier"):
        h.backward(ids, np.zeros((5, 4), np.float32))


# -- stage 2: IVFIndex.extend (2026-09-15) -------------------------------------

def test_extend_is_registered_on_the_gpu_reference_and_shipped_bindings():
    for rel in ("bindings/_mojolearn_ivf.mojo", "bindings/_mojolearn_ivf_host.mojo",
                "bindings/_mojolearn_ivf_search_host.mojo"):
        assert "ivf_flat_extend" in _registered(rel), rel
    for fam in ("ivf", "ivf_search"):
        assert "ivf_flat_extend" in host_surface.family(fam)["exports"], fam
    assert "ivf-extend" in host_surface.family("ivf")["training_lanes"]
    assert "ivf-extend" in host_surface.family("ivf_search")["inference_lanes"]


def test_extend_uses_the_build_assignment_and_one_layout_rule():
    gpu = _read("ivf/impl/neighbors/ivf_flat/ivf_flat_build.mojo")
    body = gpu[gpu.index("def ivf_flat_extend("):]
    assert "predict(ctx, dx, x_norm, centroids, labels, min_dist, kp, n_new, dim)" in body
    assert "extend_list_layout(" in body
    host = _read("ivf/host/ivf_host.mojo")
    hbody = host[host.index("def host_ivf_extend("):]
    assert "host_assign(" in hbody and "extend_list_layout(" in hbody
    layout = _read("ivf/checks/list_layout.mojo")
    assert "out_indices[slot] = UInt32(n_rows + j)" in layout


def test_extend_refuses_without_an_index_and_on_a_changed_metric():
    with pytest.raises(ValueError, match="before extend"):
        IVFIndex(n_lists=2, n_probes=1).extend(np.zeros((2, 3), np.float32))
    m = _fake_index()
    with pytest.raises(ValueError, match="features"):
        m.extend(np.zeros((2, 4), np.float32))
    m.metric = "sqeuclidean"
    with pytest.raises(ValueError, match="a built index has one metric"):
        m.extend(np.zeros((2, 3), np.float32))


def test_clone_copies_and_does_not_alias():
    m = _fake_index()
    c = m._clone()
    assert type(c) is IVFIndex and c.numeric_mode == "identical"
    for name in ("centers_", "center_norms_", "list_offsets_", "list_indices_", "list_data_"):
        assert np.asarray(getattr(c, name)).tobytes() == np.asarray(getattr(m, name)).tobytes()
        assert getattr(c, name) is not getattr(m, name)


def _extend_pair(host_cls):
    from mojolearn._cpu_reference import reference_training
    x = np.random.default_rng(6).standard_normal((600, 6)).astype(np.float32)
    with reference_training():
        base = IVFIndex(n_lists=8, n_probes=3, n_neighbors=5, random_state=2).fit(x[:400])
    return base, x


@pytest.mark.skipif(not (_built("_mojolearn_ivf_host") and _built("_mojolearn_ivf_search_host")),
                    reason="the ivf reference and search host bindings are not both built here")
def test_extend_one_call_equals_two_calls_and_the_shipped_binding(tmp_path):
    import mojolearn
    if _backend._CPU_ONLY is None:
        pytest.skip("a GPU set loaded; the reference fit would run on the GPU")
    base, x = _extend_pair(IVFIndex)
    one = base._clone().extend(x[400:600])
    two = base._clone().extend(x[400:480])
    first = np.asarray(two.extend_labels_).copy()
    two.extend(x[480:600])
    for name in ("list_offsets_", "list_indices_", "list_data_"):
        assert np.asarray(getattr(one, name)).tobytes() == np.asarray(getattr(two, name)).tobytes(), name
    assert np.concatenate([first, np.asarray(two.extend_labels_)]).tobytes() == np.asarray(one.extend_labels_).tobytes()
    ids = np.asarray(one.list_indices_)
    offs = np.asarray(one.list_offsets_)
    for l in range(8):
        seg = ids[offs[l]:offs[l + 1]]
        assert np.all(np.diff(seg) > 0), l
    assert sorted(ids.tolist()) == list(range(600))
    path = str(tmp_path / "ivf.npz")
    base.save(path)
    h = mojolearn.host_model(path)
    assert type(h).__name__ == "HostIVFIndex"
    hx = h._clone().extend(x[400:600])
    for name in ("list_offsets_", "list_indices_", "list_data_"):
        assert np.asarray(getattr(hx, name)).tobytes() == np.asarray(getattr(one, name)).tobytes(), name
    q = x[:16]
    d0, i0 = one.search(q)
    d1, i1 = hx.search(q)
    assert np.asarray(d0).tobytes() == np.asarray(d1).tobytes()
    assert np.asarray(i0).tobytes() == np.asarray(i1).tobytes()
