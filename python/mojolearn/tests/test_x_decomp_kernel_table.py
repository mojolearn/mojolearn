"""The x_decomp native kernel-call table and the manifold / linalg kits that
call it (plan NATIVE_ENTRY_POINTS_AFTER_0866, workstream F, rows N290-N301).

Written without a run (lane ne3-f-kernel-table, 2026-11-07); the orchestrator
runs it after the one compile. The table changes no numbers: each kit calls the
same kernels with the same words in the same order, so the checks here are
repeatability and agreement properties, not new expected values.
"""
import threading

import numpy as np
import pytest

from mojolearn import Isomap, LocallyLinearEmbedding, ClassicalMDS, linalg, _backend


def _x(rows, cols, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rows, cols)).astype(np.float32)


def test_host_binding_exports_the_kernel_call_table():
    module = _backend.load_host_module("_mojolearn_x_decomp_host")
    factory = module.__dict__["x_decomp_native_api_v1"]
    capsule = factory()
    assert type(capsule).__name__ == "PyCapsule"
    assert "mojolearn.x_decomp.native.v1" in repr(capsule)


@pytest.mark.parametrize("cls, settings", [
    (Isomap, {"n_neighbors": 5, "n_components": 2}),
    (LocallyLinearEmbedding, {"n_neighbors": 6, "n_components": 2}),
])
def test_repeated_transforms_return_the_same_bits(cls, settings):
    # Fitted matrices are frozen when fit returns (N291, N301): a transform
    # reads them without moving them, so the first and every later transform
    # see the same placement and return the same bytes.
    x = _x(60, 5, 1)
    q = _x(17, 5, 2)
    model = cls(**settings).fit(x)
    first = np.asarray(model.transform(q))
    for _ in range(3):
        again = np.asarray(model.transform(q))
        assert again.tobytes() == first.tobytes()


def test_isomap_transform_matches_a_fresh_fit_of_the_same_data():
    # The fused geodesic gather-min (N295) and the scaled eigenvectors kept by
    # fit give the transform of an identical model the identical bytes.
    x = _x(50, 4, 3)
    q = _x(9, 4, 4)
    a = Isomap(n_neighbors=5, n_components=2).fit(x)
    b = Isomap(n_neighbors=5, n_components=2).fit(x)
    assert np.asarray(a.transform(q)).tobytes() == np.asarray(b.transform(q)).tobytes()


def test_isomap_radius_transform_is_repeatable():
    x = _x(40, 3, 5)
    q = _x(6, 3, 6)
    model = Isomap(n_neighbors=None, radius=3.0, n_components=2).fit(x)
    first = np.asarray(model.transform(q))
    assert np.asarray(model.transform(q)).tobytes() == first.tobytes()


@pytest.mark.parametrize("cls, settings", [
    (Isomap, {"n_neighbors": 5, "n_components": 2}),
    (LocallyLinearEmbedding, {"n_neighbors": 6, "n_components": 2}),
])
def test_concurrent_transforms_of_one_model_agree(cls, settings):
    # N301: two threads reading the same fitted state must not move or free
    # each other's device copies.
    x = _x(80, 5, 7)
    q = _x(25, 5, 8)
    model = cls(**settings).fit(x)
    expected = np.asarray(model.transform(q)).tobytes()
    results = [None] * 6
    errors = []

    def run(i):
        try:
            results[i] = np.asarray(model.transform(q)).tobytes()
        except Exception as error:  # pragma: no cover - reported below
            errors.append(error)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(results))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert all(r == expected for r in results)


def test_fit_is_repeatable_across_calls_with_the_shared_zero():
    # N290: the kit's device zero lives in the shared context; a second fit
    # reuses it and returns the same bytes.
    x = _x(40, 4, 9)
    a = np.asarray(ClassicalMDS(n_components=2).fit_transform(x))
    b = np.asarray(ClassicalMDS(n_components=2).fit_transform(x))
    assert a.tobytes() == b.tobytes()


def test_nonfinite_input_refusal_is_unchanged():
    # N296: the resident kit uploads from the admitted snapshot; the refusal
    # still names the input before any upload.
    x = _x(20, 3, 10)
    x[3, 1] = np.nan
    with pytest.raises(ValueError):
        Isomap(n_neighbors=4).fit(x)


@pytest.mark.parametrize("fn", ["eigh", "svd", "qr", "svdvals"])
def test_linalg_reads_the_operand_in_place_without_writing_it(fn):
    # N293: the admitted operand is borrowed, not copied; no kit operation
    # writes it.
    a = _x(12, 12, 11)
    a = (a + a.T) if fn == "eigh" else a
    before = a.copy()
    first = getattr(linalg, fn)(a)
    assert a.tobytes() == before.tobytes()
    second = getattr(linalg, fn)(a)
    flat = lambda r: b"".join(np.asarray(p).tobytes() for p in (r if isinstance(r, tuple) else (r,)))
    assert flat(first) == flat(second)


def test_linalg_svd_signs_repeatable_with_shared_constants():
    # N294: one zero and one sign pair per kit; the svd tail returns the same
    # bytes as before on every call.
    a = _x(30, 7, 12)
    u1, s1, vt1 = linalg.svd(a, full_matrices=False)
    u2, s2, vt2 = linalg.svd(a, full_matrices=False)
    assert np.asarray(u1).tobytes() == np.asarray(u2).tobytes()
    assert np.asarray(s1).tobytes() == np.asarray(s2).tobytes()
    assert np.asarray(vt1).tobytes() == np.asarray(vt2).tobytes()
