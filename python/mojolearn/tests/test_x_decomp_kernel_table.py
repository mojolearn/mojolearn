"""The x_decomp native kernel-call table and the manifold / linalg kits that
call it (plan NATIVE_ENTRY_POINTS_AFTER_0866, workstream F, rows N290-N301).

Written without a run (lane ne3-f-kernel-table, 2026-11-07); the orchestrator
runs it after the one compile. The table changes no numbers: each kit calls the
same kernels with the same words in the same order, so the checks here are
repeatability and agreement properties, not new expected values.

The repeat, fresh-fit, radius, concurrent-transform and shared-zero checks of
the manifold kit moved to checks/native_tests/mani_decomp_repeat.mojo; this
file keeps the binding's capsule export, the refusal at the public boundary
and the linalg kit's reads of an admitted NumPy operand.
"""
import numpy as np
import pytest

from mojolearn import Isomap, linalg, _backend


def _x(rows, cols, seed):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((rows, cols)).astype(np.float32)


def test_host_binding_exports_the_kernel_call_table():
    module = _backend.load_host_module("_mojolearn_x_decomp_host")
    factory = module.__dict__["x_decomp_native_api_v1"]
    capsule = factory()
    assert type(capsule).__name__ == "PyCapsule"
    assert "mojolearn.x_decomp.native.v1" in repr(capsule)


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
