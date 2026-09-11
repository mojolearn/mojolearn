#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Buffer conversions survive another library retyping ctypes.pythonapi.

    python3 tools/check_buffer_foreign_argtypes.py              # no cuML needed
    python3 tools/check_buffer_foreign_argtypes.py --real-cuml  # import cuml instead

THE BUG (0.8.0, fixed in 0.8.1). `_buffer.py` took
`ctypes.pythonapi.PyObject_GetBuffer` by attribute access. ctypes caches that
function pointer on the process-wide `pythonapi` singleton, so every library
spelling it that way shares ONE object and the last `argtypes` assignment
wins. `treelite.model` (imported by cuML) assigns its own `_PyBuffer` there,
and every mojolearn buffer conversion in that process then failed with
"argument 2: ... expected LP__PyBuffer instance instead of pointer to
_PyBuffer". 0.8.1 takes private pointers with `_api["PyObject_GetBuffer"]`.

ORDER MATTERS. 0.8.0 assigned its argtypes when `_buffer.py` was imported, so
a foreign assignment made BEFORE `import mojolearn` is overwritten by ours and
the check would pass on the broken release. The failing order, used here, is
import mojolearn, then the foreign assignment, then fit and predict.

The input is a 2-D float32 memoryview, not a list: a list becomes an `Array`,
and `Buf` reads an `Array` from its store without `PyObject_GetBuffer`.

Exit 0 on success, 1 with the error text on failure, 2 when nothing was
tested (cuML absent under --real-cuml, or the shared pointer was not
retyped, which is not a pass).
"""

import argparse
import array
import ctypes
import random
import sys
import traceback


class _PyBuffer(ctypes.Structure):
    """treelite.model's `Py_buffer` class, spelled the way it spells it."""

    _fields_ = (
        ("buf", ctypes.c_void_p),
        ("obj", ctypes.py_object),
        ("len", ctypes.c_ssize_t),
        ("itemsize", ctypes.c_ssize_t),
        ("readonly", ctypes.c_int),
        ("ndim", ctypes.c_int),
        ("format", ctypes.c_char_p),
        ("shape", ctypes.POINTER(ctypes.c_ssize_t)),
        ("strides", ctypes.POINTER(ctypes.c_ssize_t)),
        ("suboffsets", ctypes.POINTER(ctypes.c_ssize_t)),
        ("internal", ctypes.c_void_p),
    )


def poison_pythonapi():
    """Retype the shared `ctypes.pythonapi` pointers the way a foreign library
    does. The first four lines are treelite.model's; the last two cover the
    other pointers 0.8.1 made private. Returns a one-line description."""
    api = ctypes.pythonapi
    api.PyMemoryView_FromBuffer.argtypes = [ctypes.POINTER(_PyBuffer)]
    api.PyMemoryView_FromBuffer.restype = ctypes.py_object
    api.PyObject_GetBuffer.argtypes = [ctypes.py_object, ctypes.POINTER(_PyBuffer), ctypes.c_int]
    api.PyObject_GetBuffer.restype = ctypes.c_int
    api.PyBuffer_Release.argtypes = [ctypes.POINTER(_PyBuffer)]
    api.PyMemoryView_FromMemory.argtypes = [ctypes.c_char_p, ctypes.c_ssize_t, ctypes.c_int]
    return armed()


def armed():
    """Proof the shared pointer is foreign-typed: a 0.8.0-shaped caller (its
    own Py_buffer class through the shared pointer) must be refused here.
    Returns a description when armed, None when not."""
    argtypes = ctypes.pythonapi.PyObject_GetBuffer.argtypes or []
    if len(argtypes) < 2:
        return None

    class _Ours(ctypes.Structure):
        _fields_ = _PyBuffer._fields_

    try:
        ctypes.pythonapi.PyObject_GetBuffer(bytearray(8), ctypes.byref(_Ours()), 0)
    except ctypes.ArgumentError as exc:
        return f"shared PyObject_GetBuffer retyped to {argtypes[1].__name__}; a 0.8.0 caller gets: {exc}"
    return None


def fit_predict():
    import mojolearn
    from mojolearn import _buffer

    n, d = 400, 8
    rng = random.Random(0)
    flat = array.array("f", (rng.gauss(0.0, 1.0) for _ in range(n * d)))
    X = memoryview(flat).cast("B").cast("f", shape=[n, d])
    y = array.array("i", (1 if flat[i * d] + 0.5 * flat[i * d + 1] > 0 else 0 for i in range(n)))

    # Reach: the three private pointers, directly.
    with _buffer.Buf(X, name="X") as b:
        if b.shape != (n, d):
            raise AssertionError(f"Buf read shape {b.shape}, expected {(n, d)}")
        if bytes(_buffer.memory_at(b.addr, 4, writable=False)) != bytes(memoryview(flat).cast("B")[:4]):
            raise AssertionError("memory_at read different bytes than the exporter holds")

    m = mojolearn.RandomForestClassifier(n_estimators=8, max_depth=6, random_state=7)
    m.fit(X, y)
    pred = m.predict(X).tolist()
    if len(pred) != n:
        raise AssertionError(f"predict returned {len(pred)} rows, expected {n}")
    acc = sum(int(p) == int(t) for p, t in zip(pred, y)) / n
    if acc < 0.7:
        raise AssertionError(f"training accuracy {acc:.3f} on a separable rule")
    return mojolearn, acc


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--real-cuml", action="store_true",
                    help="import cuml (and treelite.model) instead of retyping the pointers here")
    a = ap.parse_args()

    import mojolearn  # before the foreign assignment, see ORDER MATTERS
    from mojolearn import _buffer  # noqa: F401

    if a.real_cuml:
        try:
            import cuml
        except ImportError as exc:
            print(f"buffer-foreign-argtypes: NOT TESTED, cuml not importable: {exc}")
            return 2
        try:
            import treelite.model  # noqa: F401  cuML may import it lazily
        except ImportError:
            pass
        how = armed()
        source = f"cuml {cuml.__version__}"
    else:
        how = poison_pythonapi()
        source = "treelite.model spelling, no cuML"
    if how is None:
        print(f"buffer-foreign-argtypes: NOT TESTED, shared pointer not retyped ({source})")
        return 2
    print(f"armed ({source}): {how}")

    try:
        ml, acc = fit_predict()
    except Exception:
        print("buffer-foreign-argtypes: FAIL")
        traceback.print_exc(file=sys.stdout)
        return 1
    print(f"buffer-foreign-argtypes: PASS mojolearn {ml.__version__} mode={ml.numeric_mode()} "
          f"vendor={ml.vendor()} rf_train_accuracy={acc:.3f} ({source})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
