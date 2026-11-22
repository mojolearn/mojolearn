# SPDX-License-Identifier: Apache-2.0
"""Categorical inputs of GradientBoosting at the boundary (lanes cat-ctr round 11, arrow-strings, fx-hostgbdt).

Reuses every test of tests/api/test_gbdt_category_inputs.py (object, U / S arrays, pandas 2.x / 3.x columns,
Arrow string / large-string / chunked / sliced / dictionary columns, refusals by name) and adds the cat-ctr RUN
OWED cells it does not cover: NumPy scalar and bytes cells in list rows, and one category for a value given as
str and as bytes. Run it under pandas 2.x + pyarrow and under pandas 3.x with and without pyarrow; the pandas and
Arrow tests skip by themselves where their library is absent.
"""
import numpy as np

from _proving import assert_same, reexport

_cats = reexport(globals(), "tests/api/test_gbdt_category_inputs.py", "proving_reused_test_gbdt_category_inputs")
# fx-hostgbdt: the HostGBDT category map (host twin matches the model on every input form, digest, CTR models).
reexport(globals(), "tests/api/test_host_gbdt_category_map.py", "proving_reused_test_host_gbdt_category_map")


def test_proving_numpy_scalar_and_bytes_cells_match_python_cells(reference):
    cat, num, y = _cats._data()
    as_numpy = [[np.str_(cat[i]), float(num[i])] for i in range(_cats.N)]
    assert_same(_cats._fit_predict(as_numpy, y), reference, "numpy.str_ cells")
    as_bytes = [[cat[i].encode(), float(num[i])] for i in range(_cats.N)]
    assert_same(_cats._fit_predict(as_bytes, y), reference, "bytes cells")


def test_proving_numpy_float32_numeric_cells():
    cat, num, y = _cats._data()
    python_rows = [[cat[i], float(np.float32(num[i]))] for i in range(_cats.N)]
    numpy_rows = [[cat[i], np.float32(num[i])] for i in range(_cats.N)]
    assert_same(_cats._fit_predict(numpy_rows, y), _cats._fit_predict(python_rows, y), "numpy.float32 cells")


def test_proving_numpy_int64_category_cells_match_python_ints():
    _, num, y = _cats._data()
    python_rows = [[i % 4, float(num[i])] for i in range(_cats.N)]
    numpy_rows = [[np.int64(i % 4), float(num[i])] for i in range(_cats.N)]
    assert_same(_cats._fit_predict(numpy_rows, y), _cats._fit_predict(python_rows, y), "numpy.int64 cells")


def test_proving_str_and_bytes_of_one_value_are_one_category(reference):
    cat, num, y = _cats._data()
    mixed = [[cat[i] if i % 2 else cat[i].encode(), float(num[i])] for i in range(_cats.N)]
    assert_same(_cats._fit_predict(mixed, y), reference, "str and bytes mixed")


def test_proving_every_column_categorical_s_and_u_arrays():
    _, _, y = _cats._data()
    cells = [["a", "x"], ["b", "y"], ["c", "x"], ["a", "z"]] * (_cats.N // 4)
    U = np.array(cells)
    S = U.astype("S")
    import mojolearn

    def fit_predict(X):
        model = mojolearn.GradientBoosting(n_estimators=8, max_depth=3, cat_features=[0, 1], device="cpu",
                                           **_cats.CPU_CTR)
        return model.fit(X, y).predict(X)

    assert_same(fit_predict(S), fit_predict(U), "S vs U, every column categorical")
    assert_same(fit_predict(cells), fit_predict(U), "list rows vs U")


def test_proving_cpu_ctr_predict_connects_the_host_owner_on_first_use():
    # pv-lazy-connect: `import mojolearn` (and `mojolearn.ensemble`) no longer imports the CPU tree owner
    # (`mojolearn._classical_host`); the gbdt door connects it once at the first device='cpu' CTR predict
    # (bindings/trees_public_gbdt_door.mojo gbdt_host_tree_context). A fresh process, HostGBDT never imported.
    # On a GPU install nothing else loads the owner, so the first CPU CTR predict connects it.
    import subprocess
    import sys
    script = (
        "import sys, numpy as np, mojolearn, mojolearn.ensemble\n"
        "if mojolearn.vendor() != 'cpu':  # a CPU-only install loads the host forest, and with it the owner\n"
        "    assert 'mojolearn._classical_host' not in sys.modules, 'host surface imported at import'\n"
        "X = np.array([['a', 'x'], ['b', 'y'], ['c', 'x'], ['a', 'z']] * 30)\n"
        "y = np.array([0, 1, 1, 0] * 30, dtype=np.float32)\n"
        "m = mojolearn.GradientBoosting(n_estimators=8, max_depth=3, cat_features=[0, 1], device='cpu',\n"
        "                               loss='Logloss', bootstrap_type='No', random_strength=0.0).fit(X, y)\n"
        "first = m.predict(X)\n"
        "assert 'mojolearn._classical_host' in sys.modules\n"
        "assert np.array_equal(m.predict(X), first)\n"
        "print('OK', first.shape)\n"
    )
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=600)
    assert done.returncode == 0 and "OK" in done.stdout, done.stderr[-2000:]
