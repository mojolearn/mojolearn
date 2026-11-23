# SPDX-License-Identifier: Apache-2.0
"""Write the inputs and scikit-learn reference values of the batch 2 encoder Mojo checks (lane gbdt-cpu-rest-enc,
2026-11-23; docs/plans/PY2MOJO_CONVERSION.md, batch 2: the encoder and target encoder rows py2mojo-B did not reach).

The pattern is tools/py2mojo_native_fixtures_b2.py's: every array is the one the original test built (same seed, draw
order and dtype), every reference is the scikit-learn call the original test made, stored as raw little-endian words
by `_put` of tools/py2mojo_native_fixtures.py. Class labels are stored as their codes (the position in the sorted
classes, what the door's label encoding hands the programs). A scikit-learn `None` (an unknown or infrequent value
of an inverse transform) is stored as NaN, as the original tests compare it.

    checks/native_tests/fixtures/prep_enc/  prep_encoders.mojo        (test_x_prep_encoders.py; test_x_prep_parity.py
                                            test_infrequent)
    checks/native_tests/fixtures/prep_te/   prep_target_encoder.mojo  (test_x_prep_target_encoder.py;
                                            test_x_prep_parity.py test_target_encoder_cv_categories)

test_cross_fit_uses_fold_encodings compares against scikit-learn on OUR KFold folds (seed 7, shuffled). The original
asks the binding for them; here `_kfold_folds` is the same keyed permutation written out (x_prep/cpu2_prep.mojo
`c2_kfold_unit`, `_perm_image`, `_fold_of`; x_prep/py2mojo.mojo `splitmix_at`), and the Mojo check first asserts
the native fold stage gives exactly these folds, so a slip here fails the check, never hides.

Size (check-trim, 2026-11-23): the prep_enc pack was 2.8 MB for nine encoder cases. The tables are now ENC_N = 33
and INF_N = 40 rows (were the originals' 257 and 400; the encoder programs have no row threshold below their 1,024-row
chunks, so one chunk ran either way), INF_MIN_FREQ keeps min_frequency=30's share of the rows, the references are
float32 words (each value is asserted to round-trip exactly: codes, -1/-7/-2, NaN and the float32 inputs' category
values) and the one-hot blocks one byte per 0/1 word, their column count the length over the rows (no shape files),
and the 30 infrequent one-hot cases are three concatenated members. The prep_te set is unchanged.

Regenerate and repack (the sets are independent and run one per process, `--workers` of them, default every core;
the prep_enc directory is cleared first so no member of the old layout is packed):

    .pixi/envs/test/bin/python tools/py2mojo_native_fixtures_b2_enc.py && \
        python3 tools/fixture_packs.py pack checks/native_tests/fixtures
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import os
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from py2mojo_native_fixtures import ROOT, _put  # noqa: E402

ENC_N = 33  # rows of the test_x_prep_encoders.py tables (the original 257); prep_encoders.mojo ENC_N
INF_N = 40  # rows of the test_infrequent tables (the original 400); prep_encoders.mojo INF_N
INF_MIN_FREQ = 3  # min_frequency=30 of 400 rows: the same 7.5 percent of INF_N; prep_encoders.mojo INF_MIN_FREQ


def _f32_exact(array):
    """A float64 reference as float32 words, asserted exact (NaN where NaN, every value round-trips)."""
    a = np.asarray(array, dtype=np.float64)
    w = a.astype(np.float32)
    assert np.array_equal(w.astype(np.float64), a, equal_nan=True), "a reference is not exact in float32"
    return w


def _onehot_u8(array):
    """A one-hot reference as one byte per word, asserted to hold only 0 and 1."""
    a = np.asarray(array, dtype=np.float64)
    assert np.isin(a, (0.0, 1.0)).all(), "a one-hot reference holds a value other than 0 and 1"
    return a.astype(np.uint8)


def _obj_to_nan(a):
    # test_x_prep_parity.py::_obj_to_nan (None and the infrequent string as NaN)
    a = np.array(a, dtype=object)
    out = np.empty(a.shape, dtype=np.float64)
    for idx, v in np.ndenumerate(a):
        out[idx] = np.nan if (v is None or isinstance(v, str)) else float(v)
    return out


def _cats_words(cats):
    return np.concatenate([np.asarray(c, dtype=np.float64) for c in cats]), [len(c) for c in cats]


# ---------------------------------------------------------------- test_x_prep_encoders.py
def _enc_data(seed=0, n=ENC_N, d=5):
    rng = np.random.default_rng(seed)
    X = np.floor(rng.standard_normal((n, d)) * 2).astype(np.float32)
    X[:, 2] = np.float32(7)
    X[0, 3] = np.float32(-0.0)
    X[1, 3] = np.float32(0.0)
    X[:, 4] = rng.integers(0, 2, n).astype(np.float32)
    return X


def encoders():
    import sklearn.preprocessing as sk
    out = []

    def put(name, array, dtype=np.float64):
        out.append(_put("prep_enc", name, array, dtype))

    def ref(name, array):
        put(name, _f32_exact(array), np.float32)

    def onehot(name, array):
        put(name, _onehot_u8(array), np.uint8)

    # test_ordinal
    X = _enc_data()
    Xh = _enc_data(1)
    Xh[0, 0] = np.float32(99)
    put("ord-x.f32", X, np.float32)
    put("ord-xh.f32", Xh, np.float32)
    r = sk.OrdinalEncoder().fit(X)
    values, sizes = _cats_words(r.categories_)
    ref("ord-cats.f32", values)
    put("ord-sizes.i32", sizes, np.int32)
    ref("ord-t.f32", r.transform(X))
    r = sk.OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1).fit(X)
    ref("ord-th.f32", r.transform(Xh))
    # test_ordinal_missing_and_inverse
    X = _enc_data()
    X[::9, 0] = np.nan
    Xh = _enc_data(1)
    Xh[::5, 0] = np.nan
    Xh[1, 1] = np.float32(99)
    put("om-x.f32", X, np.float32)
    put("om-xh.f32", Xh, np.float32)
    for k, kw in enumerate((dict(), dict(encoded_missing_value=-7),
                            dict(handle_unknown="use_encoded_value", unknown_value=-1),
                            dict(handle_unknown="use_encoded_value", unknown_value=np.nan, encoded_missing_value=-2))):
        r = sk.OrdinalEncoder(**kw).fit(X)
        values, sizes = _cats_words(r.categories_)
        ref(f"om{k}-cats.f32", values)
        put(f"om{k}-sizes.i32", sizes, np.int32)
        Z = r.transform(X)
        ref(f"om{k}-t.f32", Z)
        ref(f"om{k}-inv.f32", _obj_to_nan(r.inverse_transform(Z)))
        if kw.get("handle_unknown"):
            Z = r.transform(Xh)
            ref(f"om{k}-th.f32", Z)
            ref(f"om{k}-invh.f32", _obj_to_nan(r.inverse_transform(Z)))
    # test_onehot
    X = _enc_data()
    Xh = _enc_data(2)
    Xh[3, 1] = np.float32(55)
    put("oh-x.f32", X, np.float32)
    put("oh-xh.f32", Xh, np.float32)
    for k, kw in enumerate((dict(handle_unknown="ignore"), dict(drop="first"),
                            dict(drop="if_binary", handle_unknown="ignore"))):
        r = sk.OneHotEncoder(sparse_output=False, **kw).fit(X)
        Z = r.transform(X)
        onehot(f"oh{k}-t.u8", Z)
        ref(f"oh{k}-inv.f32", _obj_to_nan(r.inverse_transform(Z)))
        if kw.get("handle_unknown") == "ignore":
            Z = r.transform(Xh)
            onehot(f"oh{k}-th.u8", Z)
            ref(f"oh{k}-invh.f32", _obj_to_nan(r.inverse_transform(Z)))
    # test_given_categories
    X = _enc_data(4)
    Xn = X.copy()
    Xn[5:9, 1] = np.nan
    cats = []
    for j in range(X.shape[1]):
        u = np.unique(Xn[:, j])
        num = [float(v) for v in u if v == v]
        num = sorted(num[:1] + num[2:] + [max(num) + 3.0])
        cats.append(num + ([float("nan")] if np.isnan(u).any() else []))
    put("gc-x.f32", Xn, np.float32)
    # the door's `_given_categories` words: float32, -0.0 as 0.0, NaN last
    canon = np.concatenate([np.where(np.asarray(c, dtype=np.float32) == 0, np.float32(0.0),
                                     np.asarray(c, dtype=np.float32)) for c in cats])
    put("gc-cats.f32", canon, np.float32)
    put("gc-sizes.i32", [len(c) for c in cats], np.int32)
    r = sk.OrdinalEncoder(categories=cats, handle_unknown="use_encoded_value", unknown_value=-1).fit(Xn)
    ref("gc-ord-t.f32", r.transform(Xn))
    for k, kw in enumerate((dict(handle_unknown="ignore"), dict(drop="first", handle_unknown="ignore"))):
        r = sk.OneHotEncoder(categories=cats, sparse_output=False, **kw).fit(Xn)
        onehot(f"gc-oh{k}-t.u8", r.transform(Xn))
    return out


# ---------------------------------------------------------------- test_x_prep_parity.py::test_infrequent
def _inf_data(seed=0, n=INF_N, d=4):
    rng = np.random.default_rng(seed)
    X = np.floor(np.abs(rng.standard_normal((n, d))) * 3).astype(np.float32)
    X[::11, 0] = np.nan
    return X


INF_KW = (dict(min_frequency=INF_MIN_FREQ), dict(max_categories=3), dict(min_frequency=0.05, max_categories=4),
          dict(max_categories=1))
INF_HU = ("infrequent_if_exist", "ignore", "warn")
INF_DROP = (None, "first", "if_binary")


def infrequent():
    import sklearn.preprocessing as skp
    out = []

    def put(name, array, dtype=np.float64):
        out.append(_put("prep_enc", name, array, dtype))

    X, Xh = _inf_data(), _inf_data(1)
    Xh[::7, 1] = 77.0
    put("inf-x.f32", X, np.float32)
    put("inf-xh.f32", Xh, np.float32)
    # the 30 one-hot cases in prep_encoders.mojo's unit order (k, then handle_unknown, then drop), concatenated:
    # their column counts, their transforms of Xh (one byte per word) and their inverse transforms (n x d each)
    oh_cols, oh_th, oh_inv = [], [], []
    for k, kw in enumerate(INF_KW):
        r = skp.OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, **kw).fit(X)
        counts = [-1 if a is None else len(a) for a in r.infrequent_categories_]
        put(f"inf{k}-ord-infcount.i32", counts, np.int32)
        picked = [np.asarray(a, dtype=np.float64) for a in r.infrequent_categories_ if a is not None]
        put(f"inf{k}-ord-infvals.f32", _f32_exact(np.concatenate(picked) if picked else np.zeros(0)), np.float32)
        put(f"inf{k}-ord-th.f32", _f32_exact(r.transform(Xh)), np.float32)
        for h, hu in enumerate(INF_HU):
            for dr, drop in enumerate(INF_DROP):
                if drop is not None and kw.get("max_categories") == 1:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    r = skp.OneHotEncoder(handle_unknown=hu, drop=drop, sparse_output=False, **kw).fit(X)
                    Z = r.transform(Xh)
                    oh_cols.append(Z.shape[1])
                    oh_th.append(_onehot_u8(Z).ravel(order="C"))
                    oh_inv.append(_f32_exact(_obj_to_nan(r.inverse_transform(r.transform(X)))).ravel(order="C"))
    assert len(oh_cols) == 30, "the infrequent one-hot cases are prep_encoders.mojo's 30 units"
    put("inf-oh-cols.i32", oh_cols, np.int32)
    put("inf-oh-th.u8", np.concatenate(oh_th), np.uint8)
    put("inf-oh-inv.f32", np.concatenate(oh_inv), np.float32)
    return out


# ---------------------------------------------------------------- test_x_prep_target_encoder.py
_MASK64 = (1 << 64) - 1
_GOLDEN = 0x9E3779B97F4A7C15
_KEY_MUL = 0xD1B54A32D192ED03
_KEY_ROUNDS = 6


def _splitmix_at(seed, k):
    z = (seed + (k + 1) * _GOLDEN) & _MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK64
    return z ^ (z >> 31)


def _perm_image(s, n, t):
    h = 1
    while (1 << (2 * h)) < n:
        h += 1
    mask = (1 << h) - 1
    x = t
    while True:
        L, R = x >> h, x & mask
        for r in range(_KEY_ROUNDS):
            F = _splitmix_at((s + (r + 1) * _KEY_MUL) & _MASK64, R) & mask
            L, R = R, L ^ F
        x = (L << h) | R
        if x < n:
            return x


def _fold_of(pos, n, F):
    q0, r = n // F, n % F
    B = r * (q0 + 1)
    return pos // (q0 + 1) if pos < B else r + (pos - B) // q0


def _kfold_folds(n, F, seed, shuffle):
    # c2_kfold_unit: the fold of each row's (keyed) position
    return np.array([_fold_of(_perm_image(seed, n, t) if shuffle else t, n, F) for t in range(n)], dtype=np.int32)


def _te_data(seed=0, n=400, d=3):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 5, (n, d)).astype(np.float32)
    y = (X[:, 0] * 0.5 + rng.standard_normal(n)).astype(np.float32)
    return X, y


def _codes(t):
    return np.unique(t, return_inverse=True)[1].astype(np.int32)


def target():
    from sklearn.preprocessing import TargetEncoder as SkTE
    from sklearn.model_selection import StratifiedKFold
    out = []

    def put(name, array, dtype=np.float64):
        out.append(_put("prep_te", name, array, dtype))

    # test_fit_transform_like_reference
    X, y = _te_data()
    Xh, _ = _te_data(1)
    Xh[0, 0] = 9
    yb = (y > 0.8).astype(np.int64)
    ym = np.digitize(y, [0.0, 1.0]).astype(np.int64)
    put("te-x0.f32", X, np.float32)
    put("te-xh.f32", Xh, np.float32)
    put("te-y0.f32", y, np.float32)
    put("te-yb0.i32", _codes(yb), np.int32)
    put("te-ym0.i32", _codes(ym), np.int32)
    for k, (t, kw) in enumerate(((y, dict(smooth="auto")), (y, dict(smooth=2.0)), (yb, {}), (ym, {}))):
        put(f"te-ref{k}.f64", SkTE(**kw).fit(X, t).transform(Xh))
    # test_cross_fit_uses_fold_encodings
    X, y = _te_data(3)
    folds = _kfold_folds(len(y), 5, 7, True)
    put("te-x3.f32", X, np.float32)
    put("te-y3.f32", y, np.float32)
    put("te-folds3.i32", folds, np.int32)
    cross = np.zeros((len(y), X.shape[1]))
    for k in range(5):
        tr, te = folds != k, folds == k
        cross[te] = SkTE().fit(X[tr], y[tr]).transform(X[te])
    put("te-cross3.f64", cross)
    # test_stratified_folds_unshuffled_are_the_reference
    X, y = _te_data(4)
    yb = (y > 0.8).astype(np.int64)
    ym = np.digitize(y, [0.0, 1.0]).astype(np.int64)[::-1].copy()
    put("te-x4.f32", X, np.float32)
    put("te-y4.f32", y, np.float32)
    for tag, t in (("b", yb), ("m", ym)):
        put(f"te-y{tag}4.i32", _codes(t), np.int32)
        want = np.empty(len(t), dtype=np.int32)
        for k, (_tr, te) in enumerate(StratifiedKFold(5).split(X, t)):
            want[te] = k
        put(f"te-strat-{tag}.i32", want, np.int32)
        put(f"te-ft{tag}.f64", SkTE(shuffle=False).fit_transform(X, t))
    put("te-ftc.f64", SkTE(shuffle=False).fit_transform(X, y))
    return out


def target_cv():
    # test_x_prep_parity.py::test_target_encoder_cv_categories
    import sklearn.preprocessing as skp
    import sklearn.model_selection as skm
    out = []

    def put(name, array, dtype=np.float64):
        out.append(_put("prep_te", name, array, dtype))

    rng = np.random.default_rng(6)
    X = rng.integers(0, 5, (200, 2)).astype(np.float32)
    y = rng.standard_normal(200).astype(np.float32)
    yb = (y > 0).astype(int)
    cats = [[0.0, 1.0, 2.0, 3.0, 4.0, 9.0], [0.0, 1.0, 2.0, 3.0, 4.0]]
    put("cv-x.f32", X, np.float32)
    put("cv-y.f32", y, np.float32)
    put("cv-yb.i32", _codes(yb), np.int32)
    for k, (t, cv) in enumerate(((y, skm.KFold(4)), (yb, skm.StratifiedKFold(3)), (y, list(skm.KFold(5).split(X))))):
        pairs = cv if isinstance(cv, list) else list(cv.split(X, t))
        fold = np.full(len(t), -1, dtype=np.int32)
        for f, (_tr, te) in enumerate(pairs):
            fold[te] = f
        assert (fold >= 0).all()
        put(f"cv{k}-folds.i32", fold, np.int32)
        put(f"cv{k}-nfolds.i32", [len(pairs)], np.int32)
        r = skp.TargetEncoder(categories=cats, cv=cv, smooth=2.0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            put(f"cv{k}-ft.f64", r.fit_transform(X, t))
        put(f"cv{k}-enc.f64", np.concatenate([np.asarray(e, dtype=np.float64) for e in r.encodings_]))
    return out


SETS = (encoders, infrequent, target, target_cv)


def _run(index):
    return SETS[index]()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    args = parser.parse_args()
    # encoders and infrequent both write prep_enc: cleared once here, before either runs
    shutil.rmtree(ROOT / "prep_enc", ignore_errors=True)
    with ProcessPoolExecutor(max_workers=max(1, min(len(SETS), args.workers))) as pool:
        for lines in pool.map(_run, range(len(SETS))):
            for line in lines:
                print(line)


if __name__ == "__main__":
    main()
