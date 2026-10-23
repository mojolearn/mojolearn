"""Public native CV label placement and canonical descriptor compatibility."""
import hashlib
import json

import numpy as np

import mojolearn as ml
from mojolearn import model_selection as ms
from mojolearn._verify_native_runtime import audit_native_call


def folds():
    rows = np.arange(12, dtype='i8')
    return [(rows[rows % 3 != k], rows[rows % 3 == k]) for k in range(3)]


def test_string_prediction_placement_is_native_and_matches_separate_folds():
    x = np.arange(36, dtype='f4').reshape(12, 3) / np.float32(36)
    y = ['left' if k % 2 else 'right' for k in range(12)]
    options = dict(n_estimators=3, max_depth=2, random_state=11, numeric_mode='identical')
    got = audit_native_call(ms.cross_val_predict, ml.RandomForestClassifier(**options), x, y,
                             cv=folds(), n_jobs=1).assert_native()
    expected = [None] * 12
    for train, test in folds():
        model = ml.RandomForestClassifier(**options).fit(x[train], [y[k] for k in train])
        for at, value in zip(test, model.predict(x[test])):
            expected[at] = value
    assert list(got) == expected


def test_native_split_descriptor_preserves_canonical_unicode_and_array_hashes():
    x = np.arange(36, dtype='f4').reshape(12, 3)
    y = ['café', '雪', 'quote"\\', '\ud800'] * 3
    pairs = folds()
    got = audit_native_call(ms.split_descriptor, x, y, cv=pairs).assert_native()

    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(',', ':'),
                          ensure_ascii=True, allow_nan=False).encode('ascii')

    def digest(*chunks):
        h = hashlib.sha256()
        for chunk in chunks:
            h.update(len(chunk).to_bytes(8, 'little'))
            h.update(chunk)
        return h.hexdigest()

    expected = dict(got)
    expected['X_sha256'] = digest(b'array', b'<f4', canonical(list(x.shape)), x.tobytes())
    expected['y_sha256'] = digest(b'list', canonical(y))
    expected['fold_assignment_sha256'] = digest(b'folds', canonical([
        (train.tolist(), test.tolist()) for train, test in pairs]))
    expected.pop('sha256')
    expected['sha256'] = hashlib.sha256(canonical(expected)).hexdigest()
    assert got == expected
