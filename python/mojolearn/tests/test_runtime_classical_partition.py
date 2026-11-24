"""Public classical fit, predict and archive run native entries; tree residency stays native."""
import numpy as np

import mojolearn as ml
from mojolearn._verify_native_runtime import audit_native_call


def test_public_classical_fit_predict_and_archive_use_native_entries(tmp_path):
    axis = np.linspace(-1, 1, 12, dtype=np.float32)
    x = np.stack((axis, axis * axis), axis=1)
    y = (x[:, 0] * np.float32(2) - x[:, 1] * np.float32(.5) + np.float32(3)).copy()
    model = audit_native_call(ml.LinearRegression, numeric_mode='identical').assert_native()
    audit_native_call(model.fit, x, y).assert_native()
    expected = np.asarray(audit_native_call(model.predict, x).assert_native()).copy()
    np.testing.assert_allclose(expected.reshape(-1), y, rtol=0, atol=0.001)
    archive = str(tmp_path / 'linear.npz')
    audit_native_call(model.save, archive).assert_native()
    restored = audit_native_call(ml.LinearRegression.load, archive).assert_native()
    got = np.asarray(audit_native_call(restored.predict, x).assert_native())
    np.testing.assert_array_equal(got.view(np.uint32), expected.view(np.uint32))


def test_tree_resident_type_comes_from_the_single_classical_owner():
    x = np.arange(60, dtype=np.float32).reshape(20, 3)
    y = np.arange(20, dtype=np.int32) % 2
    model = audit_native_call(ml.RandomForestClassifier, n_estimators=3, max_depth=3,
                              random_state=13, numeric_mode='identical').assert_native()
    audit_native_call(model.fit, x, y).assert_native()
    # The snapshot lives in the classical owner's native resident table
    # (bindings/trees_resident_table_run.mojo), never in the model's __dict__.
    from mojolearn import _native_trees_bridge
    entry, context = _native_trees_bridge.connect()
    kept_before = entry(None, 'trees_resident_stats', (), context)[0]
    audit_native_call(model.predict, x).assert_native()
    fields = object.__getattribute__(model, '__dict__')
    assert not fields.get('_resident_holder')
    assert entry(None, 'trees_resident_stats', (), context)[0] == kept_before + 1
