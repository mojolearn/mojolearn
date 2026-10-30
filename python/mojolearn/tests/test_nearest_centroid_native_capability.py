"""The identical native descriptor must use its available softmax export."""
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("shrink", ["none", "0.2"])
def test_identical_nearest_centroid_probabilities_through_native_descriptor(shrink):
    script = r'''
import sys
import numpy as np
import mojolearn as ml

rng = np.random.default_rng(518)
y = (np.arange(36) % 3).astype(np.int32)
x = (rng.standard_normal((36, 4)) + y[:, None]).astype(np.float32)
q = rng.standard_normal((11, 4)).astype(np.float32)
threshold = None if sys.argv[1] == "none" else float(sys.argv[1])
model = ml.NearestCentroid(shrink_threshold=threshold).fit(x, y)
probabilities = model.predict_proba(q)
scores = model.decision_function(q).astype(np.float64)
expected = np.exp(scores - scores.max(axis=1, keepdims=True))
expected /= expected.sum(axis=1, keepdims=True)
assert probabilities.dtype == np.float32
assert probabilities.shape == (11, 3)
np.testing.assert_allclose(probabilities, expected, rtol=2e-5, atol=2e-6)
np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=2e-6)
np.testing.assert_array_equal(probabilities.argmax(axis=1), model.predict(q))
print("nearest-centroid-identical-native-capability-complete", flush=True)
'''
    environment = dict(os.environ, MOJOLEARN_NUMERIC_MODE="identical")
    result = subprocess.run([sys.executable, "-c", script, shrink], env=environment,
                            capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, (result.returncode, result.stdout, result.stderr)
    assert "nearest-centroid-identical-native-capability-complete" in result.stdout
