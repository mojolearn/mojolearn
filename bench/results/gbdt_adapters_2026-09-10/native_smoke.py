"""Focused direct binding check; independent Float64 oracle, no learner suite."""
import importlib.util
import sys
import numpy as np
spec = importlib.util.spec_from_file_location('_mojolearn_gbdt', sys.argv[1])
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
assert b.gbdt_numeric_mode() == int(sys.argv[2])
raw = np.concatenate((np.linspace(-7, 7, 257, dtype=np.float32), np.array([0., -0., np.nextafter(np.float32(0), np.float32(1)), -np.nextafter(np.float32(0), np.float32(1)), 100., -100., np.finfo(np.float32).max, -np.finfo(np.float32).max], dtype=np.float32)))
p = np.empty((len(raw), 2), dtype=np.float32)
c = np.empty(len(raw), dtype=np.int32)
assert b.gbdt_binary_probabilities(raw.ctypes.data, p.ctypes.data, [len(raw)]) == 2*len(raw)
assert b.gbdt_binary_classes(raw.ctypes.data, c.ctypes.data, [len(raw)]) == len(raw)
ref = 1 / (1 + np.exp(-np.clip(raw.astype(np.float64), -700, 700)))
np.testing.assert_allclose(p[:, 1], ref, atol=2e-7, rtol=2e-6)
np.testing.assert_array_equal(c, raw > 0)
np.testing.assert_allclose(p.sum(axis=1), 1., atol=1e-7, rtol=0)
assert p.dtype == np.float32 and np.isfinite(p).all()
assert np.all((p >= 0) & (p <= 1))
print('PASS binary GPU probabilities/class codes', b.gbdt_numeric_mode(), len(raw))
