"""Load isolated candidate extensions; no learner or performance measurement."""
import importlib
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(sys.argv[1]).resolve()))
for name in ('_mojolearn_rf', '_mojolearn_trees'):
    native = importlib.import_module(name)
    assert native.forest_resident_layout() == 'packed_siblings'
    off = np.array([0, 3], dtype=np.int32)
    col = np.array([0, -1, -1], dtype=np.int32)
    threshold = np.array([2, 0, 0], dtype=np.float32)
    left = np.array([1, -1, -1], dtype=np.int32)
    leaves = np.array([-999, -999, .25, .75, .75, .25], dtype=np.float32)
    x = np.array([1, 2, 3], dtype=np.float32)
    expected = np.array([.25, .75, .25, .75, .75, .25], dtype=np.float32)
    handle = native.forest_prepare_gpu(*(v.ctypes.data for v in (off, col, threshold, left, leaves)), [1, 1, 2])
    try:
        for entry in ('forest_predict_resident_gpu', 'forest_predict_resident_into_gpu', 'forest_predict_resident_reuse_gpu'):
            actual = np.full(6, -7, dtype=np.float32)
            getattr(native, entry)(handle, x.ctypes.data, actual.ctypes.data, [3, 1, 2])
            np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
        print(json.dumps(dict(binding=name, layout=native.forest_resident_layout(), result='PASS', paths=3)))
    finally:
        native.forest_release_gpu(handle)
