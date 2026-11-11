# SPDX-License-Identifier: Apache-2.0
"""Prepare the single original PCS input, without product execution.

The original NumPy seeded stream is one indivisible fixture. No sweep or
independent model work is serialized here; native execution cases and oracle
cells are parallel in their CPU suite. This file performs no test or fit.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

root = Path(__file__).resolve().parent
source = np.random.default_rng(0).standard_normal((60, 6)).astype('f4')
source[:5] *= 6
raw = source.astype('<f4').tobytes()
(root / 'source.f32').write_bytes(raw)
metadata = {
    'origin': 'test_x_neighbors_sanity.py::_data/test_polynomial_count_sketch',
    'shape': [60, 6], 'seed': 0, 'first_five_rows_scale': 6,
    'parameters': {'degree': 3, 'gamma': .5, 'coef0': 1, 'components': 32, 'draw_seed': 7},
    'rtol': 1e-3, 'atol': 1e-3, 'numpy': np.__version__,
    'sha256': hashlib.sha256(raw).hexdigest(),
    'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'oracle': 'Native Float64 exhaustive feature-monomial expansion conditioned on actual drawn hashes; no count sketches or convolution recurrence.',
    'execution': 'Input preparation only; no Mojolearn import, compile, test, timing or identity execution.',
}
(root / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
print(json.dumps({'bytes': len(raw), 'shape': [60, 6], 'metadata': str(root / 'metadata.json')}))
