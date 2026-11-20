"""CPU IVF-PQ output copies must not inspect Python addresses without the GIL.

The numerical claims (finite distances, kept ids, two fits agree word for
word) are the native check checks/native_tests/ann_ivfpq_host_filter.mojo;
this smoke keeps the binding's GIL path: fit, filtered search and the
returned arrays, in a subprocess that turns a native crash into a failure."""
import os
import subprocess
import sys


def test_cpu_ivfpq_fit_and_filtered_search():
    # A subprocess turns a native crash into a test failure, rather than taking
    # down the suite. Both formerly crashing verifier lanes build this index.
    script = '''
import numpy as np
import mojolearn as ml
rng = np.random.default_rng(41)
x = rng.standard_normal((256, 8)).astype(np.float32)
q = x[:16].copy()
keep = np.arange(len(x)) % 3 != 0
for _ in range(2):
    model = ml.IVFPQIndex(n_lists=8, n_probes=4, pq_dim=2, pq_bits=2,
        n_neighbors=5, pq_kmeans_n_iters=3, random_state=3).fit(x)
    distances, indices = model.search(q, filter=keep)
    np.asarray(distances), np.asarray(indices), np.asarray(model.codes_)
print("CPU_IVFPQ_GIL_OK")
'''
    result = subprocess.run([sys.executable, '-X', 'faulthandler', '-c', script],
        env={**os.environ, 'MOJOLEARN_VENDOR': 'cpu'}, capture_output=True, text=True,
        timeout=120)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert 'CPU_IVFPQ_GIL_OK' in result.stdout
