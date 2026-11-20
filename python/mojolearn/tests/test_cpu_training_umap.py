# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU training for the umap lane: a runtime check that runs only where the
metrics host binding is built and the package took the CPU-only path.

The runtime check (skipped, and SAID to be skipped, when the binding is
absent or a GPU set loaded): a small UMAP fits and transforms twice through
the host binding and returns the same bytes. It is a plumbing check. The bit
claim against the GPU columns is the CPU identity gate's, not this file's.

    cd python && python3 -m mojolearn.tests.test_cpu_training_umap
"""

# Gate-runner scope: host runtime checks require the CPU-only route.
GATE_BACKENDS = ("cpu",)
import sys

import mojolearn
from mojolearn import _backend


def test_umap_fits_on_the_host_when_built():
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return
    if "_mojolearn_metrics_host" not in _backend.host_families_built():
        print("SKIP: the metrics host binding is not built")
        return
    import numpy as np
    module = _backend.load_host_module("_mojolearn_metrics_host")
    assert not bool(module.metrics_host_sabotage()), "a sabotage build loaded outside the gate"
    rng = np.random.default_rng(0)
    x = rng.standard_normal((128, 5)).astype(np.float32)
    q = rng.standard_normal((16, 5)).astype(np.float32)
    outs = []
    for _ in range(2):
        m = mojolearn.UMAP(n_neighbors=6, n_components=2, n_epochs=4, random_state=1).fit(x)
        t = m.transform(q)
        emb = np.asarray(m.embedding_)
        assert np.isfinite(emb).all() and np.isfinite(np.asarray(t)).all()
        outs.append((emb.tobytes(), np.asarray(t).tobytes()))
    assert outs[0] == outs[1], "two host fits returned different bytes"


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
