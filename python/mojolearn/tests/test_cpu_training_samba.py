# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU training for the Samba stack lanes: runtime checks that run only where
the host bindings are built and the package took the CPU-only path.

SambaStack (`python/mojolearn/_samba_impl.py`) holds no numerics: every step
is a call into the training, mamba or transformer binding.

The runtime check (skipped, and SAID to be skipped, when a binding is absent
or a GPU set loaded): where all three families are built, one Samba training
step from the same seed is the same bytes twice.
The bit claim against the GPU columns is the CPU identity gate's.

    cd python && python3 -m mojolearn.tests.test_cpu_training_samba
"""

# Gate-runner scope: host runtime checks require the CPU-only route.
GATE_BACKENDS = ("cpu",)
import sys

import mojolearn
from mojolearn import _backend


def _cpu_only_with(*basenames):
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return False
    built = _backend.host_families_built()
    for basename in basenames:
        if basename not in built:
            print(f"SKIP: {basename} is not built")
            return False
    return True


def test_samba_step_repeats_when_built():
    if not _cpu_only_with("_mojolearn_training_host", "_mojolearn_mamba_host",
                          "_mojolearn_transformer_host"):
        return
    import numpy as np
    ids = (np.arange(2 * 17, dtype=np.int32).reshape(2, 17) * 37) % 256

    def run():
        cfg = mojolearn.SambaConfig(vocab=256, d_model=32, layers=("mamba3", "attention"), n_heads=2,
                                    intermediate=64, tie_embeddings=False, dropout=0.1)
        m = mojolearn.SambaStack(cfg, generator=mojolearn.training.Generator(1), lr=1e-3, max_norm=1.0)
        out = m.train_step(ids[:, :-1], ids[:, 1:])
        p = dict(m.named_parameters())
        return np.float64(out["loss"]).tobytes() + b"".join(np.asarray(p[k]).tobytes() for k in sorted(p))

    first = run()
    assert first == run(), "the same seeded Samba step was not the same bytes twice"


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
