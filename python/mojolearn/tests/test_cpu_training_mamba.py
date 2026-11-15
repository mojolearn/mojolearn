# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU training for the Mamba block lanes: the generated host passes are
current, plus runtime checks that run only where the host bindings are built
and the package took the CPU-only path.

The generated host passes under mamba/host/gen/ are exactly what
tools/mamba_host_gen.py writes from the device source today (a device edit
that is not regenerated fails here and in the CPU identity gate's manifest
step).

The runtime checks (skipped, and SAID to be skipped, when a binding is
absent or a GPU set loaded): a Mamba-2 prefill carried by state and a decode
step equal the one-call forward's rows; the zero cotangent's backward is
zero; the backward refuses a non-finite cotangent by name.
The bit claim against the GPU columns is the CPU identity gate's.

    cd python && python3 -m mojolearn.tests.test_cpu_training_mamba
"""

# Gate-runner scope: host runtime checks require the CPU-only route.
GATE_BACKENDS = ("cpu",)
import subprocess
import sys
from pathlib import Path

import mojolearn
from mojolearn import _backend

ROOT = Path(__file__).resolve().parents[3]


def test_generated_passes_are_current():
    run = subprocess.run([sys.executable, str(ROOT / "tools" / "mamba_host_gen.py"), "--check"],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr


def _cpu_only_with(basename):
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return False
    if basename not in _backend.host_families_built():
        print(f"SKIP: {basename} is not built")
        return False
    return True


def _mamba2_block():
    import numpy as np
    rng = np.random.default_rng(5)
    dm, di, nh = 32, 64, 1
    cd, dip = di + 256, 2 * di + 256 + nh
    shapes = {"block_norm.weight": (dm,), "in_proj.weight": (dip, dm), "conv1d.weight": (cd, 1, 4),
              "conv1d.bias": (cd,), "dt_bias": (nh,), "A_log": (nh,), "D": (nh,), "norm.weight": (di,),
              "out_proj.weight": (dm, di)}
    w = {k: (np.ones(s, np.float32) if k.endswith("norm.weight")
             else (rng.standard_normal(s) / 8).astype(np.float32)) for k, s in shapes.items()}
    return mojolearn.Mamba2Block(w), rng


def test_mamba2_state_and_step_match_the_prefill_when_built():
    if not _cpu_only_with("_mojolearn_mamba_host"):
        return
    import numpy as np
    blk, rng = _mamba2_block()
    x = rng.standard_normal((2, 9, 32)).astype(np.float32)
    whole = np.asarray(blk.forward(x))
    st = blk.allocate_state(2)
    head = np.asarray(blk.forward(np.ascontiguousarray(x[:, :8]), st))
    tail = np.asarray(blk.step(np.ascontiguousarray(x[:, 8:]), st))
    assert head.tobytes() == whole[:, :8].tobytes()
    assert tail.reshape(2, 1, 32).tobytes() == whole[:, 8:].tobytes()


def test_mamba2_backward_zero_and_refusal_when_built():
    if not _cpu_only_with("_mojolearn_mamba_host"):
        return
    import numpy as np
    blk, rng = _mamba2_block()
    x = rng.standard_normal((1, 4, 32)).astype(np.float32)
    g = blk.backward(x, np.zeros_like(x))
    assert all(not np.asarray(v).any() for v in g.values())
    bad = np.zeros_like(x)
    bad[0, 1, 2] = np.inf
    try:
        blk.backward(x, bad)
    except ValueError as exc:
        assert "finite" in str(exc), str(exc)
    else:
        raise AssertionError("backward accepted a non-finite cotangent")


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
