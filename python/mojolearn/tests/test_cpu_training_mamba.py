# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU training for the Mamba block lanes (lane/cpu-training-mamba,
2026-09-15), checked from SOURCE so it runs on a box with nothing built,
plus runtime checks that run only where the host bindings are built and the
package took the CPU-only path.

What the source checks hold: the manifest's mamba family routes
`_mojolearn_mamba` to its own host binding, which registers every entry the
GPU binding registers (forward, decode step and backward for the three
blocks, and `mamba3_forward_fresh`) plus the read-backs; the generated host
passes under mamba/host/gen/ are exactly what tools/mamba_host_gen.py writes
from the device source today (a device edit that is not regenerated fails
here and in the CPU identity gate's manifest step); no generated file
imports a GPU module or keeps a thread index; the shim routes the device
GEMM through `gemm_oracle`, whose sabotage arm is the family's; and the
declared lanes are identity_break lanes.

The runtime checks (skipped, and SAID to be skipped, when a binding is
absent or a GPU set loaded): a Mamba-2 prefill carried by state and a decode
step equal the one-call forward's rows; the zero cotangent's backward is
zero; the backward refuses a non-finite cotangent by name.
The bit claim against the GPU columns is the CPU identity gate's.

    cd python && python3 -m mojolearn.tests.test_cpu_training_mamba
"""

# Gate-runner scope: host runtime checks require the CPU-only route.
GATE_BACKENDS = ("cpu",)
import re
import subprocess
import sys
from pathlib import Path

import mojolearn
from mojolearn import _backend, host_surface

ROOT = Path(__file__).resolve().parents[3]
GEN = ROOT / "mamba" / "host" / "gen"
GPU_IMPORTS = re.compile(r"^\s*from\s+(max\.gpu|std\.gpu)[\w.]*\s+import", re.M)


def _read(rel):
    return (ROOT / rel).read_text()


def _family():
    return next(f for f in host_surface.FAMILIES if f["family"] == "mamba")


def _registered(text):
    return set(re.findall(r'def_function\[\w+\]\("(\w+)"\)', text))


def test_manifest_routes_the_mamba_binding():
    fam = _family()
    assert fam["routes"] == "_mojolearn_mamba"
    assert fam["binding"] == "_mojolearn_mamba_host"
    assert host_surface.routed_modules()["_mojolearn_mamba"] == "_mojolearn_mamba_host"
    for lane in fam["training_lanes"]:
        assert lane in host_surface.TRAINING_LANE_NAMES, lane


def test_host_binding_registers_every_gpu_entry():
    gpu = _registered(_read("bindings/_mojolearn_mamba.mojo"))
    host = _registered(_read("bindings/_mojolearn_mamba_host.mojo"))
    assert gpu, "no def_function found in the GPU binding"
    # Resident sessions own GPU buffers; CPU blocks expose ordinary stateful
    # inference instead and decode_session explicitly refuses this GPU API.
    resident = {"mamba1_session_" + name for name in
                ("create", "open", "step", "export_state", "load_state", "info", "close")}
    # The Mamba-2/3 resident sessions (769936f70) are the same kind of entry:
    # on the host route their private session classes take the host arm over
    # the per-call `mamba2_decode_step` / `mamba3_decode_step`, which the host
    # binding does register.
    resident |= {f"mamba{v}_session_" + name for v in (2, 3) for name in
                 ("create", "open", "step", "export_state", "load_state", "close")}
    assert resident <= gpu and not resident & host
    missing = sorted(gpu - resident - host)
    assert not missing, f"the host binding lacks {missing}"
    assert host == set(_family()["exports"]), sorted(host ^ set(_family()["exports"]))


def test_generated_passes_are_current():
    run = subprocess.run([sys.executable, str(ROOT / "tools" / "mamba_host_gen.py"), "--check"],
                         capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr


def test_generated_passes_are_host_only():
    files = sorted(GEN.glob("*.mojo"))
    assert len(files) > 10, files
    for p in files:
        text = p.read_text()
        assert not GPU_IMPORTS.search(text), f"{p.name} imports a GPU module"
        code = "\n".join(line.split("#", 1)[0] for line in text.split("\n"))
        code = re.sub(r'"""[\s\S]*?"""', "", code)
        assert not re.search(r"\b(block_idx|thread_idx)\b", code), f"{p.name} keeps a thread index"
    for rel in ("bindings/_mojolearn_mamba_host.mojo", "mamba/host/device_shim.mojo"):
        assert not GPU_IMPORTS.search(_read(rel)), rel


def test_shim_gemm_is_the_oracle_and_carries_the_sabotage():
    shim = _read("mamba/host/device_shim.mojo")
    assert "gemm_oracle(" in shim
    assert "MOJOLEARN_HOST_SABOTAGE" in _read("gemm/host/gemm_oracle.mojo")
    assert "gemm/host/gemm_oracle.mojo" in _family()["host_modules"]


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
