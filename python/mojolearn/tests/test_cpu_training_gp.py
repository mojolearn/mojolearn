# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Workstream E, CPU training for the gp, gp-matern12, gp-matern32 and
gp-matern52-ard lanes (the gp host lane, 2026-09-14), checked from SOURCE so
it runs on a box with nothing built, plus a runtime check that runs only
where the gp host binding is built and the package took the CPU-only path.

What the source checks hold: the manifest declares the gp family, routes
`_mojolearn_gp`, covers the four GP lanes and no longer names the Gaussian
process as having no CPU path; the binding registers the GPU binding's
gpr_fit, gpr_predict and Cholesky door names and leaves
gp_parallel_available absent so it refuses by name; the two oracles import
no GPU module and nothing beyond the numerics seams, each other and the
shipped gemm host oracle; the oracles spell the device constructs a bit
claim rests on (the pinned panel width 32, the pivot comparison, the lower
triangle subtract, the logdet chain, the lml order, the structural white
kernel, the clamp spelled as a comparison); the sabotage define reaches the
distance's feature loop and the binding reads it back; the CPU identity gate runs by hand since 2026-09-15 (no push trigger).

The runtime check (skipped, and SAID to be skipped, when the binding is
absent or a GPU set loaded): the four lane kernels fit and predict twice
through the host binding on a small draw and return the same bytes, info
0 and a finite likelihood. It is a plumbing check. The bit claim against
the GPU columns is the CPU identity gate's, not this file's.

    cd python && python3 -m mojolearn.tests.test_cpu_training_gp
"""

# Gate-runner scope: host runtime checks require the CPU-only route.
GATE_BACKENDS = ("cpu",)
import re
import os
import sys
from pathlib import Path

import mojolearn
from mojolearn._cpu_reference import reference_training
from mojolearn import _backend, host_surface

ROOT = Path(__file__).resolve().parents[3]

GP_ORACLE = "gaussian_process/host/gpr_oracle.mojo"
CHOL_ORACLE = "cholesky/host/chol_oracle.mojo"
LANES = ("gp", "gp-matern12", "gp-matern32", "gp-matern52-ard")
GPU_IMPORTS = re.compile(r"^\s*from\s+(max\.gpu|std\.gpu)", re.M)


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_manifest_declares_the_gp_family():
    fam = host_surface.family("gp")
    assert fam["routes"] == "_mojolearn_gp"
    # The cholesky lane joined this family on lane/cpu-training-d-estimators
    # (ad7a1b933) and moved to the linalg family with public CPU Cholesky
    # inference; gp-normalize-y joined on lane/cpu-training-small-gaps and the
    # classifier's lanes on lane/gaussian-process-classifier
    # (tests/test_gpc_surface.py).
    assert fam["training_lanes"] == LANES + ("gp-normalize-y", "gpc", "gpc-multiclass",
        "gp-sample-y", "gp-sample-y-normalize", "gp-optimize", "gp-optimize-restarts",
        "par-gpc-fit", "par-gpc-predict")
    for module in (GP_ORACLE, CHOL_ORACLE):
        assert module in fam["host_modules"] and (ROOT / module).is_file()
    assert (ROOT / host_surface.build_shim("gp")).is_file()
    assert (ROOT / host_surface.binding_source("gp")).is_file()
    assert host_surface.routed_modules()["_mojolearn_gp"] == "_mojolearn_gp_host"
    for lane in LANES:
        assert lane in host_surface.covered_lanes(), f"{lane} is not a covered training lane"
    assert "Gaussian process" not in host_surface.no_cpu_path_sentence(), host_surface.no_cpu_path_sentence()
    assert "the Gaussian process with an RBF kernel" in host_surface.training_sentence()


def test_binding_registers_the_gpu_names():
    src = _read(host_surface.binding_source("gp"))
    exports = host_surface.family("gp")["exports"]
    gpu = _read("bindings/_mojolearn_gp.mojo")
    for name in ("gpr_fit", "gpr_predict", "cholesky_profile_jitter", "cholesky_factor",
                 "cholesky_solve", "gp_vendor", "gp_numeric_mode"):
        assert f'("{name}")' in src, f"the gp host binding does not register {name}"
        assert f'("{name}")' in gpu, f"{name} is not a GPU binding name"
        assert name in exports, f"the manifest does not list {name} for gp"
    assert '("gp_parallel_available")' not in src, "gp_parallel_available must stay absent so it refuses by name"
    # The address and params contract, word for word. The host predict entries
    # live in bindings/gp_host_predict.mojo since lane/inference-neighbors-density
    # moved them there (the gp host binding imports and registers them), so the
    # host side of the contract is that module and the binding read together.
    host = src + _read("bindings/gp_host_predict.mojo")
    for sentence in ("addrs must contain 9 addresses", "params must contain 5 values",
                     "addrs must contain 12 addresses", "params must contain 7 values",
                     "params must contain 6 values (n, nrhs, info,"):
        assert sentence in host and sentence in gpu, sentence


def test_oracles_import_no_gpu_and_no_device_module():
    gp = _read(GP_ORACLE)
    chol = _read(CHOL_ORACLE)
    for rel, text in ((GP_ORACLE, gp), (CHOL_ORACLE, chol)):
        assert not GPU_IMPORTS.search(text), f"{rel} imports a GPU module"
        assert not re.search(r"^\s*from .*import.*DeviceContext", text, re.M), f"{rel} imports DeviceContext"
    assert sorted(set(re.findall(r"^from\s+([\w.]+)\s+import", gp, re.M))) == [
        "checks.numerics", "cholesky.host.chol_oracle",
        "core.host_predict_threads", "gemm.host.identical_gemm",
        "max.algorithm", "std.memory", "std.sys.compile",
    ]
    assert sorted(set(re.findall(r"^from\s+([\w.]+)\s+import", chol, re.M))) == [
        "checks.numerics", "gemm.host.identical_gemm", "std.memory",
    ]


def test_oracles_spell_the_bit_carrying_constructs():
    chol = _read(CHOL_ORACLE)
    assert "comptime CHOL_HOST_NB_PINNED = 32" in chol
    assert "comptime CHOL_HOST_JITTER_BITS: UInt32 = 0x35800000" in chol
    assert "if not (s > Float32(0.0)):" in chol, "the pivot is spelled not (s > 0)"
    assert "a[jc * n + jc] = ftz(identical_sqrt(s))" in chol, "the root goes through the IDENTICAL sqrt seam"
    assert "a[at] = ftz(cur - upd)" in chol, "the trailing update subtracts the lower triangle"
    assert "acc = ftz(acc + ftz(identical_log(ftz(a[j * n + j]))))" in chol
    assert "logdet = ftz(identical_mul(Float32(2.0), acc))" in chol
    gp = _read(GP_ORACLE)
    assert "comptime GPR_LOG_2PI_BITS: UInt32 = 0x3FEB3F8E" in gp
    assert "return ftz(ftz(t1 + t2) + t3)" in gp, "the lml adds t1 and t2 first"
    assert "if is_self and i == j:" in gp, "the white kernel is a structural test"
    assert "if not (raw > Float32(0.0)):" in gp, "the variance clamp is a comparison, never a max"
    assert "gemm_oracle(kcross, dual, OP_TN, n_star, 1, n_train)" in gp
    assert "var third = ftz(identical_div(ss, Float32(3.0)))" in gp, "K**2 / 3 is a divide"
    assert "sync_parallelize(_rbf_rows, tasks)" in gp
    assert "sync_parallelize(_matern_rows, tasks)" in gp
    assert "slotp.unsafe_store(i * n + j" in gp, "parallel rows must own disjoint cells"


def test_sabotage_define_moves_the_distance():
    text = _read(GP_ORACLE)
    define = host_surface.sabotage_define("gp")
    assert define == "MOJOLEARN_HOST_SABOTAGE"
    assert f'is_defined["{define}"]()' in text
    assert "comptime if GPR_ORACLE_HOST_SABOTAGE:" in text
    assert "f = d - 1 - q" in text, "the sabotage arm does not walk the feature axis descending"
    assert "GPR_ORACLE_HOST_SABOTAGE" in _read(host_surface.binding_source("gp"))



@reference_training()
def test_gaussian_process_fits_on_the_host_when_built():
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return
    if "_mojolearn_gp_host" not in _backend.host_families_built():
        print("SKIP: the gp host binding is not built")
        return
    import numpy as np
    module = _backend.load_host_module("_mojolearn_gp_host")
    assert not bool(module.gp_host_sabotage()), "a sabotage build loaded outside the gate"
    rng = np.random.default_rng(0)
    x = rng.standard_normal((96, 4)).astype(np.float32)
    y = (x[:, 0] - 0.5 * x[:, 2]).astype(np.float32)
    xs = rng.standard_normal((16, 4)).astype(np.float32)
    kernels = (
        lambda: mojolearn.ConstantKernel(1.0) * mojolearn.RBF(1.0) + mojolearn.WhiteKernel(0.1),
        lambda: mojolearn.ConstantKernel(1.0) * mojolearn.Matern(1.0, nu=0.5) + mojolearn.WhiteKernel(0.1),
        lambda: mojolearn.ConstantKernel(1.0) * mojolearn.Matern(1.0, nu=1.5) + mojolearn.WhiteKernel(0.1),
        lambda: mojolearn.ConstantKernel(1.0) * mojolearn.Matern([1.0, 2.0, 0.5, 4.0], nu=2.5)
        + mojolearn.WhiteKernel(0.1),
    )
    previous_threads = os.environ.get("MOJOLEARN_CPU_THREADS")
    try:
        for make in kernels:
            outs = []
            # One and several row tasks must be the same numerical path.
            for threads in ("1", "4"):
                os.environ["MOJOLEARN_CPU_THREADS"] = threads
                m = mojolearn.GaussianProcessRegressor(kernel=make()).fit(x, y)
                assert m.info_ == 0, m.info_
                mean, std = m.predict(xs, return_std=True)
                outs.append((np.asarray(m.L_).tobytes(), np.asarray(m.alpha_).tobytes(),
                             np.float64(m.log_marginal_likelihood_value_).tobytes(),
                             np.asarray(mean).tobytes(), np.asarray(std).tobytes()))
                assert np.isfinite(m.log_marginal_likelihood_value_)
            assert outs[0] == outs[1], "GP bytes moved with CPU row task count"
    finally:
        if previous_threads is None:
            os.environ.pop("MOJOLEARN_CPU_THREADS", None)
        else:
            os.environ["MOJOLEARN_CPU_THREADS"] = previous_threads


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
