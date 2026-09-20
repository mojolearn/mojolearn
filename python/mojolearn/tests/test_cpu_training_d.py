# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU training for the workstream D estimators
(lane/cpu-training-d-estimators, 2026-09-15): cholesky, rbf-sampler,
kernel-ridge, nystroem, gmm, gmm-random-init, hdbscan and hdbscan-leaf.
Checked from SOURCE so it runs on a box with nothing built, plus runtime
checks that run only where a host binding is built and the package took the
CPU-only path.

What the source checks hold: the manifest covers the eight lanes in the gp,
kernel_methods, mixture and hdbscan families; each new host binding
registers the GPU binding's fit and scoring names and NOT the multi-GPU
probe; each new host oracle imports no DeviceContext and names the device
statements whose spelling carries the bits; the Cholesky door keeps its
validation while the kernel ridge solve and the mixture's precision Cholesky
reach the unvalidated factorization, as `potrf_lower` does on the device;
each family's sabotage define is read where its binding compiles; the CPU identity gate runs by hand since 2026-09-15 (no push trigger).

The runtime checks (skipped, and SAID to be skipped, when a binding is absent
or a GPU set loaded) fit each estimator twice through the host binding and
require the same bytes. They are plumbing checks; the bit claim against the
GPU columns is the CPU identity gate's, not this file's.

    cd python && python3 -m mojolearn.tests.test_cpu_training_d
"""

# Gate-runner scope: host runtime checks require the CPU-only route.
GATE_BACKENDS = ("cpu",)
import re
import sys
from pathlib import Path

import mojolearn
from mojolearn._cpu_reference import reference_training
from mojolearn import _backend, host_surface

ROOT = Path(__file__).resolve().parents[3]

LANES = {
    # The linalg family since lane/inference-embedding-ivf-cholesky
    # (2026-09-15): its host binding ships, so public CPU Cholesky does.
    "linalg": ("cholesky",),
    "kernel_methods": ("rbf-sampler", "kernel-ridge", "nystroem"),
    "mixture": ("gmm", "gmm-random-init"),
    "hdbscan": ("hdbscan", "hdbscan-leaf"),
}

ORACLES = {
    "kernel_methods": "kernel_methods/host/km_host_oracle.mojo",
    "mixture": "mixture/host/gmm_host_oracle.mojo",
    "hdbscan": "hdbscan/host/hdbscan_host_oracle.mojo",
}

#: The GPU binding names each new host binding must register, and the
#: multi-GPU probe it must not.
NAMES = {
    "kernel_methods": (
        ("kernel_ridge_fit", "kernel_ridge_predict", "nystroem_fit", "nystroem_transform",
         "rbf_sampler_fit", "rbf_sampler_transform", "kernel_methods_vendor",
         "kernel_methods_numeric_mode"),
        "kernel_methods_rows_parallel_available",
    ),
    "mixture": (
        ("gmm_fit", "gmm_score_samples", "gmm_predict_proba", "gmm_predict",
         "gmm_score_bic_aic", "mixture_vendor", "mixture_numeric_mode"),
        "gmm_parallel_available",
    ),
    "hdbscan": (
        ("hdbscan_fit", "hdbscan_vendor", "hdbscan_numeric_mode"),
        "hdbscan_rows_parallel_available",
    ),
}

#: (device file, device spelling, oracle spelling) triples whose arithmetic
#: carries the bits; the oracle spelling differs only in how a buffer cell is
#: addressed (a pointer load on the device, a list index on the host).
STATEMENTS = {
    "kernel_methods": (
        ("svm/impl/distance/kernel_matrices.mojo", "var e = ftz((-gain) * s)", "var e = ftz((-gain) * s)"),
        ("kernel_methods/impl/kernel_ridge/kernel_ridge.mojo", "ftz(d + alpha)", "ftz(dv + alpha)"),
        ("kernel_methods/checks/random_features.mojo", "var shifted = ftz(p + ftz(b_in.unsafe_load(j)))",
         "var shifted = ftz(pv + ftz(offset[j]))"),
        ("kernel_methods/estimator.mojo", "vt_ord[f * q + c] = -e if negative else e",
         "vt_ord[f * q + c] = -e if negative else e"),
    ),
    "mixture": (
        ("mixture/checks/estep.mojo", "var half = ftz(identical_mul(Float32(-0.5), inner))",
         "var half = ftz(identical_mul(Float32(-0.5), inner))"),
        ("mixture/checks/estep.mojo", "s + ftz(identical_exp(ftz(wlp.unsafe_load(base + k) - max_exp)))",
         "s + ftz(identical_exp(ftz(wlp[base + k] - max_exp)))"),
        ("mixture/checks/mstep.mojo", "nk.unsafe_store(k, ftz(acc + ten_eps))", "nk[k] = ftz(acc + ten_eps)"),
        ("mixture/estimator.mojo", "resp.append(ftz(row[k] / s))", "resp.append(ftz(row[k] / s))"),
    ),
    "hdbscan": (
        ("hierarchy/impl/sparse/solver/detail/mst_kernels.mojo", "if vertex_color > dst_color:",
         "if j_is_min and cu > cj:"),
        ("hdbscan/impl/detail/stabilities.mojo", "var term = ftz(lambdas.unsafe_load(i) - birth)",
         "var term = ftz(tree.lambdas[i] - birth)"),
        ("hdbscan/impl/detail/select.mojo", "or cluster_sizes[node] > max_cluster_size",
         "or cluster_sizes[node] > max_cluster_size"),
        ("hdbscan/impl/detail/condense.mojo", "lambda_value = identical_div(Float32(1.0), distance)",
         "lambda_value = identical_div(Float32(1.0), distance)"),
    ),
}


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def _registers(text, name):
    return re.search(r'def_function\[\w+\]\(\s*"' + re.escape(name) + r'"\s*\)', text) is not None


def _squash(text):
    return re.sub(r"\s+", " ", text)


def test_manifest_covers_the_eight_lanes():
    covered = host_surface.covered_lanes()
    for fam, lanes in LANES.items():
        declared = host_surface.family(fam)["training_lanes"]
        for lane in lanes:
            assert lane in declared, f"{lane} is not declared by the {fam} family"
            assert lane in covered and lane in host_surface.TRAINING_LANE_NAMES
    for fam, oracle in ORACLES.items():
        f = host_surface.family(fam)
        assert f["routes"] == f"_mojolearn_{fam}" and f["binding"] == f"_mojolearn_{fam}_host"
        assert oracle in f["host_modules"] and (ROOT / oracle).is_file()


def test_bindings_register_the_gpu_names_and_not_the_probe():
    for fam, (names, probe) in NAMES.items():
        src = _read(host_surface.binding_source(fam))
        gpu = _read(f"bindings/_mojolearn_{fam}.mojo")
        for name in names:
            assert _registers(src, name), f"the {fam} host binding does not register {name}"
            assert _registers(gpu, name), f"{name} is not a {fam} GPU binding name"
            assert name in host_surface.family(fam)["exports"]
        assert _registers(gpu, probe) and not _registers(src, probe), (
            f"{probe} must stay absent from the {fam} host binding so the multi-GPU driver refuses by name"
        )


def test_oracles_import_no_device_context():
    for oracle in ORACLES.values():
        text = _read(oracle)
        assert not re.search(r"^\s*from\s+(max\.gpu|std\.gpu)", text, re.M), f"{oracle} imports a GPU module"
        assert "DeviceContext" not in re.sub(r'"""(.|\n)*?"""', "", text), f"{oracle} names DeviceContext"


def test_oracles_spell_the_device_statements():
    for fam, triples in STATEMENTS.items():
        oracle = _squash(_read(ORACLES[fam]))
        for device, spelled, restated in triples:
            assert _squash(spelled) in _squash(_read(device)), (
                f"{device} no longer spells {spelled!r}; restate {ORACLES[fam]}"
            )
            assert _squash(restated) in oracle, f"{ORACLES[fam]} does not spell {restated!r}"


def test_gmm_scoring_parallelizes_components_not_numeric_folds():
    """The host speed path may schedule components, never split a cell fold."""
    text = _read(ORACLES["mixture"])
    estep = text[text.index("def gmmh_e_step("):text.index("def _collapse_message(")]
    assert "host_predict_task_count(ncomp)" in estep
    assert "parallel_components and n * d >= 1024" in estep
    assert "sync_parallelize(_components, component_tasks)" in estep
    assert "mahalp.unsafe_store(i * ncomp + kc, acc)" in estep
    # The complete feature fold stays inside one component worker, ascending.
    assert "for j in range(d):" in estep
    assert "acc = ftz(identical_mul_add(t, t, acc))" in estep


def test_unvalidated_factorization_where_the_device_skips_validation():
    chol = _read("cholesky/host/chol_oracle.mojo")
    assert "def chol_host_factor_lower(" in chol
    door = chol[chol.index("def chol_host_potrf("):chol.index("def chol_host_factor_lower(")]
    assert "chol_host_validate_matrix(a_in, n" in door and "return chol_host_factor_lower(a_in, n, jitter)" in door
    for oracle in (ORACLES["kernel_methods"], ORACLES["mixture"]):
        text = _read(oracle)
        assert "chol_host_factor_lower(" in text and "chol_host_potrf(" not in text, oracle


def test_sabotage_define_reaches_each_new_family():
    for fam in ORACLES:
        f = host_surface.family(fam)
        assert f["sabotage_define"] == "MOJOLEARN_HOST_SABOTAGE"
        texts = [_read(host_surface.binding_source(fam))] + [_read(m) for m in f["host_modules"]]
        assert any('is_defined["MOJOLEARN_HOST_SABOTAGE"]()' in t for t in texts), fam
    hdb = _read(ORACLES["hdbscan"])
    assert "comptime if HDBH_HOST_SABOTAGE:" in hdb and "slot = slot - 1" in hdb



def _built(basename):
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return False
    if basename not in _backend.host_families_built():
        print(f"SKIP: {basename} is not built")
        return False
    return True


def _twice(fit):
    a, b = fit(), fit()
    assert a == b, "two host fits returned different bytes"


@reference_training()
def test_estimators_fit_on_the_host_when_built():
    import numpy as np
    rng = np.random.default_rng(0)
    x = rng.standard_normal((96, 4)).astype(np.float32)
    y = rng.standard_normal(96).astype(np.float32)
    if _built("_mojolearn_gp_host"):
        a = (x.T @ x + 4.0 * np.eye(4)).astype(np.float32)
        _twice(lambda: np.asarray(mojolearn.Cholesky().fit(a).solve(np.ones((4, 1), np.float32))).tobytes())
    if _built("_mojolearn_kernel_methods_host"):
        _twice(lambda: np.asarray(mojolearn.KernelRidge(alpha=0.1, kernel="rbf", gamma=0.5).fit(x, y).predict(x[:8])).tobytes())
        _twice(lambda: np.asarray(mojolearn.Nystroem(gamma=0.5, n_components=8, random_state=1).fit(x).transform(x[:8])).tobytes())
        _twice(lambda: np.asarray(mojolearn.RBFSampler(gamma=0.5, n_components=8, random_state=1).fit(x).transform(x[:8])).tobytes())
    if _built("_mojolearn_mixture_host"):
        _twice(lambda: np.asarray(mojolearn.GaussianMixture(n_components=2, max_iter=5, random_state=3).fit(x).score_samples(x[:8])).tobytes())
    if _built("_mojolearn_hdbscan_host"):
        _twice(lambda: np.asarray(mojolearn.HDBSCAN(min_cluster_size=5).fit(x).labels_).tobytes())


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
