# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Workstream E, CPU training for the knn, pca, pca-whiten, tsvd, ols,
ridge and dbscan lanes (lane/cpu-training-e, 2026-09-14), checked from SOURCE so it
runs on a box with nothing built, plus one runtime check that runs only
where the estimators host binding is built and the package took the
CPU-only path.

What the source checks hold: the manifest lists the six lanes as training
lanes of the families that serve them and names them for the docs; the
estimators host binding registers `pca_fit`, `tsvd_fit`, `ols_fit` and
`ridge_fit` under the GPU binding's names; the three host oracles import no
GPU module (a host restatement that imports `max.gpu` or `std.gpu` is not
host only); the sabotage define reaches the PCA oracle, whose reduce is
the arm that moves every PCA, tSVD, OLS and ridge bit, and the DBSCAN
oracle's core test; the CPU identity gate runs by hand since 2026-09-15 (no push trigger).

The runtime check (skipped, and SAID to be skipped, when the binding is
absent or a GPU set loaded): the four entries run through the host
binding on a small draw, twice, and return the same bytes; the binding
reads back sabotage False. It is a plumbing check. The bit claim against
the GPU columns is the CPU identity gate's, not this file's.

    cd python && python3 -m mojolearn.tests.test_cpu_training_e
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

LANES_E = ("knn", "knn-clf", "knn-reg", "pca", "pca-whiten", "tsvd", "ols", "ridge", "dbscan")
FITS_E = ("pca_fit", "tsvd_fit", "ols_fit", "ridge_fit", "dbscan_fit")
ORACLES_E = ("decomposition/host/pca_oracle.mojo", "glm/host/glm_oracle.mojo",
             "dbscan/host/dbscan_oracle.mojo", "decomposition/host/pca_full_oracle.mojo")
GPU_IMPORTS = re.compile(r"^\s*from\s+(max\.gpu|std\.gpu)", re.M)


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_manifest_covers_the_six_lanes():
    covered = host_surface.covered_lanes()
    for lane in LANES_E:
        assert lane in covered, f"{lane} is not a covered training lane"
        assert lane in host_surface.TRAINING_LANE_NAMES, f"{lane} has no doc name"
    core = host_surface.family("core")
    for lane in ("knn", "knn-clf", "knn-reg"):
        assert lane in core["training_lanes"] and lane in core["inference_lanes"]
    est = host_surface.family("estimators")
    for lane in ("pca", "pca-whiten", "tsvd", "ols", "ridge", "dbscan"):
        assert lane in est["training_lanes"], f"{lane} is not an estimators training lane"
    for rel in ORACLES_E:
        assert rel in est["host_modules"], f"{rel} is not an estimators host module"
        assert (ROOT / rel).is_file(), f"{rel} does not exist"


def test_binding_registers_the_four_fits():
    src = _read(host_surface.binding_source("estimators"))
    exports = host_surface.family("estimators")["exports"]
    for name in FITS_E:
        assert f'("{name}")' in src, f"the estimators host binding does not register {name}"
        assert name in exports, f"the manifest does not list {name}"
    # pca_fit_full (the pca-full-whiten lane) and qn_fit (batch 2) are
    # registered now; inverse_transform stays absent.
    for absent in ("inverse_transform",):
        assert f'("{absent}")' not in src, f"{absent} must stay absent so it refuses by name"


def test_core_host_carries_the_centering_helpers():
    """Run 34869406147: ols and ridge REFUSED on every runner at
    `_mojolearn.column_mean_f64` because the Python centering step reaches
    the base binding through `_buffer._native`, which on a CPU-only install
    is the core host binding. The three helpers must be registered there
    and listed in the manifest."""
    src = _read(host_surface.binding_source("core"))
    exports = host_surface.family("core")["exports"]
    for name in ("column_mean_f64", "center_columns_f32", "scale_rows_f32"):
        assert f'("{name}")' in src, f"the core host binding does not register {name}"
        assert name in exports, f"the manifest does not list {name} for core"
        assert f"def {name}_binding(" in _read("bindings/host_helpers.mojo"), f"{name} has no host body"


def test_oracles_import_no_gpu():
    for rel in ORACLES_E:
        text = _read(rel)
        assert not GPU_IMPORTS.search(text), f"{rel} imports a GPU module"
        assert not re.search(r"^\s*from .*import.*DeviceContext", text, re.M), f"{rel} imports DeviceContext"
        assert "from checks.numerics import" in text, f"{rel} does not import the numerics leaves"


def test_sabotage_define_moves_the_gram_reduce():
    text = _read("decomposition/host/pca_oracle.mojo")
    define = host_surface.sabotage_define("estimators")
    assert f'is_defined["{define}"]()' in text
    assert "comptime if PCA_ORACLE_HOST_SABOTAGE:" in text, "the sabotage arm is not a comptime branch"
    assert "GRAM_SPLITK_CHUNKS - 1 - cc" in text, "the sabotage arm does not walk the chunks descending"
    binding = _read(host_surface.binding_source("estimators"))
    assert "PCA_ORACLE_HOST_SABOTAGE" in binding, "the binding does not read the PCA oracle's define back"
    dbscan = _read("dbscan/host/dbscan_oracle.mojo")
    assert "comptime if DBSCAN_ORACLE_HOST_SABOTAGE:" in dbscan
    assert "labels[0] += 1" in dbscan, "the DBSCAN sabotage arm must move even an all-noise result"
    assert "DBSCAN_ORACLE_HOST_SABOTAGE" in binding



def test_readme_no_longer_says_knn_training_has_no_cpu_path():
    sentence = host_surface.no_cpu_path_sentence()
    assert "k-NN" not in sentence, sentence
    assert "DBSCAN" not in sentence, sentence
    # logistic regression trains on the host since batch 2, so the sentence
    # no longer names it.
    assert "logistic regression" not in sentence, sentence
    # The README states the claim at a high level; SUPPORT_MATRIX.md carries
    # the CPU surface in full.
    for rel in ("SUPPORT_MATRIX.md",):
        text = _read(rel)
        m = re.search(r"<!--fact:no_cpu_path-->(.*?)<!--/fact-->", text, re.S)
        assert m, f"{rel} has no no_cpu_path span"
        assert "k-NN" not in m.group(1), f"{rel} still says k-NN training has no CPU path"


@reference_training()
def test_host_fits_run_and_repeat_on_a_cpu_only_install():
    """Runtime, only where it can run. Skipping is stated, never silent."""
    if _backend._CPU_ONLY is None:
        print("  skip: a GPU set loaded; the host fits are served on a CPU-only install only")
        return
    if "_mojolearn_estimators_host" not in _backend.host_families_built():
        print("  skip: _mojolearn_estimators_host is not built under", _backend.host_dir())
        return
    try:
        import numpy as np
    except ImportError:
        print("  skip: numpy is not installed")
        return
    m = _backend.load_host_module("_mojolearn_estimators_host")
    assert bool(m.estimators_host_sabotage()) is False
    rng = np.random.default_rng(7)
    n, d, k = 512, 5, 2
    x = rng.standard_normal((n, d)).astype(np.float32)
    y = rng.standard_normal(n).astype(np.float32)

    def addr(a):
        return a.ctypes.data

    def run():
        comp = np.empty((k, d), np.float32)
        mu = np.empty(d, np.float32)
        ev = np.empty(k, np.float32)
        ratio = np.empty(k, np.float32)
        sv = np.empty(k, np.float32)
        noise = float(m.pca_fit(addr(x), addr(comp), addr(mu), addr(ev), addr(ratio), addr(sv), [n, d, k]))
        tcomp = np.empty((k, d), np.float32)
        tsv = np.empty(k, np.float32)
        m.tsvd_fit(addr(x), addr(tcomp), addr(tsv), [n, d, k])
        w = np.empty(d, np.float32)
        m.ols_fit(addr(x), addr(y), addr(w), [n, d])
        r = np.empty(d, np.float32)
        m.ridge_fit(addr(x), addr(y), addr(r), [n, d, 0.5])
        labels = np.empty(n, np.int32)
        passes = int(m.dbscan_fit(addr(x), addr(labels), 0, [n, d, 0.9, 5, 0, 0, 1, 0]))
        assert passes >= 1 and labels.min() >= -1
        return comp.tobytes() + mu.tobytes() + ev.tobytes() + ratio.tobytes() + sv.tobytes() + \
            repr(noise).encode() + tcomp.tobytes() + tsv.tobytes() + w.tobytes() + r.tobytes() + \
            labels.tobytes()

    first, second = run(), run()
    assert first == second, "a host fit returned different bytes on two runs"
    assert np.isfinite(np.frombuffer(first[: k * d * 4], np.float32)).all()
    # The shape refusals are pca_validate's and reach the caller by name.
    try:
        m.pca_fit(addr(x), addr(np.empty((k, d), np.float32)), addr(np.empty(d, np.float32)),
                  addr(np.empty(k, np.float32)), addr(np.empty(k, np.float32)),
                  addr(np.empty(k, np.float32)), [n, d, d + 1])
    except Exception as exc:  # noqa: BLE001
        assert "n_components cannot exceed n_cols" in str(exc), str(exc)
    else:
        raise AssertionError("pca_fit accepted n_components > n_cols")


TESTS = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_") and callable(fn)]


def main(argv=None):
    failures = []
    for name, fn in TESTS:
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures.append((name, f"{type(exc).__name__}: {exc}"))
    for name, why in failures:
        print(f"FAIL {name}: {why}")
    if failures:
        print(f"test_cpu_training_e: RED. {len(failures)} of {len(TESTS)} checks failed.")
        return 1
    print(f"test_cpu_training_e: GREEN. {len(TESTS)} checks (vendor {mojolearn.vendor()}).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
