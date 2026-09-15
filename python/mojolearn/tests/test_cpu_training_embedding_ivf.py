# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU training for the embedding, embedding-sort, ivf and ivf-euclidean lanes
(lane/cpu-training-embedding-ivf, 2026-09-15), checked from SOURCE so it runs
on a box with nothing built, plus runtime checks that run only where the two
host bindings are built and the package took the CPU-only path.

What the source checks hold: the manifest declares the embedding and ivf
families, routes `_mojolearn_embedding` and `_mojolearn_ivf` to them, covers
the four lanes, puts them in the set diffed against their own records
(TRAINING_FIX_LANES) and not the 166-lane record's, and no longer names the
Embedding layer as having no CPU path; each host binding registers exactly
the GPU binding's names plus its four read-backs; the two host modules import
no GPU module and do not import the device entry points, and they call the
host stages the device path already runs on the host; the IVF host selection
limit is the device selector's literal; the sabotage define reaches both
modules and each binding reads it back; the CPU identity gate triggers on the
host modules and checks out both records.

The runtime checks (skipped, and SAID to be skipped, when a binding is absent
or a GPU set loaded): the host forward is the gather; the host backward is
the ascending sum with the padding row stored +0.0, the same bytes under both
plans, and a carried microbatch equals the unsplit gradient; an out-of-range
id refuses by name; an IVF search at n_probes == n_lists returns the brute
force neighbours on distinct distances, the euclidean metric returns the root
of the squared one on the same ids, and n_probes > n_lists refuses by name.
The bit claim against the GPU columns is the CPU identity gate's.

    cd python && python3 -m mojolearn.tests.test_cpu_training_embedding_ivf
"""
import re
import sys
from pathlib import Path

import mojolearn
from mojolearn import _backend, host_surface

ROOT = Path(__file__).resolve().parents[3]

EMB_LANES = ("embedding", "embedding-sort")
IVF_LANES = ("ivf", "ivf-euclidean")
EMB_HOST = "embedding/host/embedding_host.mojo"
IVF_HOST = "ivf/host/ivf_host.mojo"
GPU_IMPORTS = re.compile(r"^\s*from\s+(max\.gpu|std\.gpu)", re.M)
REGISTERED = re.compile(r'def_function\[\w+\]\(\s*"(\w+)"\s*\)')


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def _registered(rel):
    return set(REGISTERED.findall(_read(rel)))


def test_manifest_declares_both_families():
    for name, route, lanes in (("embedding", "_mojolearn_embedding", EMB_LANES),
                               ("ivf", "_mojolearn_ivf", IVF_LANES)):
        fam = host_surface.family(name)
        assert fam["routes"] == route
        assert host_surface.routed_modules()[route] == f"{route}_host"
        assert fam["training_lanes"] == lanes
        assert fam["ships_in_wheel"]
        assert (ROOT / host_surface.build_shim(name)).is_file()
        for lane in lanes:
            assert lane in host_surface.covered_lanes()
            assert lane in host_surface.fix_covered_lanes(), f"{lane} is not diffed against its own record"
            assert lane not in host_surface.record_covered_lanes()
            assert host_surface.TRAINING_LANE_NAMES[lane] in host_surface.training_sentence()
    sentence = host_surface.no_cpu_path_sentence()
    assert "Embedding" not in sentence, sentence


def test_bindings_register_the_gpu_names():
    for name in ("embedding", "ivf"):
        host = _registered(host_surface.binding_source(name))
        gpu = _registered(f"bindings/_mojolearn_{name}.mojo")
        readbacks = {f"{name}_host_{r}" for r in host_surface.READBACK}
        assert host == gpu | readbacks, (name, sorted(host ^ (gpu | readbacks)))
        assert host == set(host_surface.family(name)["exports"])


def test_host_modules_are_host_only():
    for rel, forbidden in ((EMB_HOST, ("embedding.checks.embedding_identical", "embedding.checks.embedding_sort")),
                           (IVF_HOST, ("ivf.estimator", "ivf.impl.neighbors.ivf_flat.ivf_flat_build",
                                       "ivf.impl.neighbors.ivf_flat.ivf_flat_search",
                                       "neighbors.checks", "core.row_norms"))):
        text = _read(rel)
        assert not GPU_IMPORTS.search(text), f"{rel} imports a GPU module"
        assert "DeviceContext" not in re.findall(r"^\s*from .*import (.*)$", text, re.M)
        for module in forbidden:
            assert not re.search(rf"^\s*from {re.escape(module)} import", text, re.M), (rel, module)
    for rel in ("ivf/checks/list_layout.mojo", "ivf/impl/neighbors/ivf_common.mojo",
                "ivf/impl/neighbors/ivf_flat/ivf_flat_index.mojo", "cluster/impl/kmeans_params.mojo"):
        assert not GPU_IMPORTS.search(_read(rel)), f"{rel} now imports a GPU module; the ivf host binding imports it"
    ivf = _read(IVF_HOST)
    for stage in ("host_fit_main", "build_list_layout", "merge_probed_lists", "calc_chunk_indices",
                  "postprocess_neighbors", "postprocess_distances", "ivf_index_params_validate",
                  "ivf_search_params_validate", "ivf_validate_data", '"ivf.quantizer."'):
        assert stage in ivf, f"{IVF_HOST} does not call {stage}"


def test_ivf_select_limit_is_the_device_literal():
    host = re.search(r"^comptime IVF_HOST_SELECT_LIMIT = (\d+)$", _read(IVF_HOST), re.M)
    dev = re.search(r"^comptime IDENTICAL_MAX_K = (\d+)$", _read("neighbors/checks/select_radix_identical.mojo"), re.M)
    assert host and dev and host.group(1) == dev.group(1), (host, dev)
    assert "IVF_SELECT_LIMIT = IDENTICAL_MAX_K if GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL" in _read(
        "ivf/impl/neighbors/ivf_flat/ivf_flat_search.mojo")


def test_embedding_plan_codes_are_the_device_literals():
    host = _read(EMB_HOST)
    dev = _read("embedding/checks/embedding_sort.mojo")
    for name, value in (("PLAN_SCAN", "0"), ("PLAN_SORT", "1")):
        assert re.search(rf"^comptime {name} = {value}$", dev, re.M), name
        assert re.search(rf"^comptime HOST_{name} = {value}$", host, re.M), name
    assert "0xFFFFFFFFFFFFFFFF" in dev and "0xFFFFFFFFFFFFFFFF" in host


def test_sabotage_reaches_both_and_reads_back():
    for name, rel, flag in (("embedding", EMB_HOST, "EMBEDDING_HOST_SABOTAGE"), ("ivf", IVF_HOST, "IVF_HOST_SABOTAGE")):
        define = host_surface.sabotage_define(name)
        text = _read(rel)
        assert f'is_defined["{define}"]()' in text
        assert flag in _read(host_surface.binding_source(name))


def test_workflow_triggers_and_checks_out_the_records():
    wf = _read(".github/workflows/cpu-identity-gate.yml")
    for rel in (EMB_HOST, IVF_HOST, "ivf/checks/list_layout.mojo", "python/mojolearn/embedding.py",
                "python/mojolearn/_ivf_impl.py"):
        assert f'- "{rel}"' in wf, rel
    for record in ("2026-09-15_embedding-sort", "2026-09-14_ivf-euclidean"):
        assert f"/bench/results/identity_break/{record}/" in wf, record
        assert f'- "bench/results/identity_break/{record}/**"' in wf, record


def _cpu_only_with(basename):
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return False
    if basename not in _backend.host_families_built():
        print(f"SKIP: {basename} is not built")
        return False
    return True


def test_embedding_runs_on_the_host_when_built():
    if not _cpu_only_with("_mojolearn_embedding_host"):
        return
    import numpy as np
    module = _backend.load_host_module("_mojolearn_embedding_host")
    assert not bool(module.embedding_host_sabotage()), "a sabotage build loaded outside the gate"
    assert str(module.embedding_host_column()) == "cpu"
    rng = np.random.default_rng(4)
    V, D, T = 11, 5, 37
    w = rng.standard_normal((V, D)).astype(np.float32)
    ids = rng.integers(0, V, T).astype(np.int32)
    dy = rng.standard_normal((T, D)).astype(np.float32)
    assert np.asarray(mojolearn.Embedding(V, D, weight=w).forward(ids)).tobytes() == w[ids].tobytes()
    want = np.zeros((V, D), dtype=np.float32)
    for t in range(T):
        want[ids[t]] += dy[t]
    want[2] = 0.0
    got = {}
    for plan in ("scan", "sort"):
        e = mojolearn.Embedding(V, D, padding_idx=2, weight=w, plan=plan)
        full = np.asarray(e.backward(ids, dy))
        carried = np.asarray(e.backward(ids[13:], dy[13:], grad=e.backward(ids[:13], dy[:13])))
        assert carried.tobytes() == full.tobytes(), plan
        assert full.tobytes() == want.tobytes(), plan
        assert np.signbit(full[2]).sum() == 0, "the padding row is not +0.0"
        got[plan] = full.tobytes()
    assert got["scan"] == got["sort"]
    bad = ids.copy()
    bad[5] = V
    try:
        mojolearn.Embedding(V, D, weight=w).forward(bad)
    except Exception as exc:
        assert "outside [0, 11) REFUSED" in str(exc), str(exc)
    else:
        raise AssertionError("an out-of-range id was gathered")


def test_ivf_runs_on_the_host_when_built():
    if not _cpu_only_with("_mojolearn_ivf_host"):
        return
    import numpy as np
    module = _backend.load_host_module("_mojolearn_ivf_host")
    assert not bool(module.ivf_host_sabotage()), "a sabotage build loaded outside the gate"
    rng = np.random.default_rng(5)
    x = rng.standard_normal((300, 6)).astype(np.float32)
    q = rng.standard_normal((9, 6)).astype(np.float32)
    k = 5
    m = mojolearn.IVFIndex(n_lists=4, n_probes=4, n_neighbors=k, random_state=1).fit(x)
    d2, i2 = m.search(q)
    assert np.asarray(m.n_candidates_).tolist() == [300] * 9
    exact = ((q[:, None, :].astype(np.float64) - x[None, :, :]) ** 2).sum(-1)
    assert np.asarray(i2).tolist() == np.argsort(exact, axis=1, kind="stable")[:, :k].tolist()
    e = mojolearn.IVFIndex(n_lists=4, n_probes=4, n_neighbors=k, metric="euclidean", random_state=1).fit(x)
    d1, i1 = e.search(q)
    assert np.asarray(i1).tolist() == np.asarray(i2).tolist()
    assert np.allclose(np.asarray(d1), np.sqrt(np.asarray(d2)), rtol=1e-6)
    try:
        mojolearn.IVFIndex(n_lists=4, n_probes=5, random_state=1).fit(x).search(q)
    except Exception as exc:
        assert "exceeds n_lists (4)" in str(exc), str(exc)
    else:
        raise AssertionError("n_probes > n_lists was clamped")


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
