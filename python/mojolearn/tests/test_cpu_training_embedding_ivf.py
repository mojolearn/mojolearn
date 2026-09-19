# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU training for the embedding, embedding-sort, ivf, ivf-euclidean, byte-lm
and byte-lm-resident lanes (lane/cpu-training-embedding-ivf, 2026-09-15),
checked from SOURCE so it runs on a box with nothing built, plus runtime
checks that run only where the host bindings are built and the package took
the CPU-only path.

The byte LM half: the manifest declares the two trainer lanes on the byte_lm
family and ADAPTED_MODULES routes `_mojolearn_byte_lm` to
`_byte_lm_trainer_host`, which `_backend._cpu_only_binding` reads; the adapter
holds no arithmetic (it calls only the CPU byte LM binding's step, loss and
logits); the CPU byte LM binding's sabotage read-back reports the host GEMM
oracle's arm. At runtime: a stateless and a resident trainer take the same
three steps to the same loss, parameters and gradient bytes; the trainer's
logits are the CPU binding's reference-path logits; an evaluation changes
no state; a checkpoint round trip restores the bytes; a session rollback
restores the last step's shadow once; a multi-GPU entry is absent by name.

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

# Gate-runner scope: host runtime checks require the CPU-only route.
GATE_BACKENDS = ("cpu",)
import re
import sys
from pathlib import Path

import mojolearn
from mojolearn._cpu_reference import reference_training
from mojolearn import _backend, host_surface

ROOT = Path(__file__).resolve().parents[3]

EMB_LANES = ("embedding", "embedding-sort")
IVF_LANES = ("ivf", "ivf-euclidean", "ivf-extend")
#: par-ivf joined the ivf family on 2026-09-19 (lane/laneless-public-classes):
#: DistributedIVFIndex over the same host binding, through the
#: `ivf_flat_partial_search` and `ivf_finalize_distances` entries that lane
#: added to it. It is a multi-device driver, so it is named apart from the
#: three single-index lanes this file is about, and the family assertion
#: below names it rather than widening to "contains".
IVF_DRIVER_LANES = ("par-ivf",)
#: The lanes a committed record of their own carries; ivf-extend (stage 2 of
#: lane/inference-embedding-ivf-cholesky) is in no record and is OWED against
#: the training record.
RECORDED_OWN = ("embedding", "embedding-sort", "ivf", "ivf-euclidean")
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
        assert fam["training_lanes"] == lanes + (IVF_DRIVER_LANES if name == "ivf" else ())
        # Ships since lane/ship-cpu-host-families (2026-09-16): a wheel that
        # leaves the fit out cannot check the lane that hashes the fit.
        assert fam["ships_in_wheel"], "a host family with covered lanes must ship, or the lanes are unverifiable"
        assert (ROOT / host_surface.build_shim(name)).is_file()
        for lane in lanes:
            assert lane in host_surface.covered_lanes()
            if lane not in RECORDED_OWN:
                assert lane in host_surface.record_covered_lanes()
                continue
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


def test_workflow_checks_out_the_records():
    # The CPU identity gate runs by hand since 2026-09-15 (no push trigger),
    # so only its sparse checkout must bring down the two records.
    wf = _read(".github/workflows/cpu-identity-gate.yml")
    for record in ("2026-09-15_embedding-sort", "2026-09-14_ivf-euclidean"):
        assert f"/bench/results/identity_break/{record}/" in wf, record


def _cpu_only_with(basename):
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return False
    if basename not in _backend.host_families_built():
        print(f"SKIP: {basename} is not built")
        return False
    return True


@reference_training()
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


@reference_training()
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


BYTE_LM_LANES = ("byte-lm", "byte-lm-resident")
ADAPTER = "python/mojolearn/_byte_lm_trainer_host.py"


def test_manifest_declares_the_byte_lm_trainer_lanes():
    fam = host_surface.family("byte_lm")
    assert set(BYTE_LM_LANES) <= set(fam["training_lanes"])
    assert "SmallByteLanguageModelTrainer" in fam["classes"]
    for lane in BYTE_LM_LANES:
        assert lane in host_surface.record_covered_lanes(), f"{lane} is not diffed against the record"
    assert host_surface.ADAPTED_MODULES == {"_mojolearn_byte_lm": dict(family="byte_lm", module="_byte_lm_trainer_host")}
    assert "_mojolearn_byte_lm" in _backend._MODULES
    assert "_mojolearn_byte_lm" not in host_surface.routed_modules()
    backend = _read("python/mojolearn/_backend.py")
    assert "host_surface.ADAPTED_MODULES.get(name)" in backend


def test_adapter_holds_no_arithmetic_and_calls_the_host_entries():
    text = _read(ADAPTER)
    imports = re.findall(r"^(?:from|import) (\S+)", text, re.M)
    assert set(imports) <= {"ctypes", "math", "operator", "struct", ".", "._bufcheck", "._buffer",
                            "._byte_lm_config"}, imports
    for entry in ("byte_lm_host_train_step", "byte_lm_host_loss", "byte_lm_host_logits"):
        assert f"self._host.{entry}(" in text, entry
    assert "_backend.load_host_module(HOST_BASENAME)" in text
    impl = _read("python/mojolearn/_byte_lm_impl.py")
    assert "is_cpu_trainer_binding(binding)" in impl
    src = _read(host_surface.binding_source("byte_lm"))
    assert "from gemm.host.identical_gemm import GEMM_ORACLE_HOST_SABOTAGE" in src
    assert "or ANY_BWD_SABOTAGE or GEMM_ORACLE_HOST_SABOTAGE)" in src


@reference_training()
def test_byte_lm_trainer_runs_on_the_host_when_built():
    if not _cpu_only_with("_mojolearn_byte_lm_host"):
        return
    import os
    import tempfile
    import numpy as np
    from mojolearn import _byte_lm_trainer_host as adapter
    module = _backend.load_host_module("_mojolearn_byte_lm_host")
    assert not bool(module.byte_lm_host_sabotage()), "a sabotage build loaded outside the gate"
    binding = _backend.binding("_mojolearn_byte_lm", "identical")
    assert adapter.is_cpu_trainer_binding(binding) and binding.byte_lm_vendor() == "cpu"
    assert getattr(binding, "byte_lm_parallel_create", None) is None
    try:
        binding.byte_lm_offload_step
    except AttributeError as exc:
        assert "no CPU implementation of _mojolearn_byte_lm.byte_lm_offload_step" in str(exc)
    else:
        raise AssertionError("the adapter served a multi-GPU entry")
    shape = mojolearn.ByteLanguageModelConfig()
    rng = np.random.default_rng(6)
    flat = rng.uniform(-0.125, 0.125, shape.n_total).astype(np.float32)
    ids = rng.integers(0, 256, (6, shape.length + 1)).astype(np.int32)
    schedule = {"dataset": "test"}
    s = mojolearn.SmallByteLanguageModelTrainer(flat, data_schedule=schedule)
    r = mojolearn.SmallByteLanguageModelTrainer(flat, data_schedule=schedule, resident=True)
    for k in range(3):
        a = s.train_step(ids[2 * k:2 * k + 2])
        b = r.train_step(ids[2 * k:2 * k + 2])
        assert a["loss"] == b["loss"]
    assert np.asarray(a["flat_gradients"]).tobytes() == np.asarray(r.export_gradients()["flat_gradients"]).tobytes()
    assert np.asarray(s.parameters_).tobytes() == np.asarray(r.parameters_).tobytes()
    # The reference path's logits, from the same module the adapter loaded
    # (LanguageModelInference reads its own path variable, not
    # MOJOLEARN_HOST_DIR).
    p = np.ascontiguousarray(np.asarray(s.parameters_))
    q = np.ascontiguousarray(ids[:2, :-1])
    want = np.zeros((2, shape.length, shape.vocab_size), dtype=np.float32)
    module.byte_lm_host_logits([p.ctypes.data, q.ctypes.data, want.ctypes.data], [2, shape.length],
                               list(shape.native_shape), 0, 1)
    assert np.asarray(s.logits(ids[:2, :-1])).tobytes() == want.tobytes()
    assert np.asarray(r.logits(ids[:2, :-1])).tobytes() == want.tobytes()
    before = np.asarray(s.parameters_).tobytes()
    loss = s.evaluate(ids[:2])
    assert np.isfinite(loss) and np.asarray(s.parameters_).tobytes() == before and s.step_ == 3
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ck.json")
        s.save_checkpoint(path)
        back = mojolearn.SmallByteLanguageModelTrainer.from_checkpoint(path)
        assert np.asarray(back.parameters_).tobytes() == before and back.step_ == 3
    from mojolearn._buffer import addr, addr_ro, zeros
    session = binding.byte_lm_session_create()
    st = s.state_dict()
    params = [0, 3, 2, st["config"]["lr"], st["config"]["beta1"], st["config"]["beta2"], st["config"]["eps"],
              st["config"]["weight_decay"], 0.0, 0.0, 0, 0.0]
    native = list(shape.native_shape)
    arrays = [st["parameters"], st["m"], st["v"], st["flags"]]
    assert binding.byte_lm_session_open(session, [addr_ro(x, name="s") for x in arrays], params, native) == 3
    assert binding.byte_lm_session_rollback(session) == 3, "a rollback with no step restored something"
    tok = np.ascontiguousarray(ids[:2])
    loss_out = zeros((1,), "<f4")
    flags_out = zeros((shape.n_tensors,), "<i4")
    step = list(params)
    step[0] = 1
    assert binding.byte_lm_session_step(session, [tok.ctypes.data, addr_ro(st["flags"], name="f"),
                                                  addr(loss_out, name="l"), addr(flags_out, name="f")],
                                        step, native) == 4
    assert binding.byte_lm_session_info(session) == [4, 4, 1, 1]
    assert binding.byte_lm_session_rollback(session) == 3
    assert binding.byte_lm_session_info(session) == [3, -1, 1, 1]
    assert binding.byte_lm_session_rollback(session) == 3, "the shadow was restored twice"
    out = [zeros((shape.n_total,), "<f4") for _ in range(3)] + [zeros((shape.n_tensors,), "<i4")]
    assert binding.byte_lm_session_export_state(session, [addr(x, name="o") for x in out], native) == 3
    assert out[0].tobytes() == st["parameters"].tobytes() and out[2].tobytes() == st["v"].tobytes()
    binding.byte_lm_session_close(session)
    assert binding.byte_lm_session_info(session) == [-1, -1, 0, 0]


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
