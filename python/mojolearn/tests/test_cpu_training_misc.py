# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU training for the misc lanes (lane/cpu-training-misc, 2026-09-15):
kmeans-sqrt, kmeans-classic-pp, cross-val, bootstrap,
permutation-test, monte-carlo, optim-sgd, optim-adam-clip,
cross-entropy-arms and training-primitives, checked from
SOURCE so it runs on a box with nothing built, plus runtime checks that run
only where the host bindings are built and the package took the CPU-only
path.

What the source checks hold: the manifest declares the three k-means lanes
on the core family and cross-val on the gbdt family, and names each for the
docs; kmeans-sqrt is the one lane the gate diffs against the fix record
(TRAINING_FIX_COLUMNS), because the 166-lane record carries its pre-fix
cells on every column, and the fix record's three columns carry the lane
STABLE and equal on every fixture; the covered lanes split into the record
set and the fix set with nothing lost; the CPU identity gate reads both sets
and the fix columns from the manifest and diffs each against its own columns
in the covered step and in the sabotage step; `python -m mojolearn identity`
runs only the record set on a CPU-only install; the core host binding
registers `gather_rows_bytes` (cross_val_score's fold rows) under the base
binding's name, and the oracle refuses an unsupported metric in the device's
words; the resample family routes `_mojolearn_resample` to its own host
binding, which registers the GPU binding's three entries and read-backs and
not the multi-GPU range probe, and whose oracle imports no GPU module, calls
the device path's own host stages and carries the sabotage arm; the training
family declares the four neural lanes, its host binding registers the GPU
binding's clip, accumulate and Samba operation names, the new oracle imports
no GPU module, and the no-CPU-path sentence names the blocks that still have
none rather than "the neural blocks".

The runtime checks (skipped, and SAID to be skipped, when a binding is
absent or a GPU set loaded): an unsupported metric refuses with the
device's sentence, by its name on the Python side and by its code on the
Mojo side; the fold-row gather returns the rows a Python index
returns, byte for byte, and refuses an out-of-range index before writing;
a bootstrap run twice returns the same bytes and its `r_first` slice equals
the whole run's, and the range probe refuses by name; the host embedding
forward is the gather, the two-piece accumulate is the elementwise sum, and
the clip refuses max_norm <= 0 by name.
The bit claim against the GPU columns is the CPU identity gate's.

    cd python && python3 -m mojolearn.tests.test_cpu_training_misc
"""

# Gate-runner scope: host runtime checks require the CPU-only route.
GATE_BACKENDS = ("cpu",)
import json
import re
import sys
from pathlib import Path

import mojolearn
from mojolearn._cpu_reference import reference_training
from mojolearn import _backend, host_surface

ROOT = Path(__file__).resolve().parents[3]

KMEANS_LANES = ("kmeans-sqrt", "kmeans-classic-pp")
KMEANS_ORACLE = "cluster/host/kmeans_oracle.mojo"
RESAMPLE_ORACLE = "resample/host/resample_host.mojo"
RESAMPLE_LANES = ("bootstrap", "permutation-test", "monte-carlo")
NEURAL_LANES = ("optim-sgd", "optim-adam-clip", "cross-entropy-arms", "training-primitives")
SAMBA_ORACLE = "training/host/samba_ops_oracle.mojo"
NEURAL_EXPORTS = ("clip_grad_norm", "accumulate", "accumulation_is_aligned", "embedding_forward",
                  "embedding_backward", "rms_norm_forward", "rms_norm_backward", "linear_forward",
                  "linear_backward")
GPU_IMPORTS = re.compile(r"^\s*from\s+(max\.gpu|std\.gpu)", re.M)
#: The device's refusal of a metric it does not implement. The cosine metric
#: was DELETED on 2026-09-18 (lane/kmeans-cosine-capability); what must
#: survive is the property the routed refusal existed to give, which is that
#: an unsupported metric is refused BY NAME rather than accepted and ignored.
METRIC_REFUSAL = "kmeans supports only the L2Expanded (0) and L2SqrtExpanded (1) distance metrics"
FIXTURES = ("base", "ties", "hashed", "wide", "denormal", "denormal_ftz", "dupes", "odd", "negative")


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def test_manifest_covers_the_misc_lanes():
    covered = host_surface.covered_lanes()
    core = host_surface.family("core")
    for lane in KMEANS_LANES:
        assert lane in covered, f"{lane} is not a covered training lane"
        assert lane in core["training_lanes"], f"{lane} is not a core training lane"
        assert host_surface.TRAINING_LANE_NAMES[lane] in host_surface.training_sentence()
    gbdt = host_surface.family("gbdt")
    assert "cross-val" in covered and "cross-val" in gbdt["training_lanes"]
    assert "model_selection.cross_val_score" in gbdt["classes"]


def test_fix_lanes_split_the_covered_lanes():
    covered = host_surface.covered_lanes()
    record = host_surface.record_covered_lanes()
    fixed = host_surface.fix_covered_lanes()
    # kmeans-sqrt predates its fix on the record; the embedding and ivf lanes
    # (lane/cpu-training-embedding-ivf) are not in the record at all.
    assert fixed == ["kmeans-sqrt", "embedding", "embedding-sort", "ivf", "ivf-euclidean"], fixed
    assert not set(record) & set(fixed)
    assert sorted(record + fixed) == sorted(covered)
    assert [l for l in covered if l in record] == record, "the record set lost the gate's order"
    assert host_surface.main(["--fix-covered-lanes"]) == 0
    assert host_surface.main(["--record-covered-lanes"]) == 0


def test_fix_columns_carry_the_fixed_lane_on_every_fixture():
    """Every fix lane must be carried by exactly three of the fix columns, one
    per GPU vendor, STABLE with one hash per fixture across the three; and
    the record must either lack the lane or disagree with those hashes on
    some fixture (otherwise the lane belongs back on the record). kmeans-sqrt's
    three are the record's own boxes after the fix."""
    cols = {rel: json.loads(_read(rel)) for rel in host_surface.TRAINING_FIX_COLUMNS}
    recs = [json.loads(_read(rel))["cells"] for rel in host_surface.TRAINING_GPU_COLUMNS]
    for fix, rec in zip(host_surface.TRAINING_FIX_COLUMNS[:3], host_surface.TRAINING_GPU_COLUMNS):
        assert fix.rsplit("/", 1)[1] == rec.rsplit("/", 1)[1], (fix, rec)
    for lane in host_surface.fix_covered_lanes():
        carriers = [rel for rel, j in cols.items() if f"{lane}/base" in j["cells"]]
        assert len(carriers) == 3, f"{lane}: carried by {len(carriers)} fix columns {carriers}, want 3"
        # By file name: the ivf-euclidean record's Apple column predates the
        # derived vendor label and reads "arm64" (its platform is macOS arm64).
        vendors = sorted(v for rel in carriers for v in ("amd", "apple", "nvidia")
                         if v in rel.rsplit("/", 1)[1])
        assert vendors == ["amd", "apple", "nvidia"], (lane, vendors)
        differs = 0
        for fx in FIXTURES:
            key = f"{lane}/{fx}"
            hashes = set()
            for rel in carriers:
                c = cols[rel]["cells"][key]
                assert c["verdict"] == "STABLE", (key, rel, c["verdict"])
                hashes.add(c["hashes"][0])
            assert len(hashes) == 1, f"{key}: the fix columns disagree {hashes}"
            if any(key not in r or r[key]["hashes"][0] not in hashes for r in recs):
                differs += 1
        assert differs > 0, f"{lane}: the record already carries the fixed cells; drop it from TRAINING_FIX_LANES"


def test_workflow_diffs_each_set_against_its_columns():
    text = _read(".github/workflows/cpu-identity-gate.yml")
    for flag in ("--record-covered-lanes", "--fix-covered-lanes", "--training-fix-columns"):
        assert flag in text, f"the workflow does not read {flag}"
    assert text.count('--diff $GPU_COLUMNS "$GATE_OUT/cpu-') == 2, "record diffs (covered and sabotage)"
    assert text.count('--lanes "$RECORD_COVERED_LANES"') == 2
    assert text.count('--diff $FIX_COLUMNS "$GATE_OUT/cpu-') == 2, "fix diffs (covered and sabotage)"
    assert text.count('--lanes "$FIX_COVERED_LANES"') == 2
    for rel in host_surface.TRAINING_FIX_COLUMNS:
        directory = "/" + rel.rsplit("/", 1)[0] + "/"
        assert directory in text, f"the sparse checkout does not bring down {directory}"


def test_identity_command_runs_public_reference_probes_on_a_cpu():
    text = _read("python/mojolearn/_identity.py")
    assert "set(host_surface.public_reference_lanes()) & set(host_surface.record_covered_lanes())" in text
    assert "l in cpu_record_lanes]" in text
    host_only = host_surface.PUBLIC_HOST_ONLY_LANES
    trained = set(host_surface.public_reference_lanes()) - set(host_only)
    # `identity` replays the older bundled column; `verify` has a broader,
    # independently admitted table. Only their intersection belongs here.
    assert trained & set(host_surface.record_covered_lanes())
    # Reachable from a binding that SHIPS. Two ways, and the second is not a
    # loophole (lane/expose-inference-surface, 2026-09-16): a family holding a
    # fit stays a source build while a shipped inference-only binding serves
    # its route. `kpss` is declared by `tsa`, which holds holtwinters_fit and
    # must not ship, yet a user can call it because the shipped `forecast`
    # binding serves `_mojolearn_tsa` and registers `kpss_test`. Demanding the
    # declaring family itself ship would reject a lane that in fact works.
    wheel_lanes = {lane for f in host_surface.FAMILIES if f["ships_in_wheel"] for lane in f["training_lanes"]}
    routes, shipped = host_surface.inference_routes(), set(host_surface.wheel_bindings())
    served_lanes = {lane for f in host_surface.FAMILIES
                    if not f["ships_in_wheel"] and routes.get(f["routes"]) in shipped
                    for lane in f["training_lanes"]}
    unreachable = trained - wheel_lanes - served_lanes
    assert unreachable == set(), (
        f"public reference lanes no shipped binding can serve: {sorted(unreachable)}")
    for lane, family in host_only.items():
        if family is None:
            assert lane in ("cross-val-folds",), "unclassified pure-Python lane"
            continue
        f = host_surface.family(family)
        assert f["ships_in_wheel"] and f["routes"] is None, f"{lane}: not a shipped host-only family"


def test_core_host_binding_registers_the_fold_gather():
    src = _read(host_surface.binding_source("core"))
    assert '("gather_rows_bytes")' in src
    assert "gather_rows_bytes" in host_surface.family("core")["exports"]
    helpers = _read("bindings/host_helpers.mojo")
    assert "def gather_rows_bytes_binding(" in helpers
    assert '_native("gather_rows_bytes")' in _read("python/mojolearn/model_selection.py")


def test_oracle_refuses_an_unknown_metric_in_the_device_words():
    text = _read(KMEANS_ORACLE)
    flat = re.sub(r'"\s*\n\s*"', "", text)
    assert METRIC_REFUSAL in flat
    params = re.sub(r'"\s*\n\s*"', "", _read("cluster/impl/kmeans_params.mojo"))
    assert METRIC_REFUSAL in params, "the device's refusal sentence moved; the oracle must follow it"


def test_the_cosine_metric_is_gone_from_every_kmeans_source():
    """lane/kmeans-cosine-capability, 2026-09-18. The metric was deleted, not
    refused: cuVS refuses cosine k-means too (`kmeans_common.cuh:320`), and
    the arithmetic mean does not minimize cosine distance, so the fit did not
    descend. A reappearing constant here means someone revived the dead arm
    without the update step that would make it correct."""
    for path in (KMEANS_ORACLE, "cluster/impl/kmeans_params.mojo",
                 "cluster/impl/detail/kmeans_common.mojo",
                 "cluster/impl/distance/unfused_distance_nn.mojo",
                 "python/mojolearn/cluster.py"):
        text = _read(path)
        assert "METRIC_COSINE_EXPANDED" not in text, f"{path} still declares or uses METRIC_COSINE_EXPANDED"


def test_manifest_covers_the_resample_lanes():
    fam = host_surface.family("resample")
    assert host_surface.routed_modules()["_mojolearn_resample"] == "_mojolearn_resample_host"
    for lane in RESAMPLE_LANES:
        assert lane in host_surface.record_covered_lanes(), f"{lane} is not a record-covered lane"
        assert lane in fam["training_lanes"]
    assert RESAMPLE_ORACLE in fam["host_modules"] and (ROOT / RESAMPLE_ORACLE).is_file()
    src = _read(host_surface.binding_source("resample"))
    gpu = _read("bindings/_mojolearn_resample.mojo")
    for name in ("bootstrap", "permutation_test", "monte_carlo_integrate", "resample_numeric_mode", "resample_vendor"):
        assert f'("{name}")' in src and f'("{name}")' in gpu, name
        assert name in fam["exports"], name
    assert '("resample_ranges_parallel_available")' in gpu
    assert "resample_ranges_parallel_available" not in re.findall(r'def_function\[\w+\]\("(\w+)"\)', src)
    assert (ROOT / host_surface.build_shim("resample")).is_file()


def test_resample_oracle_is_host_only_and_sabotaged():
    text = _read(RESAMPLE_ORACLE)
    assert not GPU_IMPORTS.search(text), f"{RESAMPLE_ORACLE} imports a GPU module"
    assert not re.search(r"^\s*from .*import.*DeviceContext", text, re.M)
    assert not re.search(r"^\s*from resample\.estimator import", text, re.M), "the oracle must not import the device entry points"
    for stage in ("permutation_pvalue", "distribution_standard_error", "percentile_interval",
                  "basic_interval", "narrow_for_alternative", "mc_finish_host", "draw_row_index",
                  "draw_permutation_key", "draw_uniform_in", "float_to_sortable"):
        assert stage in text, f"{RESAMPLE_ORACLE} does not call {stage}"
    define = host_surface.sabotage_define("resample")
    assert f'is_defined["{define}"]()' in text
    assert "comptime if RESAMPLE_HOST_SABOTAGE:" in text
    assert "RESAMPLE_HOST_SABOTAGE" in _read(host_surface.binding_source("resample"))


def test_manifest_covers_the_neural_lanes():
    fam = host_surface.family("training")
    for lane in NEURAL_LANES:
        assert lane in host_surface.record_covered_lanes(), f"{lane} is not a record-covered lane"
        assert lane in fam["training_lanes"]
    src = _read(host_surface.binding_source("training"))
    gpu = _read("bindings/_mojolearn_training.mojo")
    for name in NEURAL_EXPORTS:
        assert f'("{name}")' in src and f'("{name}")' in gpu, name
        assert name in fam["exports"], name
    for absent in ("clip_parallel_available", "optimizer_parallel_available"):
        assert f'("{absent}")' in gpu and f'("{absent}")' not in src, absent
    assert SAMBA_ORACLE in fam["host_modules"] and (ROOT / SAMBA_ORACLE).is_file()
    text = _read(SAMBA_ORACLE)
    assert not GPU_IMPORTS.search(text), f"{SAMBA_ORACLE} imports a GPU module"
    assert not re.search(r"^\s*from training\.samba_ops import", text, re.M)
    for oracle in ("emb_forward_oracle", "emb_backward_oracle", "gemm_oracle", "gemm_backward_a_call",
                   "gemm_backward_b_call", "identical_rsqrt"):
        assert oracle in text, oracle
    sentence = host_surface.no_cpu_path_sentence()
    assert "neural blocks" not in sentence and "Samba" not in sentence, sentence


def _cpu_only_with(basename):
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return False
    if basename not in _backend.host_families_built():
        print(f"SKIP: {basename} is not built")
        return False
    return True


@reference_training()
def test_an_unsupported_metric_refuses_on_the_host_when_built():
    """Both halves of the refusal, because they live in different languages.

    A NAME the table does not carry is refused in Python by `_metric_code`; a
    CODE the kernel does not implement is refused in Mojo by
    `KMeansParams.validate` / `host_validate_params`. `metric=2` is the code
    the deleted cosine metric used, and it must not be accepted and ignored.
    """
    if not _cpu_only_with("_mojolearn_core_host"):
        return
    import numpy as np
    x = np.random.default_rng(0).standard_normal((64, 3)).astype(np.float32)

    try:
        mojolearn.KMeans(n_clusters=4, random_state=3, metric="cosine").fit(x)
    except Exception as exc:
        assert "metric must be one of" in str(exc), str(exc)
    else:
        raise AssertionError("KMeans(metric='cosine') fit; an unknown name was accepted")

    try:
        mojolearn.KMeans(n_clusters=4, random_state=3, metric=2).fit(x)
    except Exception as exc:
        assert METRIC_REFUSAL in str(exc), str(exc)
        return
    raise AssertionError("KMeans(metric=2) fit on the host; an unknown code was accepted")


@reference_training()
def test_fold_gather_matches_python_indexing_when_built():
    if not _cpu_only_with("_mojolearn_core_host"):
        return
    import numpy as np
    from mojolearn.model_selection import _take_rows
    from mojolearn._array import Array
    x = np.random.default_rng(1).standard_normal((50, 7)).astype(np.float32)
    idx = Array.from_list([49, 0, 7, 7, 13], "<i8")
    got = np.asarray(_take_rows(x, idx))
    assert got.tobytes() == x[[49, 0, 7, 7, 13]].tobytes()
    gather = _backend.load_host_module("_mojolearn_core_host").gather_rows_bytes
    out = np.zeros((2, 7), dtype=np.float32)
    bad = np.asarray([1, 50], dtype=np.int64)
    try:
        gather(x.ctypes.data, out.ctypes.data, bad.ctypes.data, 50, 2, 28)
    except Exception as exc:
        assert "row index out of bounds" in str(exc)
        assert not out.any(), "a refused gather wrote rows"
        return
    raise AssertionError("an out-of-range fold index was gathered")


@reference_training()
def test_bootstrap_runs_on_the_host_when_built():
    if not _cpu_only_with("_mojolearn_resample_host"):
        return
    import numpy as np
    module = _backend.load_host_module("_mojolearn_resample_host")
    assert not bool(module.resample_host_sabotage()), "a sabotage build loaded outside the gate"
    assert str(module.resample_host_column()) == "cpu"
    rs = mojolearn.resample
    x = np.random.default_rng(2).standard_normal(300).astype(np.float32)
    a = rs.bootstrap(x, n_resamples=64, random_state=5)
    b = rs.bootstrap(x, n_resamples=64, random_state=5)
    assert np.asarray(a.distribution).tobytes() == np.asarray(b.distribution).tobytes()
    part = rs.bootstrap(x, n_resamples=16, random_state=5, r_first=8)
    assert np.asarray(part.distribution).tobytes() == np.asarray(a.distribution)[8:24].tobytes()
    s = np.asarray(a.sorted_distribution)
    assert (s[1:] >= s[:-1]).all()
    p = rs.permutation_test(x[:40], x[40:90], n_resamples=32, random_state=1)
    assert 0.0 < p.pvalue <= 1.0
    m = rs.monte_carlo_integrate("const", [0.0, 0.0], [1.0, 2.0], 1000)
    assert m.integral == 2.0 and m.closed_form == 2.0
    try:
        _backend.binding("_mojolearn_resample").resample_ranges_parallel_available
    except ImportError as exc:
        assert "resample_ranges_parallel_available" in str(exc)
    else:
        raise AssertionError("the host binding exported the multi-GPU range probe")


@reference_training()
def test_neural_primitives_run_on_the_host_when_built():
    if not _cpu_only_with("_mojolearn_training_host"):
        return
    import numpy as np
    T = mojolearn.training
    module = _backend.load_host_module("_mojolearn_training_host")
    assert not bool(module.training_host_sabotage()), "a sabotage build loaded outside the gate"
    rng = np.random.default_rng(3)
    w = rng.standard_normal((10, 4)).astype(np.float32)
    ids = np.asarray([3, 0, 9, 3], dtype=np.int32)
    assert np.asarray(T.embedding_forward(w, ids)).tobytes() == w[ids].tobytes()
    a = np.asarray([1.0, 2.0, -4.0], dtype=np.float32)
    b = np.asarray([0.5, -2.0, 8.0], dtype=np.float32)
    assert np.asarray(T.accumulate_grads([a, b], None)).tolist() == [1.5, 0.0, 4.0]
    try:
        T.clip_grad_norm_([a.copy()], 0.0)
    except (ValueError, RuntimeError, Exception) as exc:
        assert "max_norm" in str(exc), str(exc)
    else:
        raise AssertionError("clip_grad_norm_ accepted max_norm=0")


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
