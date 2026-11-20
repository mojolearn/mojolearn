# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU training for the misc lanes: kmeans-sqrt, kmeans-classic-pp,
cross-val, bootstrap, permutation-test, monte-carlo, optim-sgd,
optim-adam-clip, cross-entropy-arms and training-primitives.

The manifest checks: the covered lanes split into the record set and the fix
set with nothing lost (no lane is diffed against a fix record since
release-0845), and `python -m mojolearn identity` can serve every public
reference lane of the record set from a binding that ships.

The runtime checks (skipped, and SAID to be skipped, when a binding is
absent or a GPU set loaded): an unsupported metric refuses with the
device's sentence, by its name on the Python side and by its code on the
Mojo side; the fold-row gather returns the rows a Python index
returns, byte for byte, and refuses an out-of-range index before writing;
a bootstrap run twice returns the same bytes and its `r_first` slice equals
the whole run's; the host embedding forward is the gather, the two-piece
accumulate is the elementwise sum, and the clip refuses max_norm <= 0 by
name. The bit claim against the GPU columns is the CPU identity gate's.

    cd python && python3 -m mojolearn.tests.test_cpu_training_misc
"""

# Gate-runner scope: host runtime checks require the CPU-only route.
GATE_BACKENDS = ("cpu",)
import sys

import mojolearn
from mojolearn import _backend, host_surface

#: The device's refusal of a metric it does not implement. The cosine metric
#: was DELETED on 2026-09-18 (lane/kmeans-cosine-capability); what must
#: survive is the property the routed refusal existed to give, which is that
#: an unsupported metric is refused BY NAME rather than accepted and ignored.
METRIC_REFUSAL = "kmeans supports only the L2Expanded (0) and L2SqrtExpanded (1) distance metrics"


def test_fix_lanes_split_the_covered_lanes():
    covered = host_surface.covered_lanes()
    record = host_surface.record_covered_lanes()
    fixed = host_surface.fix_covered_lanes()
    # release-0845: the 2026-10-14 record carries kmeans-sqrt after its fix and
    # the embedding and ivf lanes, so no lane is diffed against a fix record.
    assert fixed == [], fixed
    assert not set(record) & set(fixed)
    assert sorted(record + fixed) == sorted(covered)
    assert [l for l in covered if l in record] == record, "the record set lost the gate's order"
    assert host_surface.main(["--fix-covered-lanes"]) == 0
    assert host_surface.main(["--record-covered-lanes"]) == 0


def test_identity_command_runs_public_reference_probes_on_a_cpu():
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
        assert f["ships_in_wheel"], f"{lane}: native host family does not ship"
        if lane == "cross-val-folds":
            # The descriptor uses CPU helpers in the mixed core binding.
            # A GPU install loads those same host operations through core's
            # routed name; that does not turn fold metadata into GPU work.
            assert family == "core" and f["routes"] == "_mojolearn"
            assert {"fold_ids", "select_fold_i64"} <= set(f["exports"])
        else:
            assert f["routes"] is None, f"{lane}: not a shipped host-only family"


def _cpu_only_with(basename):
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return False
    if basename not in _backend.host_families_built():
        print(f"SKIP: {basename} is not built")
        return False
    return True


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


def test_fold_gather_matches_python_indexing_when_built():
    if not _cpu_only_with("_mojolearn_core_host"):
        return
    import numpy as np
    from mojolearn.model_selection import _Resident
    from mojolearn._array import Array
    x = np.random.default_rng(1).standard_normal((50, 7)).astype(np.float32)
    idx = Array.from_list([49, 0, 7, 7, 13], "<i8")
    with _Resident() as res:
        got = np.asarray(res.take(x, idx))
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
