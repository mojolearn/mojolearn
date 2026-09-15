# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU training for the misc lanes (lane/cpu-training-misc, 2026-09-15):
kmeans-sqrt, kmeans-classic-pp, kmeans-cosine and cross-val, checked from
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
binding's name, and the oracle refuses the cosine metric in the device's
words.

The runtime checks (skipped, and SAID to be skipped, when a binding is
absent or a GPU set loaded): `KMeans(metric='cosine')` refuses with the
device's sentence; the fold-row gather returns the rows a Python index
returns, byte for byte, and refuses an out-of-range index before writing.
The bit claim against the GPU columns is the CPU identity gate's.

    cd python && python3 -m mojolearn.tests.test_cpu_training_misc
"""
import json
import re
import sys
from pathlib import Path

import mojolearn
from mojolearn import _backend, host_surface

ROOT = Path(__file__).resolve().parents[3]

KMEANS_LANES = ("kmeans-sqrt", "kmeans-classic-pp", "kmeans-cosine")
KMEANS_ORACLE = "cluster/host/kmeans_oracle.mojo"
COSINE_REFUSAL = "kmeans only supports L2Expanded or L2SqrtExpanded distance metrics."
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
    assert fixed == ["kmeans-sqrt"], fixed
    assert not set(record) & set(fixed)
    assert sorted(record + fixed) == sorted(covered)
    assert [l for l in covered if l in record] == record, "the record set lost the gate's order"
    assert host_surface.main(["--fix-covered-lanes"]) == 0
    assert host_surface.main(["--record-covered-lanes"]) == 0


def test_fix_columns_carry_the_fixed_lane_on_every_fixture():
    """The three fix columns must exist, be the record boxes, and carry each
    fix lane STABLE with one hash per fixture across all three; the 166-lane
    record must NOT (otherwise the lane belongs back on the record)."""
    assert len(host_surface.TRAINING_FIX_COLUMNS) == 3
    for fix, rec in zip(host_surface.TRAINING_FIX_COLUMNS, host_surface.TRAINING_GPU_COLUMNS):
        assert fix.rsplit("/", 1)[1] == rec.rsplit("/", 1)[1], (fix, rec)
    cols = [json.loads(_read(rel))["cells"] for rel in host_surface.TRAINING_FIX_COLUMNS]
    recs = [json.loads(_read(rel))["cells"] for rel in host_surface.TRAINING_GPU_COLUMNS]
    for lane in host_surface.fix_covered_lanes():
        differs = 0
        for fx in FIXTURES:
            key = f"{lane}/{fx}"
            hashes = set()
            for c in cols:
                assert c[key]["verdict"] == "STABLE", (key, c[key]["verdict"])
                hashes.add(c[key]["hashes"][0])
            assert len(hashes) == 1, f"{key}: the fix columns disagree {hashes}"
            if any(r[key]["hashes"][0] not in hashes for r in recs):
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
    assert '- "python/mojolearn/model_selection.py"' in text


def test_identity_command_runs_the_record_set_on_a_cpu():
    text = _read("python/mojolearn/_identity.py")
    assert "l in host_surface.record_covered_lanes()]" in text


def test_core_host_binding_registers_the_fold_gather():
    src = _read(host_surface.binding_source("core"))
    assert '("gather_rows_bytes")' in src
    assert "gather_rows_bytes" in host_surface.family("core")["exports"]
    helpers = _read("bindings/host_helpers.mojo")
    assert "def gather_rows_bytes_binding(" in helpers
    assert '_native("gather_rows_bytes")' in _read("python/mojolearn/model_selection.py")


def test_oracle_refuses_cosine_in_the_device_words():
    text = _read(KMEANS_ORACLE)
    flat = re.sub(r'"\s*\n\s*"', "", text)
    assert COSINE_REFUSAL in flat
    params = re.sub(r'"\s*\n\s*"', "", _read("cluster/impl/kmeans_params.mojo"))
    assert COSINE_REFUSAL in params, "the device's refusal sentence moved; the oracle must follow it"


def _cpu_only_with(basename):
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return False
    if basename not in _backend.host_families_built():
        print(f"SKIP: {basename} is not built")
        return False
    return True


def test_cosine_refuses_on_the_host_when_built():
    if not _cpu_only_with("_mojolearn_core_host"):
        return
    import numpy as np
    x = np.random.default_rng(0).standard_normal((64, 3)).astype(np.float32)
    try:
        mojolearn.KMeans(n_clusters=4, random_state=3, metric="cosine").fit(x)
    except Exception as exc:
        assert COSINE_REFUSAL in str(exc), str(exc)
        return
    raise AssertionError("KMeans(metric='cosine') fit on the host; the refusal was lifted")


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


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
