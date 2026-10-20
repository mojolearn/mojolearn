# SPDX-License-Identifier: Apache-2.0
"""The merged-binding layout of a Linux set directory: the packers' helper
(packaging/linux/merged_layout.py) and the PTX manifest's merged form
(gpu_plugins.validate_baseline_manifest). Inert bytes only, no build."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "python" / "mojolearn"))
import merged_layout as ml  # noqa: E402
import gpu_plugins as gp  # noqa: E402

NAMES = ["_mojolearn_x_trees", "_mojolearn_rf", "_mojolearn_gbdt"]


def merged(directory, members=NAMES, stem="_mojolearn_native", data=b"merged"):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / (stem + ".so")).write_bytes(data)
    (directory / ml.INDEX).write_text(json.dumps({"format": 1, "files": {stem + ".so": list(members)}}))
    return directory


def per_binding(directory, names=NAMES):
    directory.mkdir(parents=True, exist_ok=True)
    for n in names:
        (directory / (n + ".so")).write_bytes(n.encode())
    return directory


def test_layout_is_read_from_the_directory(tmp_path):
    assert ml.layout(merged(tmp_path / "m"), NAMES) == ("merged", {"_mojolearn_native.so": NAMES})
    assert ml.layout(per_binding(tmp_path / "p"), NAMES)[0] == "per-binding"
    assert ml.files_of(tmp_path / "m", NAMES) == ["_mojolearn_native.so", ml.INDEX]
    assert ml.files_of(tmp_path / "p", NAMES) == sorted(n + ".so" for n in NAMES)
    assert ml.member_file(tmp_path / "m", "_mojolearn_rf").name == "_mojolearn_native.so"
    assert ml.member_file(tmp_path / "p", "_mojolearn_rf").name == "_mojolearn_rf.so"
    assert ml.members_of(tmp_path / "m") == {n: "_mojolearn_native.so" for n in NAMES}


def test_own_file_beside_index_is_refused(tmp_path):
    d = merged(tmp_path)
    (d / "_mojolearn_rf.so").write_bytes(b"shadow")
    with pytest.raises(ml.LayoutError, match="shadow"):
        ml.layout(d, NAMES)


def test_index_must_serve_exactly_the_bindings(tmp_path):
    with pytest.raises(ml.LayoutError, match="members differ"):
        ml.layout(merged(tmp_path / "a", NAMES[:2]), NAMES)
    d = merged(tmp_path / "b")
    (d / "_mojolearn_native.so").unlink()
    with pytest.raises(ml.LayoutError, match="missing"):
        ml.layout(d, NAMES)
    with pytest.raises(ml.LayoutError, match="per-binding set differs"):
        ml.layout(per_binding(tmp_path / "c", NAMES[:2]), NAMES)


def test_host_glob(tmp_path):
    d = merged(tmp_path, ["_mojolearn_a_host", "_mojolearn_b_host"], stem="_mojolearn_native_host")
    assert ml.layout(d, ["_mojolearn_a_host", "_mojolearn_b_host"], "_mojolearn_*_host.so")[0] == "merged"


def test_switch():
    assert ml.enabled({}) is True
    assert ml.enabled({ml.SWITCH: "0"}) is False
    with pytest.raises(ml.LayoutError):
        ml.enabled({ml.SWITCH: "yes"})


def test_linux_plan_serves_one_file_per_directory():
    assert list(ml.plan("fast", False)) == ["_mojolearn_native.so"]
    assert "_mojolearn_byte_lm" in ml.plan("identical", True)["_mojolearn_native.so"]
    assert "_mojolearn_byte_lm" not in ml.plan("fast", True)["_mojolearn_native.so"]
    assert list(ml.plan("host", True)) == ["_mojolearn_native_host.so"]


def manifest(rows, indexes=None):
    doc = dict(schema=gp.PTX_MANIFEST_SCHEMA, code_format="ptx", vendor="cuda", target="sm_80",
               min_compute_capability=[8, 0], errors=[], source_commit="a" * 40, files=rows)
    if indexes is not None:
        doc["indexes"] = indexes
    return doc


PTX = [dict(sha256="0" * 64, target="sm_80", ptx_isa="8.1", approx={})]


def test_ptx_manifest_merged_form():
    rows = [dict(file="_mojolearn_native.so", sha256="1" * 64, numeric_mode="fast", ptx_modules=PTX, members=NAMES),
            dict(file="identical/_mojolearn_native.so", sha256="2" * 64, numeric_mode="identical",
                 ptx_modules=PTX, members=NAMES)]
    files = {r["file"]: r["sha256"] for r in rows}
    idx = [dict(file=ml.INDEX, sha256="3" * 64), dict(file="identical/" + ml.INDEX, sha256="4" * 64)]
    gp.validate_baseline_manifest(manifest(rows, idx), files)
    gp.validate_baseline_manifest(manifest(rows, idx), files, {i["file"]: i["sha256"] for i in idx})
    with pytest.raises(ValueError, match="index bytes"):
        gp.validate_baseline_manifest(manifest(rows, idx), files, {ml.INDEX: "f" * 64})
    with pytest.raises(ValueError, match="merged indexes"):
        gp.validate_baseline_manifest(manifest(rows, idx[:1]), files)
    bad = [dict(rows[0], ptx_modules=[])] + rows[1:]
    with pytest.raises(ValueError, match="merged-file evidence"):
        gp.validate_baseline_manifest(manifest(bad, idx), files)
    shadow = rows + [dict(file="_mojolearn_rf.so", sha256="5" * 64, numeric_mode="fast", ptx_modules=PTX)]
    with pytest.raises(ValueError, match="shadow"):
        gp.validate_baseline_manifest(manifest(shadow, idx), dict(files, **{"_mojolearn_rf.so": "5" * 64}))


def test_ptx_manifest_per_binding_delegation_still_valid():
    rows = [dict(file=n + ".so", sha256=hashlib.sha256(n.encode()).hexdigest(), numeric_mode="fast",
                 ptx_modules=PTX) for n in NAMES[1:]]
    rows.append(dict(file="_mojolearn_x_trees.so", sha256="9" * 64, numeric_mode="fast", ptx_modules=[],
                     delegates=[dict(file=r["file"], sha256=r["sha256"]) for r in rows]))
    files = {r["file"]: r["sha256"] for r in rows}
    gp.validate_baseline_manifest(manifest(rows), files)
    with pytest.raises(ValueError, match="merged indexes"):
        gp.validate_baseline_manifest(manifest(rows, [dict(file=ml.INDEX, sha256="3" * 64)]), files)
