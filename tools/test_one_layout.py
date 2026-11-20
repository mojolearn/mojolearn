# SPDX-License-Identifier: Apache-2.0
"""tools/one_layout.py: the ONE LAYOUT go/no-go from fake retained wheels, merged artifacts and
development receipts (no compile, no network)."""
import io
import json
import os
import sys
import tarfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import one_layout as ol  # noqa: E402

MiB = 2 ** 20


def blob(n, seed=0):
    # incompressible enough that deflate keeps the proportions
    return os.urandom(n) if seed is None else bytes((i * 2654435761 + seed) % 251 for i in range(n))


def wheel(path, members):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    return path


def test_wheel_rows_split_native_bytes_by_set(tmp_path):
    w = wheel(tmp_path / "mojolearn_nvidia-1.0-py3-none-manylinux_2_35_x86_64.whl", {
        "mojolearn/cuda_native/sm_89/identical/_mojolearn_native.so": blob(4000),
        "mojolearn/cuda_native/sm_89/identical/_mojolearn_merged.json": b"{}",
        "mojolearn/cuda_native/sm_89/_adapters/libmojolearn_adapter.so": blob(1000),
        "mojolearn/host/_mojolearn_native_host.so": blob(2000)})
    (row,) = ol.wheel_rows([w])
    assert row["distribution"] == "mojolearn-nvidia" and row["platform"] == "linux"
    assert row["sets"]["identical"]["files"] == 1 and row["sets"]["host"]["files"] == 1
    assert "_adapters" not in json.dumps(row["sets"])        # adapters are the same in both layouts


def test_merged_and_development_artifacts_are_read_from_their_stores(tmp_path):
    arc = tmp_path / "bincache" / "k.tar.gz"
    arc.parent.mkdir()
    with tarfile.open(arc, "w:gz") as tf:
        data = blob(3000)
        info = tarfile.TarInfo("python/mojolearn/identical/_g.so")
        info.size = len(data)
        tf.addfile(info, io.BytesIO(data))
    assert ol.merged_artifacts(arc.parent)["_g"]["bytes"] == 3000
    d = tmp_path / "dev" / "x"
    d.mkdir(parents=True)
    (d / "receipt.json").write_text(json.dumps(dict(stem="_dev_a", mode="identical", peak_rss_bytes=5,
                                                    execution=dict(elapsed_seconds=7.0))))
    (d / "_dev_a.so").write_bytes(blob(1000))
    s = tmp_path / "dev" / "support"
    s.mkdir()
    (s / "receipt.json").write_text(json.dumps(dict(name="libMojolearnMath")))
    (s / "libMojolearnMath.dylib").write_bytes(b"x")
    dev = ol.dev_artifacts(tmp_path / "dev")
    assert list(dev) == ["_dev_a"] and dev["_dev_a"]["seconds"] == 7.0


def test_set_ratio_central_and_pessimistic():
    release = [("_g1", ["a", "b"]), ("_g2", ["c", "d"])]
    dev_groups = [("_d_a", ["a"]), ("_d_b", ["b"]), ("_d_c", ["c"]), ("_d_d", ["d"])]
    merged = {"_g1": dict(deflate=100), "_g2": dict(deflate=100)}
    dev = {"_d_a": dict(deflate=80), "_d_b": dict(deflate=70), "_d_c": dict(deflate=60)}   # _d_d unmeasured
    r = ol.set_ratio(release, dev_groups, merged, dev)
    assert r["worst_group_ratio"] == 1.5 and r["missing"] == ["_d_d"]
    assert r["central_ratio"] == round((80 + 70 + 60 + 70) / 200, 3)          # median fill
    assert r["pessimistic_ratio"] == round((150 + 150) / 200, 3)              # _g2 grown by 1.5
    host = ol.set_ratio(release, dev_groups, merged, {}, worst_seen=2.0)
    assert host["pessimistic_ratio"] == 2.0 and host["grown_by"] == 2.0


def doc_for(macos_bytes, nvidia_bytes, central=1.4, pessimistic=1.4, grown=2.5, usage=None):
    wheels = [dict(wheel="mojolearn-1-py3-none-macosx_26_0_arm64.whl", distribution="mojolearn",
                   platform="darwin", file_bytes=macos_bytes,
                   sets={"identical": dict(files=8, compressed=macos_bytes - 6 * MiB, uncompressed=0)}),
              dict(wheel="mojolearn_nvidia-1-py3-none-manylinux_2_35_x86_64.whl", distribution="mojolearn-nvidia",
                   platform="linux", file_bytes=nvidia_bytes,
                   sets={"identical": dict(files=4, compressed=nvidia_bytes - MiB, uncompressed=0)})]
    ratios = {"identical": dict(central_ratio=central, pessimistic_ratio=pessimistic, grown_by=grown)}
    return dict(wheels=ol.estimate_wheels(wheels, ratios), pypi_usage=usage or {})


def test_go_when_every_wheel_fits_its_project_limit_pessimistically():
    v = ol.verdict(doc_for(50 * MiB, 33 * MiB))
    assert v["status"] == "GO" and all(r["state"] == "fits" for r in v["wheels"])
    nv = next(r for r in v["wheels"] if r["distribution"] == "mojolearn-nvidia")
    assert nv["limit_bytes"] == 250 * MiB and nv["pessimistic_bytes"] > nv["central_bytes"]   # Linux grown


def test_hold_when_only_the_central_estimate_fits_the_100_mib_mojolearn_limit():
    v = ol.verdict(doc_for(64 * MiB, 33 * MiB, central=1.3, pessimistic=1.7))
    assert v["status"] == "HOLD"
    mac = next(r for r in v["wheels"] if r["distribution"] == "mojolearn")
    assert mac["state"] == "fits only the central estimate" and mac["limit_bytes"] == 100 * MiB
    assert "measure the missing development artifacts" in v["reasons"][0]


def test_no_go_over_the_file_limit_or_the_project_size():
    assert ol.verdict(doc_for(90 * MiB, 33 * MiB, central=1.3, pessimistic=1.3))["status"] == "NO-GO"
    v = ol.verdict(doc_for(50 * MiB, 33 * MiB, usage={"mojolearn": 10 * 2 ** 30 - 10 * MiB}))
    assert v["status"] == "NO-GO" and "project size" in v["reasons"][0]
    p = next(x for x in v["projects"] if x["distribution"] == "mojolearn")
    assert p["releases_left_central"] == 0


def test_load_cost_counts_native_files_and_index_bytes():
    plans = {"release": {"identical": [("_g", ["a", "b"])]},
             "development": {"identical": [("_d_a", ["a"]), ("_d_b", ["b"])]}}
    out = ol.load_cost(plans)
    assert out["release"]["native_files"] == 1 and out["development"]["native_files"] == 2
    assert out["release"]["members"] == out["development"]["members"] == 2
