"""PyPI wheel provenance: digests against PyPI's JSON API, the manifest, and the loaded-from check."""
import hashlib
import json
import os
from types import SimpleNamespace

import pytest

import pypi_wheel_install as P


V = "0.8.44"


def _files(tmp_path, vendor="nvidia", legacy=False):
    names = {dist: "%s-%s-py3-none-manylinux_2_34_x86_64.whl" % (dist.replace("-", "_"), V)
             for dist in P.distributions(vendor, legacy)}
    digests = {}
    for dist, fname in names.items():
        data = ("bytes of " + fname).encode()
        (tmp_path / fname).write_bytes(data)
        digests[dist] = {fname: hashlib.sha256(data).hexdigest()}
    return names, digests


def _fetch(digests, tamper=None):
    def fetch(url):
        name = url.split("/pypi/")[1].split("/")[0]
        urls = [{"filename": f, "digests": {"sha256": "0" * 64 if f == tamper else d}}
                for f, d in digests[name].items()]
        return {"urls": urls}
    return fetch


IMPORT = "import mojolearn"  # the probe's record of `import mojolearn` (module_file / import_error)


def _loaded(manifest, **over):
    rec = {}
    for f in manifest["files"]:
        rec[f["distribution"]] = {"version": manifest["version"], "location": "/v/lib/python3.12/site-packages",
                                  "direct_url": {"url": "file:///cache/" + f["file"],
                                                 "archive_info": {"hashes": {"sha256": f["sha256"]}}}}
    rec[IMPORT] = {"module_file": "/v/lib/python3.12/site-packages/mojolearn/__init__.py"}
    for k, v in over.items():
        rec[k].update(v)
    return rec


def test_distributions_per_vendor():
    # a Linux box installs mojolearn-core and its own vendor's packages; Apple installs mojolearn
    assert P.distributions("nvidia") == ["mojolearn-core", "mojolearn-nvidia", "mojolearn-nvidia-ptx"]
    assert P.distributions("amd") == ["mojolearn-core", "mojolearn-amd"]
    assert P.distributions("apple") == ["mojolearn"]
    # a release from before mojolearn-core: the Linux core was mojolearn itself
    assert P.distributions("nvidia", legacy=True) == ["mojolearn", "mojolearn-nvidia"]
    assert P.distributions("amd", legacy=True) == ["mojolearn", "mojolearn-amd"]
    assert P.distributions("apple", legacy=True) == ["mojolearn"]
    with pytest.raises(ValueError):
        P.distributions("cpu")


def test_manifest_records_specs_index_files_and_pypi_digest_match(tmp_path):
    names, digests = _files(tmp_path)
    m = P.build_manifest(V, "nvidia", str(tmp_path), fetch=_fetch(digests))
    assert m["index_url"] == "https://pypi.org/simple"
    assert m["requested"] == ["mojolearn-core==" + V, "mojolearn-nvidia==" + V, "mojolearn-nvidia-ptx==" + V]
    assert [f["file"] for f in m["files"]] == [names["mojolearn-core"], names["mojolearn-nvidia"],
                                               names["mojolearn-nvidia-ptx"]]
    assert m["files"][0]["file"].startswith("mojolearn_core-%s-" % V)
    assert all(f["pypi_digest_match"] and f["sha256"] == f["pypi_sha256"] for f in m["files"])
    assert m["all_digests_match"]


def test_legacy_manifest_names_the_old_core(tmp_path):
    names, digests = _files(tmp_path, "amd", legacy=True)
    m = P.build_manifest(V, "amd", str(tmp_path), fetch=_fetch(digests), legacy=True)
    assert m["requested"] == ["mojolearn==" + V, "mojolearn-amd==" + V]
    assert P.refusal(m, _loaded(m)) is None


def test_apple_manifest_is_mojolearn_alone(tmp_path):
    names, digests = _files(tmp_path, "apple")
    m = P.build_manifest(V, "apple", str(tmp_path), fetch=_fetch(digests))
    assert m["requested"] == ["mojolearn==" + V]
    assert P.refusal(m, _loaded(m)) is None


def test_digest_mismatch_is_recorded_and_refused(tmp_path):
    names, digests = _files(tmp_path, "amd")
    m = P.build_manifest(V, "amd", str(tmp_path), fetch=_fetch(digests, tamper=names["mojolearn-amd"]))
    assert not m["all_digests_match"]
    assert [f["pypi_digest_match"] for f in m["files"]] == [True, False]
    assert "PyPI" in P.refusal(m, _loaded(m))


def test_missing_or_ambiguous_download_refuses(tmp_path):
    with pytest.raises(SystemExit):
        P.pick_file(str(tmp_path), "mojolearn", V)
    (tmp_path / ("mojolearn-%s-a.whl" % V)).write_bytes(b"a")
    (tmp_path / ("mojolearn-%s-b.whl" % V)).write_bytes(b"b")
    with pytest.raises(SystemExit):
        P.pick_file(str(tmp_path), "mojolearn", V)


def test_loaded_from_the_verified_files_passes(tmp_path):
    _, digests = _files(tmp_path)
    m = P.build_manifest(V, "nvidia", str(tmp_path), fetch=_fetch(digests))
    assert P.refusal(m, _loaded(m)) is None


@pytest.mark.parametrize("over,needle", [
    ({IMPORT: {"module_file": "/repo/python/mojolearn/__init__.py"}}, "site-packages"),
    ({"mojolearn-core": {"version": "0.8.43"}}, "0.8.43"),
    ({"mojolearn-nvidia": {"version": "0.8.45"}}, "0.8.45"),
    ({"mojolearn-nvidia-ptx": {"version": "0.8.46"}}, "0.8.46"),
    ({"mojolearn-core": {"direct_url": None}}, "not installed from the verified file"),
    ({"mojolearn-core": {"direct_url": {"url": "file:///x/other.whl"}}}, "not installed from the verified file"),
    ({"mojolearn-core": {"error": "PackageNotFoundError: mojolearn-core"}}, "mojolearn-core is not installed"),
    ({IMPORT: {"import_error": "ImportError: no"}}, "does not import"),
])
def test_refuses_when_ours_did_not_load_from_those_files(tmp_path, over, needle):
    _, digests = _files(tmp_path)
    m = P.build_manifest(V, "nvidia", str(tmp_path), fetch=_fetch(digests))
    assert needle in P.refusal(m, _loaded(m, **over))


def test_installed_hash_must_equal_the_verified_file(tmp_path):
    _, digests = _files(tmp_path)
    m = P.build_manifest(V, "nvidia", str(tmp_path), fetch=_fetch(digests))
    f = m["files"][0]
    bad = {"direct_url": {"url": "file:///c/" + f["file"], "archive_info": {"hash": "sha256=" + "1" * 64}}}
    assert f["distribution"] == "mojolearn-core"
    assert "hash" in P.refusal(m, _loaded(m, **{"mojolearn-core": bad}))


def test_install_refuses_a_nonempty_download_directory(tmp_path):
    (tmp_path / "old.whl").write_bytes(b"x")
    with pytest.raises(SystemExit):
        P.install(SimpleNamespace(dir=str(tmp_path), version=V, vendor="nvidia", python="python",
                                  manifest=str(tmp_path / "m.json"), log=None))


def test_bench_board_cell_record_carries_the_manifest_and_guard_refuses(tmp_path, monkeypatch):
    import bench_board as B
    _, digests = _files(tmp_path)
    m = P.build_manifest(V, "nvidia", str(tmp_path), fetch=_fetch(digests))
    rec = B.pypi_wheel_record(m)
    assert rec["sha256"] == m["files"][0]["sha256"] and rec["pypi"]["files"] == m["files"]
    assert rec["requested"] == m["requested"] and rec["index_url"] == P.INDEX_URL
    path = tmp_path / "wheel-manifest.json"
    path.write_text(json.dumps(m))
    args = SimpleNamespace(wheel_manifest=str(path), mojolearn_version=None, mojolearn_wheel=None,
                           opponents_only=False)
    monkeypatch.setattr(B.PYPI, "loaded_receipt", lambda py, man: _loaded(man))
    wheel, why = B.pypi_wheel_guard(args, "python", "nvidia")
    assert why is None and wheel["pypi"]["loaded"]["mojolearn-core"]["version"] == V
    assert wheel["file"] == m["files"][0]["file"] and wheel["file"].startswith("mojolearn_core-")
    assert args.mojolearn_version == V        # the box's mojolearn.requested
    _, why = B.pypi_wheel_guard(args, "python", "amd")
    assert "nvidia" in why
    monkeypatch.setattr(B.PYPI, "loaded_receipt",
                        lambda py, man: _loaded(man, **{IMPORT: {"module_file": "/src/mojolearn/__init__.py"}}))
    _, why = B.pypi_wheel_guard(args, "python", "nvidia")
    assert "site-packages" in why
    # A GPU board timing a PyPI version without a manifest has no provenance: refused.
    _, why = B.pypi_wheel_guard(SimpleNamespace(wheel_manifest=None, mojolearn_version=V, mojolearn_wheel=None,
                                                opponents_only=False), "python", "amd")
    assert "--wheel-manifest" in why
    assert B.pypi_wheel_guard(SimpleNamespace(wheel_manifest=None, mojolearn_version=None, mojolearn_wheel=None,
                                              opponents_only=False), "python", "amd") == (None, None)
