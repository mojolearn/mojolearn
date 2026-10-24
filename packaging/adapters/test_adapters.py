# SPDX-License-Identifier: Apache-2.0
"""The framework adapters' packaging glue, on inert files (no compiler, no GPU, no torch, no network).

    .pixi/envs/test/bin/python -m pytest -q packaging/adapters/test_adapters.py

Covers packaging/adapters/{pins.json,manifest.py,fetch_torch.py,build.sh}, the
pins against python/pyproject.toml's extras, the installed loader's constants
(python/mojolearn/adapters/_shipped.py), stage_libs.py's adapter RUNPATHs,
ptx_contract.py's IDENTICAL scope and the native-source inventory rule.
"""
import hashlib
import importlib.util
import io
import json
import os
import pathlib
import posixpath
import subprocess
import sys
import tomllib
import zipfile

import pytest

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


manifest = load("adapter_manifest_under_test", HERE / "manifest.py")
fetch = load("adapter_fetch_under_test", HERE / "fetch_torch.py")
shipped = load("adapter_shipped_under_test", REPO / "python/mojolearn/adapters/_shipped.py")
stage_libs = load("stage_libs_adapters_under_test", REPO / "packaging/linux/stage_libs.py")
sys.path.insert(0, str(REPO / "packaging/linux"))
ptx_contract = load("ptx_contract_adapters_under_test", REPO / "packaging/linux/ptx_contract.py")
sys.path.insert(0, str(REPO / "tools"))
qualification = load("check_linux_release_qualification_adapters", REPO / "tools/check_linux_release_qualification.py")

PINS = json.loads((HERE / "pins.json").read_text())


# ---------------------------------------------------------------- the pins

def test_pyproject_extras_are_the_pins():
    extras = tomllib.loads((REPO / "python/pyproject.toml").read_text())["project"]["optional-dependencies"]
    assert extras["torch"] == [PINS["torch"]["requirement"]]
    assert extras["max"] == [PINS["max"]["requirement"]]
    assert PINS["max"]["requirement"] == "max==" + PINS["max"]["version"]


def test_every_vendor_pins_one_torch_of_the_series():
    assert set(PINS["torch"]["build"]) == set(manifest.VENDORS)
    for vendor, pin in PINS["torch"]["build"].items():
        assert shipped.series(pin["version"]) == PINS["torch"]["series"], vendor
        assert pin["backend"] == {"cuda": "cuda", "hip": "rocm", "metal": "mps"}[vendor]
        assert pin["index_url"].startswith("https://")
    assert "rocm" in PINS["torch"]["build"]["hip"]["version"]


def test_the_package_lists_the_adapters_and_ships_their_files_on_macos():
    config = tomllib.loads((REPO / "python/pyproject.toml").read_text())["tool"]["setuptools"]
    assert "mojolearn.adapters" in config["packages"]
    assert "_adapters/*" in config["package-data"]["mojolearn"]


def test_the_loader_and_the_builder_name_the_same_files():
    for name in ("SCHEMA", "ABI", "DIRECTORY", "MANIFEST", "FILES"):
        assert getattr(shipped, name) == getattr(manifest, name), name
    assert stage_libs.ADAPTER_DIR == manifest.DIRECTORY
    assert {stage_libs.ADAPTER_NATIVE, stage_libs.ADAPTER_TORCH} == {manifest.FILES["native"], manifest.FILES["torch"]}


def test_series():
    assert shipped.series("2.13.0+rocm7.2") == "2.13"
    assert shipped.series("2.13.1") == "2.13"
    # the installed torch passes the series check with or without a local label
    assert shipped.series("2.13.0+cu130") == shipped.series("2.13.0") == PINS["torch"]["series"]
    assert shipped.series("2.12.0") != PINS["torch"]["series"]


# ---------------------------------------------------------------- the manifest

def make_build(tmp, vendor="cuda", arch="sm_89", platform="linux", command_extra=()):
    out, logs, prefix = tmp / "out", tmp / "logs", tmp / "env"
    out.mkdir()
    logs.mkdir()
    (prefix / "conda-meta").mkdir(parents=True)
    for name, data in (("libmojolearn_adapter.so", b"native"), ("mojolearn_torch.so", b"torch"),
                       ("mojolearn_ops.mojoc", b"mojoc")):
        (out / name).write_bytes(data)
    for role in manifest.FILES:
        (logs / f"{role}-command.json").write_text(json.dumps(["tool", role, *command_extra]))
    (prefix / "conda-meta" / f"max-{PINS['max']['version']}-3.14release.json").write_text(
        json.dumps(dict(name="max", version=PINS["max"]["version"])))
    (prefix / "conda-meta" / "mojo-1.0.0-release.json").write_text(json.dumps(dict(name="mojo", version="1.0.0")))
    pin = PINS["torch"]["build"][vendor]
    torch = dict(version=pin["version"], series=PINS["torch"]["series"], backend=pin["backend"],
                 backend_version="12.8", cxx11_abi=1 if platform == "linux" else None, wheel="torch.whl",
                 wheel_sha256="c" * 64, index_url=pin["index_url"], python_tag="cp312")
    (tmp / "torch-build.json").write_text(json.dumps(torch))
    return out, logs, prefix


def record(tmp, out, logs, prefix, vendor="cuda", arch="sm_89", platform="linux"):
    return manifest.record(out, vendor, arch, platform, logs, tmp / "torch-build.json", prefix,
                           "Mojo 1.0.0", "a" * 40)


def test_record_finalize_check_round_trip(tmp_path):
    out, logs, prefix = make_build(tmp_path)
    doc = record(tmp_path, out, logs, prefix)
    assert doc["unstaged"]["native"]["sha256"] == hashlib.sha256(b"native").hexdigest()
    assert doc["requirements"] == dict(torch=PINS["torch"]["requirement"], max=PINS["max"]["requirement"])
    assert "adapters/torch/registration.cpp" in doc["sources"] and "packaging/adapters/pins.json" in doc["sources"]
    (out / "libmojolearn_adapter.so").write_bytes(b"native, staged")   # stage_libs rewrote the RUNPATH
    final = manifest.finalize(out)
    assert not (out / manifest.BUILD_RECORD).exists()
    assert final["files"]["native"]["sha256"] == hashlib.sha256(b"native, staged").hexdigest()
    assert sorted(p.name for p in out.iterdir()) == sorted([*manifest.FILES.values(), manifest.MANIFEST])
    manifest.check(out, vendor="cuda", arch="sm_89")


def test_check_refuses_changed_extra_missing_and_mislabeled(tmp_path):
    out, logs, prefix = make_build(tmp_path)
    record(tmp_path, out, logs, prefix)
    manifest.finalize(out)
    with pytest.raises(manifest.ManifestError, match="architecture"):
        manifest.check(out, vendor="cuda", arch="sm_90a")
    with pytest.raises(manifest.ManifestError, match="vendor"):
        manifest.check(out, vendor="hip")
    (out / "stray.so").write_bytes(b"x")
    with pytest.raises(manifest.ManifestError, match="exactly"):
        manifest.check(out)
    (out / "stray.so").unlink()
    (out / "mojolearn_torch.so").write_bytes(b"torch, edited after the manifest")
    with pytest.raises(manifest.ManifestError, match="differ"):
        manifest.check(out)
    (out / "mojolearn_torch.so").unlink()
    with pytest.raises(manifest.ManifestError):
        manifest.check(out)


def test_record_refuses_the_test_only_route_and_unpinned_frameworks(tmp_path):
    out, logs, prefix = make_build(tmp_path, command_extra=("-DMOJOLEARN_ADAPTER_TEST_HOST_STAGING=1",))
    with pytest.raises(manifest.ManifestError, match="qualification-only"):
        record(tmp_path, out, logs, prefix)
    for path in (prefix / "conda-meta").glob("max-*.json"):
        path.write_text(json.dumps(dict(name="max", version="26.4.0")))
    (logs / "torch-command.json").write_text(json.dumps(["c++"]))
    (logs / "native-command.json").write_text(json.dumps(["mojo"]))
    (logs / "max-command.json").write_text(json.dumps(["mojo"]))
    with pytest.raises(manifest.ManifestError, match="max 26.4.0"):
        record(tmp_path, out, logs, prefix)


def test_record_refuses_a_vendor_arch_platform_that_is_no_set(tmp_path):
    out, logs, prefix = make_build(tmp_path)
    for vendor, arch, platform in (("cuda", "gfx942", "linux"), ("metal", "sm_89", "macos"), ("cuda", "sm_89", "macos"),
                                   ("hip", "apple", "linux")):
        with pytest.raises(manifest.ManifestError):
            manifest.record(out, vendor, arch, platform, logs, tmp_path / "torch-build.json", prefix, "m", "a" * 40)


def test_refresh_follows_rewritten_bytes(tmp_path):
    out, logs, prefix = make_build(tmp_path, vendor="metal", arch="apple", platform="macos")
    record(tmp_path, out, logs, prefix, vendor="metal", arch="apple", platform="macos")
    manifest.finalize(out)
    data = {p.name: p.read_bytes() for p in out.iterdir()}
    data["libmojolearn_adapter.so"] = b"stripped and re-signed"
    new = manifest.refresh(data[manifest.MANIFEST], lambda n: data[n], "shrink_wheel.py")
    digests = {n: (hashlib.sha256(d).hexdigest(), len(d)) for n, d in data.items() if n != manifest.MANIFEST}
    doc = manifest.validate(json.loads(new), digests, vendor="metal", arch="apple")
    assert doc["post_build"] == ["shrink_wheel.py"]


# ---------------------------------------------------------------- fetch_torch

PAGE = """<html><body>
<a href="https://files.example/torch-2.13.0-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=aaaa">x</a>
<a href="https://files.example/torch-2.13.0-cp314-cp314-manylinux_2_28_x86_64.whl#sha256=bbbb">x</a>
<a href="https://files.example/torch-2.13.0-cp314-cp314-manylinux_2_28_aarch64.whl#sha256=cccc">x</a>
<a href="https://files.example/torch-2.13.0-cp312-none-macosx_11_0_arm64.whl#sha256=dddd">x</a>
<a href="../torch-2.13.0%2Brocm7.2-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=eeee">x</a>
<a href="https://files.example/torch-2.12.0-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=ffff">x</a>
<a href="/whl/rocm7.2/torch-2.13.0%2Brocm7.2-cp314-cp314-manylinux_2_28_x86_64.whl#sha256=gggg">x</a>
<a href="https://download-r2.example/whl/rocm7.2/torch-2.12.0%2Brocm7.2-cp312-cp312-manylinux_2_28_x86_64.whl#sha256=hhhh">x</a>
</body></html>"""


def test_index_and_wheel_choice():
    files = fetch.index_files(PAGE, "https://download.example/whl/rocm7.2/torch/")
    names = [f[0] for f in files]
    assert "torch-2.13.0+rocm7.2-cp312-cp312-manylinux_2_28_x86_64.whl" in names
    # relative href: resolved against the index page URL
    assert files[4][1] == "https://download.example/whl/rocm7.2/torch-2.13.0%2Brocm7.2-cp312-cp312-manylinux_2_28_x86_64.whl"
    # absolute-path href: resolved against the page's host
    assert files[6][1] == "https://download.example/whl/rocm7.2/torch-2.13.0%2Brocm7.2-cp314-cp314-manylinux_2_28_x86_64.whl"
    # full URL on another host with a sha256 fragment (what download.pytorch.org emits)
    assert files[7] == ("torch-2.12.0+rocm7.2-cp312-cp312-manylinux_2_28_x86_64.whl",
                        "https://download-r2.example/whl/rocm7.2/torch-2.12.0%2Brocm7.2-cp312-cp312-manylinux_2_28_x86_64.whl",
                        "hhhh")
    pick = fetch.choose_wheel(files, "2.13.0", "linux", "cp312")
    assert (pick["wheel"], pick["index_sha256"]) == ("torch-2.13.0-cp312-cp312-manylinux_2_28_x86_64.whl", "aaaa")
    assert fetch.choose_wheel(files, "2.13.0", "linux", "cp310")["python_tag"] == "cp314"   # highest when absent
    assert fetch.choose_wheel(files, "2.13.0", "macos", "cp312")["index_sha256"] == "dddd"
    assert fetch.choose_wheel(files, "2.13.0+rocm7.2", "linux", "cp312")["index_sha256"] == "eeee"
    with pytest.raises(SystemExit):
        fetch.choose_wheel(files, "2.14.0", "linux")


def fake_torch_wheel(path, version="2.13.0", cuda="12.8", hip=None, abi=1, extra=()):
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("torch/version.py", f"__version__ = '{version}'\ndebug = False\n"
                                       f"cuda: Optional[str] = {cuda!r}\nhip: Optional[str] = {hip!r}\n")
        z.writestr("torch/include/torch/extension.h", "// header\n")
        z.writestr("torch/share/cmake/Torch/TorchConfig.cmake",
                   f'set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI={abi}")\n')
        for name in ("libtorch.so", "libtorch_cpu.so", "libc10.so", "libtorch_cuda.so", "libtorch.dylib"):
            z.writestr("torch/lib/" + name, b"lib")
        z.writestr("torch/__init__.py", "")
        for name in extra:
            z.writestr(name, "x")


def test_extract_and_describe(tmp_path):
    wheel = tmp_path / "torch.whl"
    fake_torch_wheel(wheel)
    dest = tmp_path / "tree"
    fetch.extract(wheel, dest, "linux")
    assert sorted(p.name for p in (dest / "torch/lib").iterdir()) == ["libc10.so", "libtorch.so", "libtorch_cpu.so"]
    assert not (dest / "torch/__init__.py").exists()
    pin = PINS["torch"]["build"]["cuda"]
    chosen = dict(wheel="torch.whl", url="https://x/torch.whl", python_tag="cp312")
    rec = fetch.describe(dest, pin, "linux", chosen, "d" * 64)
    assert (rec["version"], rec["series"], rec["backend"], rec["backend_version"], rec["cxx11_abi"]) == \
        ("2.13.0", "2.13", "cuda", "12.8", 1)
    with pytest.raises(SystemExit, match="rocm"):
        fetch.describe(dest, PINS["torch"]["build"]["hip"] | dict(version="2.13.0"), "linux", chosen, "d" * 64)


def test_version_rule():
    # a public pin accepts the local label PyPI's wheel reports in torch/version.py
    assert fetch.version_matches("2.13.0+cu130", "2.13.0")
    assert fetch.version_matches("2.13.0", "2.13.0")
    # a labelled pin is exact
    assert fetch.version_matches("2.13.0+rocm7.2", "2.13.0+rocm7.2")
    assert not fetch.version_matches("2.13.0+rocm7.1", "2.13.0+rocm7.2")
    assert not fetch.version_matches("2.13.0", "2.13.0+rocm7.2")
    # a different public version is refused
    assert not fetch.version_matches("2.12.0+cu130", "2.13.0")
    assert not fetch.version_matches("2.13.1", "2.13.0")
    assert not fetch.version_matches(None, "2.13.0")


def test_describe_accepts_a_labelled_wheel_for_a_public_pin(tmp_path):
    wheel = tmp_path / "torch.whl"
    fake_torch_wheel(wheel, version="2.13.0+cu130", cuda="13.0")
    dest = tmp_path / "tree"
    fetch.extract(wheel, dest, "linux")
    chosen = dict(wheel="torch.whl", url="https://x/torch.whl", python_tag="cp314")
    rec = fetch.describe(dest, PINS["torch"]["build"]["cuda"], "linux", chosen, "d" * 64)
    assert (rec["version"], rec["version_pin"], rec["series"], rec["backend"]) == \
        ("2.13.0+cu130", "2.13.0", "2.13", "cuda")
    with pytest.raises(SystemExit, match=r"says torch 2\.13\.0\+cu130, pins\.json 2\.12\.0 \(compared public version\)"):
        fetch.describe(dest, PINS["torch"]["build"]["cuda"] | dict(version="2.12.0"), "linux", chosen, "d" * 64)


def test_describe_refuses_a_labelled_pin_that_differs(tmp_path):
    wheel = tmp_path / "rocm.whl"
    fake_torch_wheel(wheel, version="2.13.0+rocm7.1", cuda=None, hip="7.1.1")
    dest = tmp_path / "rocm"
    fetch.extract(wheel, dest, "linux")
    with pytest.raises(SystemExit, match=r"says torch 2\.13\.0\+rocm7\.1, pins\.json 2\.13\.0\+rocm7\.2 \(compared exactly\)"):
        fetch.describe(dest, PINS["torch"]["build"]["hip"], "linux", dict(wheel="w", url="u", python_tag="cp312"), "e" * 64)


def test_record_keeps_the_built_and_pinned_torch_versions(tmp_path):
    out, logs, prefix = make_build(tmp_path)
    doc = json.loads((tmp_path / "torch-build.json").read_text())
    (tmp_path / "torch-build.json").write_text(json.dumps(doc | dict(version="2.13.0+cu130")))
    rec = record(tmp_path, out, logs, prefix)
    assert (rec["torch_version_built"], rec["torch_version_pin"], rec["torch"]["version"]) == \
        ("2.13.0+cu130", "2.13.0", "2.13.0+cu130")
    (tmp_path / "torch-build.json").write_text(json.dumps(doc | dict(version="2.12.0+cu130")))
    (out / manifest.BUILD_RECORD).unlink()
    with pytest.raises(manifest.ManifestError, match="2.12.0"):
        record(tmp_path, out, logs, prefix)


def test_rocm_wheel_and_unsafe_members(tmp_path):
    wheel = tmp_path / "rocm.whl"
    fake_torch_wheel(wheel, version="2.13.0+rocm7.2", cuda=None, hip="7.2.53211")
    dest = tmp_path / "rocm"
    fetch.extract(wheel, dest, "linux")
    rec = fetch.describe(dest, PINS["torch"]["build"]["hip"], "linux", dict(wheel="w", url="u", python_tag="cp312"), "e" * 64)
    assert (rec["backend"], rec["backend_version"]) == ("rocm", "7.2.53211")
    bad = tmp_path / "bad.whl"
    fake_torch_wheel(bad, extra=("torch/include/../../escape.h",))
    with pytest.raises(SystemExit, match="unsafe"):
        fetch.extract(bad, tmp_path / "bad", "linux")


def test_macos_takes_every_dylib_and_no_abi(tmp_path):
    wheel = tmp_path / "mac.whl"
    fake_torch_wheel(wheel, cuda=None)
    dest = tmp_path / "mac"
    fetch.extract(wheel, dest, "macos")
    assert sorted(p.name for p in (dest / "torch/lib").iterdir()) == ["libtorch.dylib"]
    rec = fetch.describe(dest, PINS["torch"]["build"]["metal"], "macos", dict(wheel="w", url="u", python_tag="cp312"), "f" * 64)
    assert (rec["backend"], rec["cxx11_abi"]) == ("mps", None)


# ---------------------------------------------------------------- build.sh, staging, scope

def test_build_script_parses_and_refuses_before_building(tmp_path):
    script = HERE / "build.sh"
    assert subprocess.run(["bash", "-n", str(script)]).returncode == 0
    env = dict(os.environ, MOJOLEARN_ADAPTER_TEST_HOST_STAGING="1")
    r = subprocess.run(["bash", str(script), str(tmp_path / "o"), str(tmp_path / "l"), "cuda", "sm_89"],
                       env=env, capture_output=True, text=True)
    assert r.returncode == 2 and "test-only" in r.stderr
    r = subprocess.run(["bash", str(script), str(tmp_path / "o"), str(tmp_path / "l"), "rocm", "gfx942"],
                       capture_output=True, text=True)
    assert r.returncode == 2 and not (tmp_path / "o").exists()


def test_adapter_runpaths_reach_the_staged_runtime_and_installed_torch():
    origin = "site-packages/mojolearn/cuda_native/sm_89/_adapters"
    torch_paths = stage_libs.ADAPTER_RUNPATHS[stage_libs.ADAPTER_TORCH].split(":")
    assert "$ORIGIN" in torch_paths
    assert posixpath.normpath(posixpath.join(origin, "../../../../torch/lib")) == "site-packages/torch/lib"
    assert "$ORIGIN/../../../../torch/lib" in torch_paths
    native = stage_libs.ADAPTER_RUNPATHS[stage_libs.ADAPTER_NATIVE].split(":")
    # the shared runtime at mojolearn/.libs, as the identical tier finds it
    assert posixpath.normpath(posixpath.join(origin, "../../../.libs")) == "site-packages/mojolearn/.libs"
    assert "$ORIGIN/../../../.libs" in native


def test_the_torch_registration_may_need_only_system_torch_and_the_adapter():
    ok = ["libmojolearn_adapter.so", "libtorch_cpu.so", "libc10.so", "libstdc++.so.6", "libc.so.6", "libgcc_s.so.1"]
    assert stage_libs.adapter_torch_errors(ok) == []
    assert stage_libs.adapter_torch_errors(ok + ["libpython3.12.so.1.0", "libtorch_python.so"]) == \
        ["libpython3.12.so.1.0", "libtorch_python.so"]


def test_the_identical_passes_cover_the_cuda_adapter_library_only():
    assert ptx_contract.is_identical_cuda("mojolearn/cuda_native/sm_89/_adapters/libmojolearn_adapter.so")
    assert ptx_contract.is_identical_cuda("mojolearn/cuda/sm_89/_adapters/libmojolearn_adapter.so")
    assert not ptx_contract.is_identical_cuda("mojolearn/cuda_native/sm_89/_adapters/mojolearn_torch.so")
    assert not ptx_contract.is_identical_cuda("mojolearn/hip_native/gfx942/_adapters/libmojolearn_adapter.so")


def test_adapter_sources_are_native_build_inputs_in_both_inventories():
    for rel in ("packaging/adapters/build.sh", "packaging/adapters/manifest.py", "packaging/adapters/pins.json",
                "packaging/adapters/fetch_torch.py", "adapters/torch/registration.cpp", "adapters/native/abi.h",
                "adapters/native/c_api.mojo", "python/mojolearn/adapters/torch.py"):
        assert qualification.is_native_source(rel), rel
    assert not qualification.is_native_source("packaging/adapters/test_adapters.py")
    shell = (REPO / "tools/linux_surface_qualification.sh").read_text()
    for needle in ("rel.startswith('packaging/adapters/') and not rel.startswith('packaging/adapters/test_')",
                   "'adapters/torch/registration.cpp', 'adapters/native/abi.h'"):
        assert needle in shell


def test_the_adapter_modules_pass_the_wheel_python_audit():
    """Each imports only its own framework, lazily, and no platform math (portable_math/wheel.py)."""
    sys.path.insert(0, str(REPO / "packaging/portable_math"))
    wheel = load("portable_math_wheel_adapters", REPO / "packaging/portable_math/wheel.py")
    for leaf in ("__init__.py", "_shipped.py", "torch.py", "max.py"):
        path = REPO / "python/mojolearn/adapters" / leaf
        rel = "mojolearn/adapters/" + leaf
        assert wheel.dependency_errors(path, rel) == [], rel
        assert wheel.python_math_errors(path, rel) == [], rel
        assert wheel.numpy_errors(path, rel) == [], rel
    # the allowance is per file: torch in max.py is still refused
    assert wheel.ADAPTER_LAZY_IMPORTS["mojolearn/adapters/max.py"] == {"max"}
