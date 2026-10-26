import copy
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import pytest

import release_macos_assembly as assembly


def index(stem="_mojolearn_dev_one", member="_mojolearn_one", key="a" * 64):
    return {"format": 1, "layout": "development", "files": {
        f".development/{key}/{stem}.so": [member]}, "provenance": {stem: key}}


def test_flatten_preserves_logical_groups_and_original_provenance():
    original = index()
    saved = copy.deepcopy(original)
    flat, pairs = assembly.flatten_index(original)
    assert original == saved
    assert {k: v for k, v in flat.items() if k != "files"} == {
        k: v for k, v in original.items() if k != "files"}
    assert flat["files"] == {"_mojolearn_dev_one.so": ["_mojolearn_one"]}
    assert pairs == [(next(iter(original["files"])), "_mojolearn_dev_one.so", ["_mojolearn_one"], "a" * 64)]


@pytest.mark.parametrize("damage", ["escape", "wrong-key", "duplicate-member", "unknown-schema", "empty"])
def test_flatten_refuses_ambiguous_or_escaping_inputs(damage):
    doc = index()
    if damage == "escape":
        doc["files"] = {"../_mojolearn_dev_one.so": ["_mojolearn_one"]}
    elif damage == "wrong-key":
        doc["provenance"]["_mojolearn_dev_one"] = "b" * 64
    elif damage == "duplicate-member":
        doc["files"][f".development/{'b' * 64}/_mojolearn_dev_two.so"] = ["_mojolearn_one"]
        doc["provenance"]["_mojolearn_dev_two"] = "b" * 64
    elif damage == "unknown-schema":
        doc["dispatch_override"] = True
    else:
        doc["files"].clear()
        doc["provenance"].clear()
    with pytest.raises(ValueError):
        assembly.flatten_index(doc)


def test_prepare_copies_admitted_groups_without_mutating_or_linking_originals(tmp_path, monkeypatch):
    root, output, cache = (tmp_path / name for name in ("source", "assembly", "cache"))
    pkg = root / "python/mojolearn"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("# source-only fixture\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qm", "source fixture"], check=True)
    admission = {"active": [], "indexes": {}, "receipts": {}, "support": []}
    for mode, letter in (("host", "a"), ("identical", "b")):
        doc = index(key=letter * 64)
        admission["indexes"][mode] = doc
        binary = pkg / mode / next(iter(doc["files"]))
        binary.parent.mkdir(parents=True)
        binary.write_bytes((mode + "-compiled-bytes").encode())
        assembly.write(pkg / mode / assembly.INDEX, doc)
        receipt = cache / "artifacts" / (letter * 64) / "receipt.json"
        receipt.parent.mkdir(parents=True)
        assembly.write(receipt, {"sha256": assembly.sha(binary)})
        admission["active"].append(str(binary))
        admission["receipts"][letter * 64] = str(receipt)
    helper = pkg / ".dylibs/libMojolearnMath.dylib"
    helper.parent.mkdir()
    helper.write_bytes(b"already-compiled-helper")
    receipt = cache / "support.json"
    assembly.write(receipt, {"sha256": assembly.sha(helper)})
    admission["support"] = [{"name": "portable_math", "installed": str(helper), "receipt": str(receipt)}]

    def admit(self, name, argv, **kwargs):
        assert name == "admission", "preparation must not invoke compilation, staging or runtime checks"
        assembly.write(output / "admission.json", admission)

    monkeypatch.setattr(assembly.sys, "platform", "darwin")
    monkeypatch.setattr(assembly.Runner, "run", admit)
    result = assembly.prepare(root, cache, output, Path("/usr/bin/python3"))
    assert result["counts"] == {"host": 1, "identical": 1}
    assert result["compiler_invoked"] is False and result["runtime_qualified"] is False
    for row in result["groups"]:
        source = Path(row["source"]["path"])
        copied = output / "source/python/mojolearn" / row["staged"]
        assert copied.read_bytes() == source.read_bytes()
        assert copied.stat().st_ino != source.stat().st_ino
        copied.write_bytes(b"simulated-signing-transformation")
        assert assembly.sha(source) == row["source"]["sha256"]
    assert helper.read_bytes() == b"already-compiled-helper"


def test_native_source_refresh_retains_frontend_and_unrelated_inputs():
    original = {"native": "old", "shared": "same", "frontend": "same", "builder": "same"}
    current = dict(original, native="new")
    native = {"native": "new", "shared": "same"}
    frontends = {"torch": {"frontend": "same", "shared": "same", "builder": "same"}}
    assert assembly.native_source_refresh(original, current, native, frontends) == current
    for name in ("shared", "frontend", "builder", "unknown"):
        with pytest.raises(ValueError, match="non-native sources changed"):
            assembly.native_source_refresh(original, dict(current, **{name: "changed"}), native, frontends)
    with pytest.raises(ValueError, match="non-native sources changed"):
        assembly.native_source_refresh(original, {k: v for k, v in current.items() if k != "frontend"}, native, frontends)


def wheel_fixture(tmp_path):
    pkg = tmp_path / "pkg"
    document = {"groups": [], "indexes": {}}
    data = {"mojolearn/_adapters/libmojolearn_adapter.so": b"native",
            "mojolearn/_adapters/mojolearn_torch.so": b"torch",
            "mojolearn/_adapters/mojolearn_ops.mojoc": b"max",
            "mojolearn/_adapters/adapters.json": b"{}"}
    for mode in assembly.MODES:
        flat, _ = assembly.flatten_index(index())
        path = pkg / mode / assembly.INDEX
        path.parent.mkdir(parents=True)
        assembly.write(path, flat)
        document["indexes"][mode] = {"staged_document": flat}
        document["groups"].append({"staged": mode + "/_mojolearn_dev_one.so"})
        data[f"mojolearn/{mode}/{assembly.INDEX}"] = path.read_bytes()
        data[f"mojolearn/{mode}/_mojolearn_dev_one.so"] = b"stripped-and-signed"
    return pkg, document, data


@pytest.mark.parametrize("damage", [None, "missing", "stray", "changed-index", "missing-max"])
def test_wheel_membership_gate_covers_every_flat_group_and_adapter(tmp_path, damage):
    pkg, document, data = wheel_fixture(tmp_path)
    if damage == "missing":
        del data["mojolearn/host/_mojolearn_dev_one.so"]
    elif damage == "stray":
        data["mojolearn/stale.so"] = b"stale"
    elif damage == "changed-index":
        data[f"mojolearn/host/{assembly.INDEX}"] = b'{}'
    elif damage == "missing-max":
        del data["mojolearn/_adapters/mojolearn_ops.mojoc"]
    wheel = tmp_path / "fixture.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, contents in data.items():
            archive.writestr(name, contents)
    if damage:
        with pytest.raises(ValueError):
            assembly.verify_wheel(wheel, pkg, document)
    else:
        result = assembly.verify_wheel(wheel, pkg, document)
        assert len(result) == 5
        assert result["mojolearn/host/_mojolearn_dev_one.so"] == hashlib.sha256(b"stripped-and-signed").hexdigest()


def test_old_complete_adapter_manifest_cannot_bypass_per_role_proof(tmp_path, monkeypatch):
    # adapter_copy must demand explicit current native/transitive admission;
    # the old manifest's local-source hashes alone are insufficient.
    from types import SimpleNamespace
    api = SimpleNamespace(check=lambda *args, **kwargs: {"schema": "mojolearn.adapters.v1"})
    monkeypatch.setattr(assembly, "module", lambda *args: api)
    proof = tmp_path / "old-manifest.json"
    assembly.write(proof, {"schema": "mojolearn.adapters.v1", "complete": True})
    with pytest.raises(ValueError, match="per-role adapter composition proof"):
        assembly.adapter_copy(tmp_path, tmp_path / "old", proof, tmp_path / "staged", {})


def test_frozen_source_refresh_removes_old_python_and_preserves_staged_native(tmp_path, monkeypatch):
    root, output = tmp_path / "repo", tmp_path / "assembly"
    source_pkg = root / "python/mojolearn"
    source_pkg.mkdir(parents=True)
    (source_pkg / "__init__.py").write_text("# frozen source\n")
    (source_pkg / "new_name.py").write_text("NEW = True\n")
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(["git", "-C", str(root), "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
                    "commit", "-qm", "frozen fixture"], check=True)
    pkg = output / "source/python/mojolearn"
    (pkg / "host").mkdir(parents=True)
    (pkg / "obsolete.py").write_text("must not survive\n")
    native = pkg / "host/group.so"
    native.write_bytes(b"admitted-staged-bytes")
    helper = tmp_path / "math.dylib"
    helper.write_bytes(b"helper")
    document = dict(source_root=str(root), source_commit="0" * 40, python="unused", cache="unused",
                    groups=[], indexes={}, staging_tools={}, dependency_inputs={},
                    portable_math={"source": assembly.reference(helper)}, staged_files={"host/group.so": assembly.sha(native)})

    def admission(self, name, *args, **kwargs):
        if name == "frozen-admission":
            assembly.write(output / "frozen-admission.json", {"active": [], "indexes": {}})
        else:
            assert name == "frozen-input-equality"
            assembly.write(output / "frozen-input-equality.json", {"complete": True})

    monkeypatch.setattr(assembly.Runner, "run", admission)
    monkeypatch.setattr(assembly, "adapter_copy", lambda *args, **kwargs: None)
    assembly.refresh_source(output, document, assembly.Runner(output, document), "unused", "unused")
    assert not (pkg / "obsolete.py").exists()
    assert (pkg / "new_name.py").read_text() == "NEW = True\n"
    assert native.read_bytes() == b"admitted-staged-bytes"
    assert (output / "prepared-source/python/mojolearn/obsolete.py").exists()
    assert document["prepared_source_commit"] == "0" * 40
    assert document["source_commit"] == assembly.revision(root)


def test_changed_staging_checker_invalidates_check_reuse(tmp_path, monkeypatch):
    root, output = tmp_path / "repo", tmp_path / "assembly"
    root.mkdir()
    output.mkdir()
    (root / "checker.py").write_text("changed checker")
    document = dict(source_root=str(root), python="unused", cache="unused", groups=[], indexes={},
                    staging_tools={"checker.py": "0" * 64})
    monkeypatch.setattr(assembly, "git", lambda *args: "frozen")
    monkeypatch.setattr(assembly.Runner, "run", lambda *args, **kwargs:
                        assembly.write(output / "frozen-admission.json", {"active": [], "indexes": {}}))
    with pytest.raises(ValueError, match="check must be rerun"):
        assembly.refresh_source(output, document, assembly.Runner(output, document), "unused", "unused")
