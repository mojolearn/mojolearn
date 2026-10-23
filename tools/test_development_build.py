"""Development build dependency/cache/activation contracts; no Mojo compilation."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bincache
import development_build as dev
import merged_bindings as merged

SOURCE_REPO = Path(__file__).resolve().parent.parent


FACTS = {"available": True, "compiler_sha256": "test-compiler", "version": "test", "os": "test"}
ENV = {"MOJOLEARN_GPU_ARCHS": "sm_89"}
A, B, BYTE, HOST = "_mojolearn_a", "_mojolearn_b", "_mojolearn_byte_lm", "_mojolearn_tokenizer_host"


def write(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


@pytest.fixture
def repo(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    for path in dev.BUILD_INPUTS:
        write(root, path, "# synthetic build input\n")
    release = {"index_file": "_mojolearn_merged.json", "dirs": {
        "identical": {"groups": {"_mojolearn_native": [A, B]}, "optional": {BYTE: ["linux", "darwin"]}},
        "host": {"groups": {"_mojolearn_native_host": [HOST]}, "optional": {}},
        "fast": {"groups": {"_mojolearn_native": [A, B]}, "optional": {}}},
        "darwin_metal1": [B], "darwin_metal1_suffix": "_m1"}
    partition = {"groups": {"identical": {"trees": [A], "dense": [B], "neural": [BYTE]},
                            "host": {"runtime": [HOST]}}}
    write(root, "bindings/merged_groups.json", json.dumps(release))
    write(root, dev.MANIFEST, json.dumps(partition))
    for member, imports in [(A, "from core.shared import value\nfrom tree.leaf import leaf\n"),
                             (B, "from core.shared import value\n"), (BYTE, ""), (HOST, "")]:
        write(root, f"bindings/{member}.mojo", imports + f"def build_{member}() -> PythonObject:\n    pass\n")
    write(root, "core/shared.mojo", "comptime value = 1\n")
    write(root, "tree/leaf.mojo", "comptime leaf = 1\n")
    monkeypatch.setattr(merged, "REPO", root)
    monkeypatch.setattr(merged, "MANIFEST", root / "bindings/merged_groups.json")
    monkeypatch.setattr(merged, "DEVELOPMENT_MANIFEST", root / dev.MANIFEST)
    monkeypatch.setattr(dev, "REPO", root)
    monkeypatch.setattr(dev, "toolchain_facts", lambda _root: FACTS)
    return root


def plan(repo, tmp_path, env=None, **kwargs):
    return dev.make_plan(repo, tmp_path / "cache", ["identical"], env or ENV, facts=FACTS, plat="linux", **kwargs)


def keys(value):
    return {row["stem"]: row["key"] for row in value["groups"]}


def changed(before, after):
    a, b = keys(before), keys(after)
    return {name for name in a if a[name] != b[name]}


def cache_row(cache, row, data=b"native-test-bytes"):
    target = cache / "artifacts" / row["key"]
    target.mkdir(parents=True, exist_ok=True)
    path = target / (row["stem"] + ".so")
    path.write_bytes(data)
    dev.atomic_json(target / "receipt.json", {"key": row["key"], "fields": row["fields"],
        "exit_code": 0, "sha256": bincache.sha256_file(path)})
    return path


def test_single_family_and_shared_import_invalidation(repo, tmp_path):
    before = plan(repo, tmp_path)
    write(repo, "tree/leaf.mojo", "comptime leaf = 2\n")
    family = plan(repo, tmp_path)
    assert changed(before, family) == {"_mojolearn_dev_trees"}
    write(repo, "core/shared.mojo", "comptime value = 2\n")
    shared = plan(repo, tmp_path)
    assert changed(family, shared) == {"_mojolearn_dev_trees", "_mojolearn_dev_dense"}
    write(repo, "unrelated/new.mojo", "comptime nothing = 1\n")
    write(repo, "docs/new.md", "Documentation only.\n")
    assert keys(shared) == keys(plan(repo, tmp_path))


def test_unknown_import_widens_and_records_reason(repo, tmp_path):
    path = repo / f"bindings/{A}.mojo"
    path.write_text("from unsupported_local_package import value\n" + path.read_text())
    before = plan(repo, tmp_path)
    row = next(r for r in before["groups"] if r["members"] == [A])
    assert row["fields"]["source"]["scope"] == "tree-fallback"
    assert row["fields"]["source"]["unresolved"] == [{"file": f"bindings/{A}.mojo", "import": "unsupported_local_package"}]
    write(repo, "unrelated/new.mojo", "comptime value = 4\n")
    assert changed(before, plan(repo, tmp_path)) == {"_mojolearn_dev_trees"}


def test_flags_target_toolchain_and_builder_invalidate(repo, tmp_path):
    base = plan(repo, tmp_path)
    assert len(changed(base, plan(repo, tmp_path, dict(ENV, MOJOLEARN_MOJO_BUILD_FLAGS="-O2")))) == 3
    assert len(changed(base, plan(repo, tmp_path, {"MOJOLEARN_GPU_ARCHS": "gfx942"}))) == 3
    changed_compiler = dev.make_plan(repo, tmp_path / "cache", ["identical"], ENV,
                                    facts=dict(FACTS, compiler_sha256="another"), plat="linux")
    assert len(changed(base, changed_compiler)) == 3
    write(repo, "bindings/merged_build.sh", "# changed compiler command\n")
    assert len(changed(base, plan(repo, tmp_path))) == 3


def test_generator_inputs_invalidate_only_owner(repo, tmp_path):
    path = repo / f"bindings/{A}.mojo"
    path.write_text("from tokenizer.impl.unicode_class import value\n" + path.read_text())
    write(repo, "tokenizer/impl/unicode_class.mojo", "from .unicode_table_generated import value\n")
    write(repo, "tokenizer/impl/unicode_table_generated.mojo", "comptime value = 1\n")
    write(repo, "tokenizer/tools/generate.py", "# first generator\n")
    before = plan(repo, tmp_path)
    write(repo, "tokenizer/tools/generate.py", "# repaired generator\n")
    assert changed(before, plan(repo, tmp_path)) == {"_mojolearn_dev_trees"}


def test_release_plan_unchanged_and_optional_has_one_owner(repo):
    release_bytes = merged.MANIFEST.read_bytes()
    release = merged.plan("identical", "linux", True)
    assert release == [("_mojolearn_native", "default", [A, B, BYTE])]
    for plat in ("linux", "darwin"):
        enabled = merged.plan("identical", plat, True, "development")
        assert sum(BYTE in members for _, _, members in enabled) == 1
        disabled = merged.plan("identical", plat, False, "development")
        assert all(BYTE not in members for _, _, members in disabled)
    metal = merged.plan("identical", "darwin", True, "development")
    assert ("_mojolearn_dev_dense_m1", "metal1", [B]) in metal
    assert merged.plan("identical", "linux", True) == release
    assert merged.MANIFEST.read_bytes() == release_bytes
    with pytest.raises(ValueError, match="identical and host only"):
        merged.plan("fast", "darwin", True, "development")


@pytest.mark.parametrize("change", ["missing", "duplicate", "unknown"])
def test_partition_cannot_silently_omit_or_duplicate_new_bindings(repo, change):
    doc = json.loads(merged.DEVELOPMENT_MANIFEST.read_text())
    groups = doc["groups"]["identical"]
    if change == "missing":
        del groups["trees"]
    elif change == "duplicate":
        groups["trees"].append(B)
    else:
        groups["trees"].append("_mojolearn_new")
    merged.DEVELOPMENT_MANIFEST.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="partition differs"):
        merged.plan("identical", "linux", True, "development")


def test_selection_reuses_current_cache_and_rejects_unknown_group(repo, tmp_path):
    first = plan(repo, tmp_path, selection=["trees"])
    assert first["counts"] == {"groups": 3, "reuse": 0, "build_selected": 1, "build_unselected": 2}
    for row in first["groups"]:
        cache_row(tmp_path / "cache", row)
    warm = plan(repo, tmp_path)
    assert warm["counts"]["reuse"] == 3
    assert warm["counts"]["build_selected"] == 0
    write(repo, "tree/leaf.mojo", "comptime leaf = 3\n")
    affected = plan(repo, tmp_path, selection=["identical:trees"])
    assert affected["counts"] == {"groups": 3, "reuse": 2, "build_selected": 1, "build_unselected": 0}
    with pytest.raises(dev.BuildError, match="matches nothing"):
        plan(repo, tmp_path, selection=["misspelled"])


def test_tampered_or_failed_cache_entry_is_never_reused(repo, tmp_path):
    value = plan(repo, tmp_path)
    row = value["groups"][0]
    artifact = cache_row(tmp_path / "cache", row)
    assert dev.cached_artifact(tmp_path / "cache", row) == artifact
    artifact.write_bytes(b"changed")
    assert dev.cached_artifact(tmp_path / "cache", row) is None
    cache_row(tmp_path / "cache", row)
    receipt_path = artifact.parent / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["exit_code"] = 137
    dev.atomic_json(receipt_path, receipt)
    assert dev.cached_artifact(tmp_path / "cache", row) is None


def test_partial_selection_never_replaces_complete_index(repo, tmp_path):
    value = plan(repo, tmp_path, selection=["trees"])
    index = write(repo, "python/mojolearn/identical/_mojolearn_merged.json", '{"files":{"old.so":[]}}')
    original = index.read_bytes()
    cache_row(tmp_path / "cache", value["groups"][0])
    result = dev.activate(value, repo, tmp_path / "cache")
    assert result["state"] == "incomplete"
    assert len(result["missing"]) == 2
    assert index.read_bytes() == original


def test_activation_stages_complete_immutable_set_and_keeps_release_bytes(repo, tmp_path):
    value = plan(repo, tmp_path)
    for row in value["groups"]:
        cache_row(tmp_path / "cache", row)
    release = write(repo, "python/mojolearn/identical/_mojolearn_native.so", "old-release-bytes")
    result = dev.activate(value, repo, tmp_path / "cache")
    assert result["state"] == "activated"
    directory = release.parent
    index = json.loads((directory / "_mojolearn_merged.json").read_text())
    assert sorted(n for names in index["files"].values() for n in names) == sorted([A, B, BYTE])
    assert all((directory / filename).read_bytes() == b"native-test-bytes" for filename in index["files"])
    assert all(filename.startswith(".development/") for filename in index["files"])
    assert release.read_text() == "old-release-bytes"
    snapshot = (directory / "_mojolearn_merged.json").read_bytes()
    installed = next(iter(index["files"]))
    (directory / installed).write_bytes(b"modified-in-place")
    with pytest.raises(dev.BuildError, match="immutable installed artifact"):
        dev.activate(value, repo, tmp_path / "cache")
    assert (directory / "_mojolearn_merged.json").read_bytes() == snapshot


def test_member_override_refused_before_index_publication(repo, tmp_path):
    value = plan(repo, tmp_path)
    for row in value["groups"]:
        cache_row(tmp_path / "cache", row)
    write(repo, f"python/mojolearn/identical/{A}.so", "override")
    with pytest.raises(dev.BuildError, match="shadow"):
        dev.activate(value, repo, tmp_path / "cache")
    assert not (repo / "python/mojolearn/identical/_mojolearn_merged.json").exists()


def test_compile_failure_and_source_change_do_not_enter_cache(repo, tmp_path, monkeypatch):
    row = plan(repo, tmp_path)["groups"][0]
    monkeypatch.setattr(dev, "run_logged", lambda *_: 137)
    result = dev.compile_group(row, repo, tmp_path / "cache", ENV, tmp_path / "evidence")
    assert result["exit_code"] == 137
    assert dev.cached_artifact(tmp_path / "cache", row) is None

    def changing_build(command, *_):
        output = Path(command[command.index("--outdir") + 1])
        (output / (row["stem"] + ".so")).write_bytes(b"completed")
        write(repo, "tree/leaf.mojo", "comptime leaf = 999\n")
        return 0
    monkeypatch.setattr(dev, "run_logged", changing_build)
    result = dev.compile_group(row, repo, tmp_path / "cache", ENV, tmp_path / "evidence")
    assert result["exit_code"] == 1
    assert "inputs changed" in result["reason"]
    assert dev.cached_artifact(tmp_path / "cache", row) is None


def test_successful_build_is_cached_without_reinvoking_compiler(repo, tmp_path, monkeypatch):
    row = plan(repo, tmp_path)["groups"][0]
    commands = []
    def compiler(command, *_):
        commands.append(command)
        output = Path(command[command.index("--outdir") + 1])
        (output / (row["stem"] + ".so")).write_bytes(b"completed")
        return 0
    monkeypatch.setattr(dev, "run_logged", compiler)
    first = dev.compile_group(row, repo, tmp_path / "cache", ENV, tmp_path / "evidence")
    second = dev.compile_group(row, repo, tmp_path / "cache", ENV, tmp_path / "evidence")
    assert first["state"] == "built"
    assert second["state"] == "reused"
    assert len(commands) == 1
    assert "--layout" in commands[0] and "development" in commands[0]
    assert str(tmp_path / "cache" / "work" / row["key"] / "roots") in commands[0]


def test_group_root_is_stable_and_only_rewritten_on_content_change(repo, tmp_path):
    from types import SimpleNamespace
    output = tmp_path / "root.mojo"
    args = SimpleNamespace(stem="_mojolearn_dev_trees", members=A, out=str(output))
    merged.cmd_root(args)
    stamp = output.stat().st_mtime_ns
    merged.cmd_root(args)
    assert output.stat().st_mtime_ns == stamp
    args.members = A + "," + B
    merged.cmd_root(args)
    assert "build_" + B in output.read_text()


def test_parallelism_respects_cpu_and_memory():
    assert dev.pool_size(8, 2, 4, (16, 20 * 2 ** 30)) == 4
    assert dev.pool_size(8, 4, 1, (8, 100 * 2 ** 30)) == 2
    with pytest.raises(dev.BuildError, match="measured"):
        dev.pool_size(2, 1, None)
    with pytest.raises(dev.BuildError, match="cannot fit"):
        dev.pool_size(2, 1, 20, (8, 10 * 2 ** 30))


def test_host_target_normalization_and_refusals(repo, tmp_path):
    env = dev.build_environment("host", dict(ENV, MOJOLEARN_TARGET_COLUMN="nvidia"))
    assert env["MOJOLEARN_TARGET_COLUMN"] == "cpu"
    assert "MOJOLEARN_GPU_ARCHS" not in env
    with pytest.raises(dev.BuildError, match="identical only"):
        plan(repo, tmp_path, dict(ENV, MOJOLEARN_NUMERIC_MODE="fast"))
    with pytest.raises(dev.BuildError, match="fault-injection"):
        plan(repo, tmp_path, dict(ENV, MOJOLEARN_BUILD_DEFINES="MOJOLEARN_TEST_SABOTAGE=1"))
    with pytest.raises(dev.BuildError, match="outside"):
        dev.make_plan(repo, repo / "cache", ["host"], ENV, facts=FACTS, plat="linux")


def test_development_shell_executes_the_fingerprinted_compiler(repo, tmp_path):
    # Capture the actual shell command with inert tool shims. This exercises
    # group/root/index assembly without invoking any native compiler.
    for relative in ("bindings/merged_build.sh", "bindings/build_defines.sh", "tools/merged_bindings.py"):
        shutil.copy2(SOURCE_REPO / relative, repo / relative)
    compiler = write(repo, ".pixi/envs/default/bin/mojo", "#!/bin/sh\nexit 99\n")
    compiler.chmod(0o755)
    bin_dir = tmp_path / "bin"
    uname = write(bin_dir, "uname", '#!/bin/sh\ncase "$1" in -m) echo x86_64;; *) echo Linux;; esac\n')
    uname.chmod(0o755)
    capture = tmp_path / "compiler-argv.json"
    pixi = write(bin_dir, "pixi", '#!' + sys.executable + '\nimport json, os, pathlib, sys\n'
        'pathlib.Path(os.environ["CAPTURE"]).write_text(json.dumps(sys.argv[1:]))\n'
        'pathlib.Path(sys.argv[sys.argv.index("-o") + 1]).write_bytes(b"inert-native-output")\n')
    pixi.chmod(0o755)
    output = tmp_path / "output"
    env = dict(os.environ, PATH=str(bin_dir) + os.pathsep + os.environ["PATH"], CAPTURE=str(capture),
               MOJOLEARN_MERGED_PYTHON=sys.executable, MOJOLEARN_GPU_ARCHS="sm_89",
               MOJOLEARN_NUMERIC_MODE="identical", MOJOLEARN_PACKAGE_BYTE_LM="1",
               PIXI_ENVIRONMENT_NAME="unrelated-active-environment")
    result = subprocess.run(["sh", "bindings/merged_build.sh", "gpu", "--layout", "development",
        "--group", "_mojolearn_dev_trees", "--outdir", str(output), "--rootdir", str(tmp_path / "roots")],
        cwd=repo, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    command = json.loads(capture.read_text())
    assert command[:9] == ["run", "--manifest-path", str(repo / "pixi.toml"), "--environment", "default",
                           "--locked", "--no-install", "--executable", str(compiler)]
    assert command[9] == "build"
    assert (output / "_mojolearn_dev_trees.so").read_bytes() == b"inert-native-output"


def test_return_to_release_index_removes_development_metadata(repo, tmp_path):
    from types import SimpleNamespace
    directory = tmp_path / "installed"
    dev.atomic_json(directory / "_mojolearn_merged.json", {"format": 1, "layout": "development",
        "files": {".development/old/tree.so": [A], ".development/old/dense.so": [B, BYTE]},
        "provenance": {"tree": "old"}})
    merged.cmd_index(SimpleNamespace(outdir=str(directory), stem="_mojolearn_native", members=",".join([A, B, BYTE])))
    index = json.loads((directory / "_mojolearn_merged.json").read_text())
    assert index == {"format": 1, "files": {"_mojolearn_native.so": [A, B, BYTE]}}
