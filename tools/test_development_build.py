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
C_FACTS = {"available": True, "compiler": "cc", "resolved": "/usr/bin/cc", "sha256": "test-cc",
           "version": "cc test", "machine": "x86_64"}
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
    for path in dev.PORTABLE_MATH_INPUTS:
        if not (root / path).exists():
            write(root, path, "/* synthetic portable math input */\n")
    monkeypatch.setattr(merged, "REPO", root)
    monkeypatch.setattr(merged, "MANIFEST", root / "bindings/merged_groups.json")
    monkeypatch.setattr(merged, "DEVELOPMENT_MANIFEST", root / dev.MANIFEST)
    monkeypatch.setattr(dev, "REPO", root)
    monkeypatch.setattr(dev, "toolchain_facts", lambda _root: FACTS)
    monkeypatch.setattr(dev, "c_toolchain_facts", lambda *_: C_FACTS)
    monkeypatch.setattr(dev.build_process, "cancelled", lambda: False)
    monkeypatch.setattr(dev, "available_resources", lambda: (16, 100 * 2 ** 30))
    monkeypatch.setattr(dev, "physical_memory", lambda: 128 * 2 ** 30)
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
    path = target / row.get("file", row["stem"] + ".so")
    path.write_bytes(data)
    dev.atomic_json(target / "receipt.json", {"key": row["key"], "fields": row["fields"],
        "exit_code": 0, "sha256": bincache.sha256_file(path)})
    return path


def cache_plan(cache, value):
    """Cache every group and support artifact of a plan."""
    for row in value["groups"] + value["support"]:
        cache_row(cache, row)


def test_larger_closures_start_first_without_changing_plan_or_keys(repo, tmp_path):
    value = plan(repo, tmp_path, env=dict(ENV, MOJOLEARN_PACKAGE_BYTE_LM="0"))
    original = json.dumps(value, sort_keys=True)
    rows = value["groups"]
    ordered = dev.build_submission_order(reversed(rows))
    assert [r["stem"] for r in ordered] == ["_mojolearn_dev_trees", "_mojolearn_dev_dense"]
    assert {id(r) for r in ordered} == {id(r) for r in rows}
    assert json.dumps(value, sort_keys=True) == original
    # Equal source closures have a stable tie break, independent of the
    # manifest/submission input order, without looking at artifact state.
    tie = dict(rows[1], stem="_mojolearn_dev_alphabetic_first", key="different-key")
    expected = [tie, rows[1]]
    assert dev.build_submission_order(expected) == expected
    assert dev.build_submission_order(reversed(expected)) == expected


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
    write(repo, "bindings/merged_build.sh", 'base_flags="-O3"\n')
    assert len(changed(base, plan(repo, tmp_path))) == 3


def test_tool_edits_that_keep_the_recipe_keep_every_key(repo, tmp_path):
    base = plan(repo, tmp_path)
    for name in ("tools/development_build.py", "tools/build_process.py", "tools/build_sizing.py",
                 "tools/bincache.py", "tools/merged_bindings.py"):
        path = repo / name
        write(repo, name, (path.read_text() if path.exists() else "") + "# controller edit\nx = 1\n")
    # Comments and blank lines in the build scripts are not part of the recipe.
    for name in dev.BUILD_INPUTS:
        path = repo / name
        path.write_text(path.read_text() + "\n# a new comment\n\n")
    after = plan(repo, tmp_path)
    assert keys(after) == keys(base)
    assert after["support"][0]["key"] == base["support"][0]["key"]


def test_recipe_changes_change_the_key(repo, tmp_path, monkeypatch):
    base = plan(repo, tmp_path)
    write(repo, "bindings/build_defines.sh", 'MOJOLEARN_BUILD_DEFINE_FLAGS="-D NEW_DEFINE=1"\n')
    assert len(changed(base, plan(repo, tmp_path))) == 3
    defines = plan(repo, tmp_path)
    assert len(changed(defines, plan(repo, tmp_path, dict(ENV, MOJOLEARN_BUILD_DEFINES="X=1")))) == 3
    assert len(changed(defines, plan(repo, tmp_path, dict(ENV, MOJOLEARN_COMPILE_JOBS="3")))) == 3
    original = dev.group_command
    monkeypatch.setattr(dev, "group_command", lambda *args: original(*args) + ["--extra-flag"])
    assert len(changed(defines, plan(repo, tmp_path))) == 3


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
    assert len(result["missing"]) == 3
    assert "support:portable_math" in result["missing"]
    assert index.read_bytes() == original


def test_activation_stages_complete_immutable_set_and_keeps_release_bytes(repo, tmp_path):
    value = plan(repo, tmp_path)
    cache_plan(tmp_path / "cache", value)
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
    cache_plan(tmp_path / "cache", value)
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


def test_unrelated_writes_during_compile_do_not_trip_the_input_guard(repo, tmp_path, monkeypatch):
    path = repo / f"bindings/{A}.mojo"
    path.write_text("from tokenizer.impl.unicode_class import value\n" + path.read_text())
    write(repo, "tokenizer/impl/unicode_class.mojo", "comptime value = 1\n")
    write(repo, "tokenizer/tools/generate.py", "# generator\n")
    row = next(r for r in plan(repo, tmp_path)["groups"] if r["stem"] == "_mojolearn_dev_trees")
    evidence = tmp_path / "evidence"

    def noisy_build(command, *_):
        output = Path(command[command.index("--outdir") + 1])
        (output / (row["stem"] + ".so")).write_bytes(b"completed")
        # What compilers, Python, editors and this tool write while a group
        # compiles: none of it reaches the compiler, so none of it is a key.
        write(repo, "core/__pycache__/shared.cpython-314.pyc", "bytecode")
        write(repo, "tokenizer/tools/__pycache__/generate.cpython-314.pyc", "bytecode")
        write(repo, "tokenizer/tools/generate.py.swp", "editor swap")
        write(repo, "tokenizer/tools/.DS_Store", "finder")
        write(repo, "notes/README.txt", "not an input")
        write(repo, "python/mojolearn/identical/_mojolearn_merged.json", "{}")
        write(repo, "unrelated/never_imported.mojo", "comptime other = 2\n")
        (evidence / "sibling-receipt.json").write_text("{}")
        shared = repo / "core/shared.mojo"
        os.utime(shared, (shared.stat().st_atime + 60, shared.stat().st_mtime + 60))  # mtime only
        return 0
    monkeypatch.setattr(dev, "run_logged", noisy_build)
    evidence.mkdir()
    result = dev.compile_group(row, repo, tmp_path / "cache", ENV, evidence)
    assert result["state"] == "built", result.get("reason")
    assert dev.cached_artifact(tmp_path / "cache", row) is not None


def test_source_edit_during_compile_still_trips_the_guard_and_names_it(repo, tmp_path, monkeypatch):
    row = plan(repo, tmp_path)["groups"][0]

    def editing_build(command, *_):
        output = Path(command[command.index("--outdir") + 1])
        (output / (row["stem"] + ".so")).write_bytes(b"completed")
        write(repo, "tree/leaf.mojo", "comptime leaf = 2\n")
        return 0
    monkeypatch.setattr(dev, "run_logged", editing_build)
    result = dev.compile_group(row, repo, tmp_path / "cache", ENV, tmp_path / "evidence")
    assert result["exit_code"] == 1
    assert "inputs changed" in result["reason"]
    assert "source.files.tree/leaf.mojo" in result["reason"]
    assert dev.cached_artifact(tmp_path / "cache", row) is None


def test_version_banner_ignores_stderr_noise(tmp_path):
    # The L4 failure: tcmalloc printed a PID- and address-bearing warning on
    # stderr before the banner, and that line became part of every key.
    tool = tmp_path / "mojo"
    tool.write_text("#!/bin/sh\n"
                    "echo \"$$ external/tcmalloc+/tcmalloc/internal/system_allocator.h:713] "
                    "Warning: Unable to mbind memory (errno=1, base=0x$$00000000)\" >&2\n"
                    "echo 'Mojo 1.0.0 (ed45d567)'\n")
    tool.chmod(0o755)
    first = dev.version_banner([str(tool), "--version"])
    assert first == "Mojo 1.0.0 (ed45d567)"
    assert dev.version_banner([str(tool), "--version"]) == first
    assert dev.version_banner([str(tmp_path / "missing"), "--version"]) is None


def test_changed_fields_names_only_moved_inputs():
    before = {"source": {"files": {"a.mojo": "1", "b.mojo": "2"}}, "toolchain": {"version": "Mojo 1"}}
    after = {"source": {"files": {"a.mojo": "1", "b.mojo": "3"}}, "toolchain": {"version": "Mojo 1"}}
    assert dev.changed_fields(before, after) == ["source.files.b.mojo"]
    assert dev.changed_fields(before, before) == []


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


@pytest.mark.parametrize("workers,reservation,resources", [
    (8, None, (2, 100 * 2 ** 30)),
    (2, 90, (8, 100 * 2 ** 30)),
    (1, 1, (8, 0)),
    (1, 0, (8, 100 * 2 ** 30)),
    (1, float("nan"), (8, 100 * 2 ** 30)),
    (1, float("inf"), (8, 100 * 2 ** 30)),
])
def test_single_compiler_must_fit_real_allocation(workers, reservation, resources):
    with pytest.raises(dev.BuildError):
        dev.pool_size(1, workers, reservation, resources)


def test_serial_and_parallel_budgets_preserve_available_memory_headroom():
    gib = 2 ** 30
    serial = dev.resource_budget(1, 8, None, (8, 100 * gib))
    assert serial["jobs"] == 1
    assert serial["rss_bytes"] == 80 * gib
    assert serial["min_available_bytes"] == 20 * gib
    assert serial["reservation_source"] == "available-allocation"
    parallel = dev.resource_budget(4, 8, 30, (32, 100 * gib))
    assert parallel["jobs"] == 2
    assert parallel["rss_bytes"] == 30 * gib
    assert parallel["min_available_bytes"] == 20 * gib
    assert parallel["reservation_source"] == "measured"


def test_guard_controls_are_not_compiler_inputs_or_cache_keys(repo, tmp_path, monkeypatch):
    limits = dict(seconds=321, stall_seconds=12, lock_seconds=9,
                  rss_bytes=6 * 2 ** 30, min_available_bytes=2 * 2 ** 30)
    env = dict(ENV, **{dev.GUARD_ENV: json.dumps(limits)})
    assert keys(plan(repo, tmp_path)) == keys(plan(repo, tmp_path, env))
    calls = []

    def guarded(command, **kwargs):
        calls.append((command, kwargs))
        return dict(complete=True, exit_code=0, status="passed")

    monkeypatch.setattr(dev.build_process, "run", guarded)
    assert dev.run_logged(["fake-compiler"], repo, env, tmp_path / "build.log") == 0
    command, passed = calls[0]
    assert command == ["fake-compiler"]
    assert passed["env"] == ENV
    assert passed["seconds"] == 321
    assert passed["stall_seconds"] == 12
    assert passed["rss_bytes"] == 6 * 2 ** 30
    assert passed["min_available_bytes"] == 2 * 2 ** 30


@pytest.mark.parametrize("status,code", [
    ("timeout", 124), ("stalled", 124), ("rss-limit", 124),
    ("memory-pressure", 124), ("interrupted", 130), ("controller-error", 125),
])
def test_guard_verdict_is_retained_and_cannot_publish_cache(repo, tmp_path, monkeypatch, status, code):
    row = plan(repo, tmp_path)["groups"][0]

    def guarded(command, **kwargs):
        output = Path(command[command.index("--outdir") + 1])
        (output / (row["stem"] + ".so")).write_bytes(b"partial-native-output")
        result = dict(complete=True, exit_code=code, status=status)
        log = kwargs["log"]
        dev.atomic_json(log.with_suffix(log.suffix + ".execution.json"), result)
        return result

    monkeypatch.setattr(dev.build_process, "run", guarded)
    result = dev.compile_group(row, repo, tmp_path / "cache", ENV, tmp_path / "evidence")
    assert result["exit_code"] == code
    assert result["execution"]["status"] == status
    assert dev.cached_artifact(tmp_path / "cache", row) is None
    retained = json.loads((tmp_path / "evidence" / (row["mode"] + "-" + row["stem"] + ".json")).read_text())
    assert retained == result


@pytest.mark.parametrize("verdict", [
    dict(complete=False, exit_code=0, status="running"),
    dict(complete=True, exit_code=0, status="interrupted"),
])
def test_zero_exit_without_guard_completion_is_refused(repo, tmp_path, monkeypatch, verdict):
    monkeypatch.setattr(dev.build_process, "run", lambda *_, **__: verdict)
    with pytest.raises(dev.BuildError, match="did not certify"):
        dev.run_logged(["fake-compiler"], repo, ENV, tmp_path / "build.log")


@pytest.mark.parametrize("produce_empty", [False, True])
def test_stale_or_empty_output_cannot_satisfy_success(repo, tmp_path, monkeypatch, produce_empty):
    row = plan(repo, tmp_path)["groups"][0]
    output = tmp_path / "cache/work" / row["key"] / "output" / (row["stem"] + ".so")
    output.parent.mkdir(parents=True)
    output.write_bytes(b"abandoned-earlier-build")

    def compiler(*_):
        assert not output.exists()
        if produce_empty:
            output.touch()
        return 0

    monkeypatch.setattr(dev, "run_logged", compiler)
    result = dev.compile_group(row, repo, tmp_path / "cache", ENV, tmp_path / "evidence")
    assert result["exit_code"] == 1
    assert "nonempty" in result["reason"]
    assert dev.cached_artifact(tmp_path / "cache", row) is None


def test_empty_artifact_with_matching_hash_is_not_reused(repo, tmp_path):
    row = plan(repo, tmp_path)["groups"][0]
    cache_row(tmp_path / "cache", row, b"")
    assert dev.cached_artifact(tmp_path / "cache", row) is None


def test_cancellation_after_compiler_success_prevents_cache_receipt(repo, tmp_path, monkeypatch):
    row = plan(repo, tmp_path)["groups"][0]
    cancelled = [False]
    monkeypatch.setattr(dev.build_process, "cancelled", lambda: cancelled[0])

    def compiler(command, *_):
        output = Path(command[command.index("--outdir") + 1])
        (output / (row["stem"] + ".so")).write_bytes(b"completed")
        cancelled[0] = True
        return 0

    monkeypatch.setattr(dev, "run_logged", compiler)
    result = dev.compile_group(row, repo, tmp_path / "cache", ENV, tmp_path / "evidence")
    assert result["exit_code"] == 130
    assert result["state"] == "interrupted"
    assert dev.cached_artifact(tmp_path / "cache", row) is None


def test_lock_wait_has_a_deadline(monkeypatch, tmp_path):
    from types import SimpleNamespace
    monkeypatch.setattr(dev.build_process, "cancelled", lambda: False)
    ticks = iter([0.0, 1.0])
    monkeypatch.setattr(dev, "time", SimpleNamespace(monotonic=lambda: next(ticks)))

    def busy(*_):
        raise BlockingIOError("another compiler holds the lock")

    monkeypatch.setattr(dev.fcntl, "flock", busy)
    with pytest.raises(dev.BuildError, match="lock wait exceeded"):
        with dev.lock(tmp_path / "busy.lock", 0.5):
            pytest.fail("busy lock was entered")


def configure_main(repo, monkeypatch):
    monkeypatch.setattr(dev.os, "environ", dict(ENV))
    monkeypatch.setattr(merged, "default_platform", lambda: "linux")
    monkeypatch.setattr(bincache, "first_line", lambda *_: "synthetic-source-commit")


def test_warm_main_has_zero_compiler_attempts_and_needs_no_compiler_reservation(repo, tmp_path, monkeypatch):
    configure_main(repo, monkeypatch)
    cache_plan(tmp_path / "cache", plan(repo, tmp_path))

    def forbidden(*_):
        pytest.fail("warm identical cache must not start or reserve a compiler")

    monkeypatch.setattr(dev, "run_logged", forbidden)
    monkeypatch.setattr(dev, "available_resources", forbidden)
    evidence = tmp_path / "evidence"
    assert dev.main(["build", "--sets", "identical", "--cache", str(tmp_path / "cache"),
                     "--evidence", str(evidence), "--jobs", "8"]) == 0
    report = json.loads((evidence / "result.json").read_text())
    assert report["state"] == "activated"
    assert report["compile_attempts"] == 0
    assert report["jobs"] == 0


def test_main_cancellation_scope_covers_workers_and_preserves_indexes(repo, tmp_path, monkeypatch):
    configure_main(repo, monkeypatch)
    active, cancelled = [], []

    @dev.contextlib.contextmanager
    def scope():
        active.append(True)
        try:
            yield
        finally:
            active.pop()

    def interrupted(*_):
        assert active
        raise KeyboardInterrupt("synthetic controller interrupt")

    monkeypatch.setattr(dev.build_process, "cancellation_scope", scope)
    monkeypatch.setattr(dev.build_process, "cancel_active", lambda: cancelled.append(True))
    monkeypatch.setattr(dev, "compile_group", interrupted)
    index = write(repo, "python/mojolearn/identical/_mojolearn_merged.json", "previous-index")
    evidence = tmp_path / "evidence"
    assert dev.main(["build", "--sets", "identical", "--cache", str(tmp_path / "cache"),
                     "--evidence", str(evidence)]) == 130
    assert cancelled
    assert not active
    assert index.read_text() == "previous-index"
    assert json.loads((evidence / "result.json").read_text())["state"] == "interrupted"


def test_main_passes_measured_guard_budget_and_retains_compile_attempt_count(repo, tmp_path, monkeypatch):
    configure_main(repo, monkeypatch)
    seen = []

    def guarded_failure(command, root, env, log):
        seen.append(json.loads(env[dev.GUARD_ENV]))
        return 124

    monkeypatch.setattr(dev, "run_logged", guarded_failure)
    index = write(repo, "python/mojolearn/identical/_mojolearn_merged.json", "previous-index")
    evidence = tmp_path / "evidence"
    assert dev.main(["build", "--sets", "identical", "--cache", str(tmp_path / "cache"),
                     "--evidence", str(evidence), "--jobs", "2", "--memory-per-job-gib", "30",
                     "--group-seconds", "321", "--stall-seconds", "12", "--lock-seconds", "9"]) == 1
    report = json.loads((evidence / "result.json").read_text())
    assert len(seen) == report["plan"]["counts"]["build_selected"] == report["compile_attempts"]
    assert report["jobs"] == 2
    assert all(value["rss_bytes"] == 30 * 2 ** 30 and value["min_available_bytes"] == 20 * 2 ** 30
               and value["seconds"] == 321 and value["stall_seconds"] == 12
               and value["lock_seconds"] == 9 for value in seen)
    assert all(result["exit_code"] == 124 for result in report["results"])
    assert index.read_text() == "previous-index"


@pytest.mark.parametrize("option,value", [
    ("--group-seconds", "nan"), ("--group-seconds", "0"),
    ("--stall-seconds", "inf"), ("--stall-seconds", "0"),
    ("--lock-seconds", "-1"),
])
def test_main_refuses_unbounded_controls_before_starting_compiler(repo, tmp_path, monkeypatch, option, value):
    configure_main(repo, monkeypatch)
    monkeypatch.setattr(dev, "run_logged", lambda *_: pytest.fail("invalid controls must not execute"))
    evidence = tmp_path / "evidence"
    assert dev.main(["build", "--sets", "identical", "--cache", str(tmp_path / "cache"),
                     "--evidence", str(evidence), option, value]) == 1
    assert "finite and positive" in json.loads((evidence / "result.json").read_text())["error"]


def test_activation_waits_for_all_existing_index_locks_before_writing(repo, tmp_path, monkeypatch):
    value = dev.make_plan(repo, tmp_path / "cache", ["identical", "host"], ENV, facts=FACTS, plat="linux")
    cache_plan(tmp_path / "cache", value)
    indexes = [write(repo, f"python/mojolearn/{mode}/_mojolearn_merged.json", "previous-index")
               for mode in ("identical", "host")]
    seen = []

    @dev.contextlib.contextmanager
    def bounded_lock(path, seconds):
        seen.append((path, seconds))
        if len(seen) == 2:
            raise dev.BuildError("lock wait exceeded")
        yield

    monkeypatch.setattr(dev, "lock", bounded_lock)
    with pytest.raises(dev.BuildError, match="lock wait"):
        dev.activate(value, repo, tmp_path / "cache", 17)
    assert all(index.read_text() == "previous-index" for index in indexes)
    expected = {Path(dev.tempfile.gettempdir()) / ("mojolearn-merged-index-" +
                dev.hashlib.sha256(os.path.realpath(index.parent).encode()).hexdigest()[:16] + ".lock")
                for index in indexes}
    assert {path for path, _ in seen} == expected
    assert all(seconds == 17 for _, seconds in seen)


@pytest.mark.parametrize("mode,plat,arch,workers", [
    ("identical", "linux", "gfx942", "8"),
    ("identical", "linux", "gfx950", "8"),
    ("identical", "linux", "sm_89", "2"),
    ("identical", "linux", "sm_120a", "2"),
    ("host", "linux", "gfx942", "2"),
    ("host", "linux", "", "2"),
    ("identical", "darwin", "", "1"),
    ("host", "darwin", "", "1"),
])
def test_supported_compiler_worker_defaults(mode, plat, arch, workers):
    env = {"MOJOLEARN_GPU_ARCHS": arch} if arch else {}
    assert dev.build_environment(mode, env, plat)["MOJOLEARN_COMPILE_JOBS"] == workers
    env["MOJOLEARN_COMPILE_JOBS"] = "3"
    assert dev.build_environment(mode, env, plat)["MOJOLEARN_COMPILE_JOBS"] == "3"


def test_worker_overrides_are_effective_and_keyed(repo, tmp_path):
    env = {"MOJOLEARN_GPU_ARCHS": "gfx942"}
    base = plan(repo, tmp_path, env)
    explicit_default = plan(repo, tmp_path, dict(env, MOJOLEARN_COMPILE_JOBS="8"))
    assert keys(base) == keys(explicit_default)
    overridden = plan(repo, tmp_path, dict(env, MOJOLEARN_COMPILE_JOBS="4"))
    assert len(changed(base, overridden)) == 3
    amd_override = dict(env, MOJOLEARN_AMD_COMPILE_JOBS="6")
    assert dev.build_environment("identical", amd_override, "linux")["MOJOLEARN_COMPILE_JOBS"] == "6"
    assert len(changed(base, plan(repo, tmp_path, amd_override))) == 3
    both = dict(amd_override, MOJOLEARN_COMPILE_JOBS="4")
    assert dev.build_environment("identical", both, "linux")["MOJOLEARN_COMPILE_JOBS"] == "4"
    assert dev.build_environment("host", amd_override, "linux")["MOJOLEARN_COMPILE_JOBS"] == "2"


def test_mixed_group_pool_reserves_the_largest_selected_compiler(repo, tmp_path):
    value = dev.make_plan(repo, tmp_path / "cache", ["identical", "host"],
                          {"MOJOLEARN_GPU_ARCHS": "gfx942"}, facts=FACTS, plat="linux")
    assert dev.planned_compiler_workers(value) == 8
    assert dev.pool_size(8, dev.planned_compiler_workers(value), 1, (16, 100 * 2 ** 30)) == 2
    for row in value["groups"]:
        row["selected"] = row["mode"] == "host"
    assert dev.planned_compiler_workers(value) == 2
    for row in value["groups"]:
        row["action"] = "reuse"
    assert dev.planned_compiler_workers(value) == 1


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
    for relative in ("bindings/merged_build.sh", "bindings/build_defines.sh", "bindings/compile_jobs.sh",
                     "tools/merged_bindings.py"):
        # 095db6a93 dropped tools/*.py from dev.BUILD_INPUTS, so the synthetic repo has no tools/ directory
        (repo / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(SOURCE_REPO / relative, repo / relative)
    # This assembly fixture does not exercise Unicode generation. Supply its
    # already-generated input, as a normal compiler checkout does.
    write(repo, "tokenizer/impl/unicode_table_generated.mojo", "# synthetic generated table\n")
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


def fake_c_compiler(calls, data=b"\x7fELF portable math"):
    def compiler(command, root, env, log):
        calls.append(command)
        assert "stage.build" in command[3]  # packaging/portable_math/stage.py build()
        Path(command[-2]).write_bytes(data)
        return 0
    return compiler


def test_portable_math_is_keyed_built_once_and_reused(repo, tmp_path, monkeypatch):
    first = plan(repo, tmp_path)
    (row,) = first["support"]
    assert row["install"] == ".libs/libMojolearnMath.so" and row["action"] == "build"
    calls = []
    monkeypatch.setattr(dev, "run_logged", fake_c_compiler(calls))
    built = dev.build_support(row, repo, tmp_path / "cache", ENV, tmp_path / "evidence")
    assert built["state"] == "built", built.get("reason")
    assert dev.build_support(row, repo, tmp_path / "cache", ENV, tmp_path / "evidence")["state"] == "reused"
    assert len(calls) == 1
    assert plan(repo, tmp_path)["support"][0]["action"] == "reuse"
    # Its inputs are the C source, the constants, stage.py and the C compiler;
    # none of them is a group input, and no group input moves it.
    write(repo, "packaging/portable_math/portable_math.c", "/* repaired */\n")
    edited = plan(repo, tmp_path)
    assert edited["support"][0]["key"] != row["key"] and changed(first, edited) == set()
    monkeypatch.setattr(dev, "c_toolchain_facts", lambda *_: dict(C_FACTS, sha256="other-cc"))
    other_cc = plan(repo, tmp_path)["support"][0]["key"]
    assert other_cc not in (row["key"], edited["support"][0]["key"])
    write(repo, "tree/leaf.mojo", "comptime leaf = 7\n")
    assert plan(repo, tmp_path)["support"][0]["key"] == other_cc


def test_portable_math_source_edit_during_build_is_refused(repo, tmp_path, monkeypatch):
    (row,) = plan(repo, tmp_path)["support"]
    calls = []
    def editing(command, *rest):
        fake_c_compiler(calls)(command, *rest)
        write(repo, "packaging/portable_math/powers_of_ten.h", "/* edited */\n")
        return 0
    monkeypatch.setattr(dev, "run_logged", editing)
    result = dev.build_support(row, repo, tmp_path / "cache", ENV, tmp_path / "evidence")
    assert result["exit_code"] == 1
    assert "inputs.packaging/portable_math/powers_of_ten.h" in result["reason"]
    assert dev.cached_artifact(tmp_path / "cache", row) is None


def test_activation_refuses_without_portable_math_and_installs_it(repo, tmp_path):
    value = plan(repo, tmp_path)
    for row in value["groups"]:
        cache_row(tmp_path / "cache", row)
    index = write(repo, "python/mojolearn/identical/_mojolearn_merged.json", "previous-index")
    result = dev.activate(value, repo, tmp_path / "cache")
    assert result["state"] == "incomplete" and result["missing"] == ["support:portable_math"]
    assert index.read_text() == "previous-index"
    library = repo / "python/mojolearn/.libs/libMojolearnMath.so"
    assert not library.exists()
    cache_row(tmp_path / "cache", value["support"][0], b"portable-math-bytes")
    result = dev.activate(value, repo, tmp_path / "cache")
    assert result["state"] == "activated"
    assert library.read_bytes() == b"portable-math-bytes"
    assert result["support"] == {"portable_math": str(library)}
    with pytest.raises(dev.BuildError, match="support artifacts"):
        dev.activate({k: v for k, v in value.items() if k != "support"}, repo, tmp_path / "cache")


def test_portable_math_uses_the_macos_loader_path(repo, tmp_path):
    (row,) = dev.make_plan(repo, tmp_path / "cache", ["host"], {}, facts=FACTS, plat="darwin")["support"]
    assert (row["file"], row["install"]) == ("libMojolearnMath.dylib", ".dylibs/libMojolearnMath.dylib")


def test_main_builds_portable_math_after_groups_and_activates_host(repo, tmp_path, monkeypatch):
    configure_main(repo, monkeypatch)
    value = dev.make_plan(repo, tmp_path / "cache", ["host"], ENV, facts=FACTS, plat="linux")
    for row in value["groups"]:
        cache_row(tmp_path / "cache", row)
    calls = []
    compiler = fake_c_compiler(calls)
    def guarded(command, root, env, log):
        if command[:2] == ["sh", "tokenizer/tools/gen_unicode_table.sh"]:
            return 0
        assert json.loads(env[dev.GUARD_ENV])["jobs"] == 1
        return compiler(command, root, env, log)
    monkeypatch.setattr(dev, "run_logged", guarded)
    evidence = tmp_path / "evidence"
    assert dev.main(["build", "--sets", "host", "--cache", str(tmp_path / "cache"),
                     "--evidence", str(evidence)]) == 0
    report = json.loads((evidence / "result.json").read_text())
    assert report["state"] == "activated" and report["compile_attempts"] == 1 and len(calls) == 1
    assert (repo / "python/mojolearn/.libs/libMojolearnMath.so").is_file()


GIB = 2 ** 30


def test_unmeasured_budget_uses_the_group_floor_on_a_loaded_machine():
    # The M4 case: 16 GiB total, 3.7 GiB available; 80% of available would
    # kill a group that needs ~10 GB, so the group floor sets the limit.
    budget = dev.resource_budget(1, 1, None, (10, int(3.7 * GIB)), floor_bytes=12 * GIB,
                                 total_bytes=16 * GIB)
    assert budget["jobs"] == 1 and budget["rss_bytes"] == 12 * GIB
    assert budget["reservation_source"] == "group-floor"
    assert budget["min_available_bytes"] == 0
    # A large idle box keeps the 80%-of-available default.
    roomy = dev.resource_budget(1, 2, None, (128, 300 * GIB), floor_bytes=12 * GIB, total_bytes=755 * GIB)
    assert roomy["rss_bytes"] == 240 * GIB and roomy["reservation_source"] == "available-allocation"


def test_budget_below_a_known_need_is_refused_before_compiling():
    with pytest.raises(dev.BuildError, match="cannot fit.*--memory-per-job-gib"):
        dev.resource_budget(1, 1, None, (8, 3 * GIB), floor_bytes=12 * GIB, total_bytes=8 * GIB)
    with pytest.raises(dev.BuildError, match="below a recorded compiler need.*--memory-per-job-gib 13"):
        dev.resource_budget(1, 1, 4, (8, 100 * GIB), recorded_bytes=int(12.5 * GIB))


def test_peaks_are_recorded_and_size_the_next_plan_without_changing_keys(repo, tmp_path):
    cache = tmp_path / "cache"
    before = plan(repo, tmp_path)
    row = before["groups"][0]
    assert row["recorded_need_bytes"] == 0
    dev.record_peak(cache, row, {"status": "rss-limit", "peak_rss_bytes": 4 * GIB})
    assert dev.recorded_need(cache, row["mode"], row["stem"]) == 6 * GIB
    dev.record_peak(cache, row, {"status": "stalled", "peak_rss_bytes": 2 * GIB})  # lower bound only raises
    assert dev.recorded_need(cache, row["mode"], row["stem"]) == 6 * GIB
    dev.record_peak(cache, row, {"status": "passed", "peak_rss_bytes": 10 * GIB})
    after = plan(repo, tmp_path)
    assert abs(after["groups"][0]["recorded_need_bytes"] - 12 * GIB) <= 1
    assert keys(after) == keys(before)


def test_limit_reason_and_peak_reach_the_group_receipt(repo, tmp_path, monkeypatch):
    row = plan(repo, tmp_path)["groups"][0]
    evidence = tmp_path / "evidence"
    evidence.mkdir()

    def limited(command, root, env, log):
        Path(str(log) + ".execution.json").write_text(json.dumps({
            "complete": True, "status": "rss-limit", "exit_code": 124, "peak_rss_bytes": 3 * GIB,
            "reason": "rss limit 2960077619 bytes (2.76 GiB) exceeded: process group at 3221225472 bytes"}))
        return 124
    monkeypatch.setattr(dev, "run_logged", limited)
    result = dev.compile_group(row, repo, tmp_path / "cache", ENV, evidence)
    assert result["exit_code"] == 124 and result["peak_rss_bytes"] == 3 * GIB
    assert result["reason"].startswith("rss limit 2960077619 bytes")
    assert dev.recorded_need(tmp_path / "cache", row["mode"], row["stem"]) == int(4.5 * GIB)


def test_main_refuses_a_too_small_machine_before_starting_a_compiler(repo, tmp_path, monkeypatch):
    configure_main(repo, monkeypatch)
    monkeypatch.setattr(dev, "available_resources", lambda: (8, 3 * GIB))
    monkeypatch.setattr(dev, "physical_memory", lambda: 8 * GIB)
    monkeypatch.setattr(dev, "run_logged", lambda *_: pytest.fail("must refuse before compiling"))
    evidence = tmp_path / "evidence"
    assert dev.main(["build", "--sets", "identical", "--cache", str(tmp_path / "cache"),
                     "--evidence", str(evidence)]) == 1
    error = json.loads((evidence / "result.json").read_text())["error"]
    assert "cannot fit" in error and "--memory-per-job-gib" in error


@pytest.mark.parametrize("flag", ["-O0", "-O 0", "-O=0", "--optimization-level 0",
                                  "--optimization-level=0", "-D KEEP=1 -O0 -O3"])
@pytest.mark.parametrize("variable", ["MOJOLEARN_MOJO_BUILD_FLAGS", "MOJOLEARN_BUILD_EXTRA_DEFINES"])
def test_unoptimized_development_images_rejected_before_cache(repo, tmp_path, flag, variable):
    with pytest.raises(dev.BuildError, match="incorrect zero"):
        plan(repo, tmp_path, dict(ENV, **{variable: flag}))


def test_standard_development_optimization_flags_preserved():
    assert dev.build_environment("identical", dict(ENV, MOJOLEARN_MOJO_BUILD_FLAGS=""), "linux")["MOJOLEARN_MOJO_BUILD_FLAGS"] == ""
    assert "MOJOLEARN_MOJO_BUILD_FLAGS" not in dev.build_environment("identical", ENV, "linux")
    assert dev.build_environment("identical", dict(ENV, MOJOLEARN_MOJO_BUILD_FLAGS="-O2"), "linux")["MOJOLEARN_MOJO_BUILD_FLAGS"] == "-O2"


# Parallel group scheduler (--parallel auto). Pure functions and a main run
# with a fake compiler; nothing here compiles.

def test_packing_respects_the_memory_budget_and_the_slot_count():
    # Biggest first: 9 + 4 + 2 + 1 fit 16; the second 4 would make 20.
    assert dev.next_launches([9 * GIB, 4 * GIB, 4 * GIB, 2 * GIB, 1 * GIB], [], 16 * GIB, 8, 32 * GIB) == [0, 1, 3, 4]
    # At most one group per slot (core or cap).
    assert dev.next_launches([GIB] * 10, [], 100 * GIB, 3, 100 * GIB) == [0, 1, 2]
    # Running groups keep their reservation and their slots.
    assert dev.next_launches([3 * GIB, 2 * GIB], [12 * GIB], 16 * GIB, 8, 32 * GIB) == [0]
    assert dev.next_launches([GIB, GIB], [GIB, GIB], 100 * GIB, 3, 100 * GIB) == [0]
    assert dev.next_launches([], [GIB], 100 * GIB, 3, 100 * GIB) == []


def test_a_group_larger_than_the_budget_runs_alone_or_is_refused():
    # A group above the available budget but below the machine ceiling runs alone.
    assert dev.next_launches([11 * GIB, GIB], [], 10 * GIB, 8, int(12.8 * GIB)) == [0]
    # While it runs nothing fits beside it.
    assert dev.next_launches([GIB], [11 * GIB], 10 * GIB, 8, int(12.8 * GIB)) == []
    with pytest.raises(dev.BuildError, match="cannot fit.*--memory-per-job-gib"):
        dev.next_launches([14 * GIB], [], 10 * GIB, 8, int(12.8 * GIB))


def test_biggest_recorded_need_starts_first():
    rows = [{"mode": "host", "stem": name, "fields": {"source": {"files": ["f"] * files}}}
            for name, files in (("small", 50), ("big", 1), ("mid", 10), ("tie", 99))]
    ordered, needs = dev.packing_order(rows, [GIB, 12 * GIB, 4 * GIB, GIB])
    assert [r["stem"] for r in ordered] == ["big", "mid", "tie", "small"]
    assert needs == [12 * GIB, 4 * GIB, GIB, GIB]


def test_missing_records_take_the_flag_then_the_set_maximum_then_the_floor():
    rows = [{"mode": "host", "stem": "a", "recorded_need_bytes": 3 * GIB},
            {"mode": "host", "stem": "b", "recorded_need_bytes": 0},
            {"mode": "host", "stem": "c", "recorded_need_bytes": 7 * GIB},
            {"mode": "identical", "stem": "d", "recorded_need_bytes": 0}]
    assert dev.group_needs(rows, rows, None, 12 * GIB) == [
        (3 * GIB, "recorded"), (7 * GIB, "set-max"), (7 * GIB, "recorded"), (12 * GIB, "group-floor")]
    assert dev.group_needs(rows, rows, 2, 12 * GIB) == [
        (3 * GIB, "recorded"), (2 * GIB, "memory-per-job-gib"), (7 * GIB, "recorded"),
        (2 * GIB, "memory-per-job-gib")]


def test_group_cap_variable_parallel_and_jobs():
    assert dev.group_cap(None, None, None) is None
    assert dev.group_cap(dev.parse_parallel("auto"), None, "6") == 6
    assert dev.group_cap(dev.parse_parallel("4"), None, "6") == 4
    assert dev.group_cap(None, 2, "") == 2
    assert dev.group_cap(dev.parse_parallel("1"), None, None) == 1
    for bad in ("0", "-1", "x"):
        with pytest.raises(dev.BuildError):
            dev.parse_parallel(bad)
    with pytest.raises(dev.BuildError, match=dev.GROUP_WORKERS_ENV):
        dev.group_cap(None, None, "0")


def test_threads_budget_and_pressure_floor():
    assert dev.launch_threads(16, [], [None] * 3) == [6, 5, 5]
    assert dev.launch_threads(4, [], [None] * 9) == [1, 1, 1, 1]
    assert dev.launch_threads(192, [], [None]) == [192]
    # Other load shrinks the budget; our own running groups do not.
    assert dev.current_budget(80 * GIB, 100 * GIB, []) == 80 * GIB
    assert dev.current_budget(80 * GIB, 70 * GIB, [30 * GIB]) == 80 * GIB
    assert dev.current_budget(80 * GIB, 50 * GIB, []) == 40 * GIB
    # The 20% system reserve, lowered for a group that runs alone above the budget.
    assert dev.pressure_floor(100 * GIB, 80 * GIB, 100 * GIB, 30 * GIB) == 20 * GIB
    assert dev.pressure_floor(10 * GIB, 8 * GIB, 10 * GIB, 9 * GIB) == 1 * GIB


def test_group_worker_cap_is_never_a_key_field(repo, tmp_path):
    assert keys(plan(repo, tmp_path)) == keys(plan(repo, tmp_path, dict(ENV, **{dev.GROUP_WORKERS_ENV: "3"})))


def test_auto_compile_threads_follow_the_caller_and_amd():
    assert dev.auto_compile_jobs("identical", {"MOJOLEARN_GPU_ARCHS": "sm_89"}, "linux")
    assert dev.auto_compile_jobs("host", {}, "darwin")
    assert not dev.auto_compile_jobs("host", {"MOJOLEARN_COMPILE_JOBS": "1"}, "darwin")
    assert not dev.auto_compile_jobs("identical", {"MOJOLEARN_GPU_ARCHS": "gfx942"}, "linux")
    assert dev.auto_compile_jobs("host", {"MOJOLEARN_GPU_ARCHS": "gfx942"}, "linux")


def fake_group_compiler(seen):
    def compiler(command, root, env, log):
        seen.append((command[command.index("--group") + 1], env["MOJOLEARN_COMPILE_JOBS"],
                     json.loads(env[dev.GUARD_ENV])))
        output = Path(command[command.index("--outdir") + 1])
        (output / (command[command.index("--group") + 1] + ".so")).write_bytes(b"built")
        Path(str(log) + ".execution.json").write_text(json.dumps(
            {"complete": True, "status": "passed", "exit_code": 0, "peak_rss_bytes": GIB}))
        return 0
    return compiler


def test_main_packs_groups_keys_their_threads_and_reuses_them(repo, tmp_path, monkeypatch):
    configure_main(repo, monkeypatch)
    monkeypatch.setattr(dev, "physical_memory", lambda: 128 * GIB)
    cache = tmp_path / "cache"
    cache_plan_support = plan(repo, tmp_path)
    for row in cache_plan_support["support"]:
        cache_row(cache, row)
    seen = []
    monkeypatch.setattr(dev, "run_logged", fake_group_compiler(seen))
    evidence = tmp_path / "evidence"
    assert dev.main(["build", "--sets", "identical", "--cache", str(cache), "--evidence", str(evidence)]) == 0
    report = json.loads((evidence / "result.json").read_text())
    groups = len(cache_plan_support["groups"])
    # No records: every group needs the 12 GiB Linux floor; 80 GiB holds all of them.
    assert len(seen) == groups == report["compile_attempts"]
    assert report["groups_concurrent_max"] == groups
    assert report["host_cores"] == 16 and report["host_memory_bytes"] == 128 * GIB
    assert report["scheduler"]["parallel"] == "auto" and report["scheduler"]["slots"] == 16
    # All start on the first tick and share the 16 cores: threads sum to the cores.
    assert sum(int(jobs) for _, jobs, _ in seen) == 16
    assert {int(jobs) for _, jobs, _ in seen} <= {16 // groups, 16 // groups + 1}
    # Packing reserves 12 GiB; the runaway guard allows its documented 50% headroom.
    assert all(limits["rss_bytes"] == 18 * GIB and limits["min_available_bytes"] == 20 * GIB
               for _, _, limits in seen)
    assert all(r["workers_at_launch"] in (16 // groups, 16 // groups + 1) and r["peak_rss_bytes"] == GIB
               and r["queue_wait_seconds"] >= 0 and r["compile_seconds"] >= 0 for r in report["results"])
    assert sorted(r["groups_running_at_launch"] for r in report["results"]) == list(range(1, groups + 1))
    assert max(r["threads_running_at_launch"] for r in report["results"]) == 16
    assert report["threads_running_max"] == 16
    assert report["state"] == "activated"
    # The keys differ from the platform default's, and the next plan finds them.
    built = {r["stem"]: r["key"] for r in report["plan"]["groups"]}
    assert built != keys(cache_plan_support)
    again = plan(repo, tmp_path)
    assert keys(again) == built and all(r["action"] == "reuse" for r in again["groups"])
    # Recorded peaks (1 GiB passed -> 1.2 GiB) size the next packing.
    assert all(abs(r["recorded_need_bytes"] - int(1.2 * GIB)) <= 1 for r in again["groups"])


def test_parallel_one_is_the_serial_budget(repo, tmp_path, monkeypatch):
    configure_main(repo, monkeypatch)
    cache = tmp_path / "cache"
    for row in plan(repo, tmp_path)["support"]:
        cache_row(cache, row)
    seen = []
    monkeypatch.setattr(dev, "run_logged", fake_group_compiler(seen))
    evidence = tmp_path / "evidence"
    assert dev.main(["build", "--sets", "identical", "--cache", str(cache), "--evidence", str(evidence),
                     "--parallel", "1"]) == 0
    report = json.loads((evidence / "result.json").read_text())
    assert report["jobs"] == 1 and report["groups_concurrent_max"] == 1
    assert report["scheduler"] == {"parallel": 1, "slots": 1}
    # The old serial run: platform default compiler workers, 80% of available memory.
    assert {jobs for _, jobs, _ in seen} == {"2"}
    assert all(limits["jobs"] == 1 and limits["rss_bytes"] == 80 * GIB for _, _, limits in seen)
    assert report["state"] == "activated"


def test_group_worker_variable_caps_concurrency(repo, tmp_path, monkeypatch):
    configure_main(repo, monkeypatch)
    monkeypatch.setattr(dev.os, "environ", dict(ENV, **{dev.GROUP_WORKERS_ENV: "2"}))
    monkeypatch.setattr(dev, "physical_memory", lambda: 128 * GIB)
    cache = tmp_path / "cache"
    for row in plan(repo, tmp_path)["support"]:
        cache_row(cache, row)
    seen = []
    monkeypatch.setattr(dev, "run_logged", fake_group_compiler(seen))
    evidence = tmp_path / "evidence"
    assert dev.main(["build", "--sets", "identical", "--cache", str(cache), "--evidence", str(evidence)]) == 0
    report = json.loads((evidence / "result.json").read_text())
    assert report["scheduler"]["slots"] == 2 and report["groups_concurrent_max"] <= 2
    assert all(dev.GROUP_WORKERS_ENV not in r["fields"]["environment"] for r in report["plan"]["groups"])
    assert report["state"] == "activated"


def test_caller_compile_jobs_keep_their_keys_under_the_scheduler(repo, tmp_path, monkeypatch):
    configure_main(repo, monkeypatch)
    env = dict(ENV, MOJOLEARN_COMPILE_JOBS="1")
    monkeypatch.setattr(dev.os, "environ", dict(env))
    monkeypatch.setattr(dev, "physical_memory", lambda: 128 * GIB)
    cache = tmp_path / "cache"
    before = plan(repo, tmp_path, env)
    for row in before["support"]:
        cache_row(cache, row)
    seen = []
    monkeypatch.setattr(dev, "run_logged", fake_group_compiler(seen))
    evidence = tmp_path / "evidence"
    assert dev.main(["build", "--sets", "identical", "--cache", str(cache), "--evidence", str(evidence)]) == 0
    report = json.loads((evidence / "result.json").read_text())
    assert {jobs for _, jobs, _ in seen} == {"1"}
    assert {r["stem"]: r["key"] for r in report["plan"]["groups"]} == keys(before)
    assert report["state"] == "activated"


def test_launch_threads_sum_to_the_cores_after_every_start():
    # 32-vCPU box, 15 groups fit memory: every core is given out.
    threads = dev.launch_threads(32, [], [None] * 15)
    assert sum(threads) == 32 and len(threads) == 15 and threads[0] >= threads[-1] >= 2
    # A group finished and freed its 3 threads: the next start takes all 3.
    running = threads[1:]
    assert dev.launch_threads(32, running, [None, None]) == [2, 1]
    assert sum(running) + 3 == 32
    # No free core: nothing starts beside the running groups.
    assert dev.launch_threads(32, [16, 16], [None]) == []
    # Fixed counts (AMD's 8) take their cores; chosen groups get what is left.
    assert dev.launch_threads(32, [8, 8], [8, None, None]) == [8, 4, 4]
    assert dev.launch_threads(32, [8, 8, 8], [8, None]) == [8]
    assert dev.launch_threads(32, [8, 8, 8, 8], [8, None]) == []
    # A lone group above the cores still starts, alone.
    assert dev.launch_threads(4, [], [8, None]) == [8]


def test_longest_group_starts_first_recorded_seconds_then_source_size(tmp_path):
    root = tmp_path / "src"
    for name, size in (("small.mojo", 10), ("large.mojo", 1000), ("mid.mojo", 100)):
        write(root, name, "x" * size)

    def row(stem, files, seconds=0.0):
        return {"mode": "identical", "stem": stem, "recorded_seconds": seconds,
                "fields": {"repo": str(root), "source": {"files": {f: "h" for f in files}}}}
    # First run, no records: the largest source closure starts first.
    first = [row("s", ["small.mojo"]), row("l", ["large.mojo"]), row("m", ["mid.mojo"])]
    ordered, needs = dev.packing_order(first, [12 * GIB] * 3)
    assert [r["stem"] for r in ordered] == ["l", "m", "s"] and needs == [12 * GIB] * 3
    # Recorded seconds outrank size and memory; an unrecorded group is
    # estimated from its bytes at the recorded seconds per byte (600 s / 110 bytes).
    later = [row("s", ["small.mojo"], 500.0), row("l", ["large.mojo"], 0.0), row("m", ["mid.mojo"], 100.0)]
    ordered, needs = dev.packing_order(later, [GIB, 3 * GIB, 2 * GIB])
    assert [r["stem"] for r in ordered] == ["l", "s", "m"] and needs == [3 * GIB, GIB, 2 * GIB]
    assert dev.estimated_seconds(later) == [500.0, pytest.approx(1000 * 600 / 110), 100.0]


def test_unrecorded_groups_pack_by_what_this_run_measured():
    floor, ceiling = 12 * GIB, 200 * GIB
    # Nothing measured yet: the floor stays.
    assert dev.live_need("group-floor", floor, 0, floor, ceiling) == (floor, "group-floor")
    # A passed group of the set measured 3 GiB: unrecorded groups pack at 3 GiB ...
    assert dev.live_need("group-floor", floor, 3 * GIB, floor, ceiling) == (3 * GIB, "set-max-live")
    assert dev.live_need("set-max-live", 3 * GIB, 4 * GIB, floor, ceiling) == (4 * GIB, "set-max-live")
    # ... but keep the floor as their rss limit.
    # 1.5x the need, never below 4 GiB, never below the floor for a live estimate.
    assert dev.guard_rss(3 * GIB, "set-max-live", floor) == max(floor, int(4.5 * GIB))
    assert dev.guard_rss(3 * GIB, "recorded", floor) == int(4.5 * GIB)
    assert dev.guard_rss(1 * GIB, "recorded", floor) == 4 * GIB
    # A set maximum from older records only rises; own records and the flag stay.
    assert dev.live_need("set-max", 5 * GIB, 3 * GIB, floor, ceiling) == (5 * GIB, "set-max")
    assert dev.live_need("set-max", 5 * GIB, 7 * GIB, floor, ceiling) == (7 * GIB, "set-max")
    assert dev.live_need("recorded", 2 * GIB, 9 * GIB, floor, ceiling) == (2 * GIB, "recorded")
    assert dev.live_need("memory-per-job-gib", 2 * GIB, 9 * GIB, floor, ceiling) == (2 * GIB, "memory-per-job-gib")
    assert dev.live_need("group-floor", floor, 300 * GIB, floor, ceiling) == (ceiling, "set-max-live")


def test_sliding_window_starts_a_group_as_soon_as_a_core_frees(repo, tmp_path, monkeypatch):
    configure_main(repo, monkeypatch)
    monkeypatch.setattr(dev, "physical_memory", lambda: 128 * GIB)
    monkeypatch.setattr(dev, "available_resources", lambda: (2, 100 * GIB))
    cache = tmp_path / "cache"
    first = plan(repo, tmp_path)
    for row in first["support"]:
        cache_row(cache, row)
    seen = []
    monkeypatch.setattr(dev, "run_logged", fake_group_compiler(seen))
    evidence = tmp_path / "evidence"
    assert dev.main(["build", "--sets", "identical", "--cache", str(cache), "--evidence", str(evidence)]) == 0
    report = json.loads((evidence / "result.json").read_text())
    groups = len(first["groups"])
    assert groups > 2 and len(seen) == groups
    # Two cores: two groups at a time, one thread each, never more threads than cores.
    assert report["groups_concurrent_max"] == 2 and report["threads_running_max"] == 2
    assert {jobs for _, jobs, _ in seen} == {"1"}
    assert all(r["threads_running_at_launch"] <= 2 and r["groups_running_at_launch"] <= 2
               for r in report["results"])
    assert report["state"] == "activated"


def test_input_check_rebuilds_only_the_finished_group(repo, tmp_path, monkeypatch):
    row = plan(repo, tmp_path)["groups"][0]
    monkeypatch.setattr(dev, "make_plan", lambda *_, **__: pytest.fail("no whole-set re-plan after a compile"))
    monkeypatch.setattr(dev, "cached_artifact", lambda *_: None)

    def build(command, *_):
        output = Path(command[command.index("--outdir") + 1])
        (output / (row["stem"] + ".so")).write_bytes(b"completed")
        return 0
    monkeypatch.setattr(dev, "run_logged", build)
    reads = []
    monkeypatch.setattr(dev, "toolchain_facts", lambda root: reads.append(root) or FACTS)
    probe = dev.ToolchainProbe(repo)
    result = dev.compile_group(row, repo, tmp_path / "cache", ENV, tmp_path / "evidence", probe,
                               {"groups_running_at_launch": 3, "threads_running_at_launch": 7,
                                "cores": 8, "submitted": 0.0})
    assert result["state"] == "built", result.get("reason")
    assert result["groups_running_at_launch"] == 3 and result["threads_running_at_launch"] == 7
    assert result["queue_wait_seconds"] >= 0 and "submitted" not in result
    assert len(reads) == 1
    receipt = json.loads((tmp_path / "evidence" / (row["mode"] + "-" + row["stem"] + ".json")).read_text())
    assert receipt["groups_running_at_launch"] == 3 and "queue_wait_seconds" in receipt
    # The one-group fields equal the plan's: same key.
    assert dev.digest(dev.current_group_fields(row, repo, tmp_path / "cache", ENV, FACTS)) == row["key"]


def test_one_toolchain_read_serves_groups_that_finished_before_it(monkeypatch, tmp_path):
    import threading
    gate, reads = threading.Event(), []

    def slow_facts(root):
        reads.append(root)
        gate.wait(5)
        return FACTS
    monkeypatch.setattr(dev, "toolchain_facts", slow_facts)
    monkeypatch.setattr(dev.build_process, "cancelled", lambda: False)
    probe = dev.ToolchainProbe(tmp_path)
    moment = dev.time.monotonic()
    out = []
    workers = [threading.Thread(target=lambda: out.append(probe.facts_after(moment))) for _ in range(4)]
    for worker in workers:
        worker.start()
    dev.time.sleep(0.2)
    gate.set()
    for worker in workers:
        worker.join(10)
    assert out == [FACTS] * 4 and len(reads) == 1
    # A group that finished after that read began needs a new read.
    assert probe.facts_after(dev.time.monotonic()) == FACTS and len(reads) == 2


# ---------------------------------------------------------------------------
# --layout release: the macOS wheel's compile through the one Mac cache
# ---------------------------------------------------------------------------

RELEASE_GROUPS = {("identical", "_mojolearn_native"), ("identical", "_mojolearn_native_m1"),
                  ("host", "_mojolearn_native_host")}
# A synthetic merged_build.sh that writes the declared output, as the real one does.
FAKE_MERGED_BUILD = """#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
: "${MOJOLEARN_COMPILE_JOBS:-2}${MOJOLEARN_PACKAGE_BYTE_LM:-0}${MOJOLEARN_NUMERIC_MODE:-}${MOJOLEARN_BUILD_DEFINES:-}"
mkdir -p "$(dirname "$MOJOLEARN_BINCACHE_OUTPUTS")"
printf 'compiled %s\\n' "$*" > "$MOJOLEARN_BINCACHE_OUTPUTS"
"""


@pytest.fixture
def mac(repo, monkeypatch):
    """The fixture repo seen as a Mac: fixed OS and Xcode facts, no GPU probe."""
    monkeypatch.setattr(bincache, "os_fields", lambda: {"id": "darwin", "version_id": "26", "machine": "arm64"})
    monkeypatch.setattr(bincache, "darwin_toolchain", lambda: {"xcode": "Xcode test"})
    monkeypatch.setattr(bincache, "device_arch", lambda: "")
    monkeypatch.setattr(bincache, "build_platform", lambda: "darwin")
    write(repo, "bindings/merged_build.sh", FAKE_MERGED_BUILD)
    return repo


def release_plan(repo, tmp_path, env=None, **kwargs):
    return dev.release_plan(repo, tmp_path / "cache", tmp_path / "mac", ["identical", "host"],
                            {} if env is None else env, facts=FACTS, plat="darwin", c_facts=C_FACTS, **kwargs)


def wheel_key(repo, tmp_path, mode, stem, **release_env):
    """The key packaging/macos/build_release_wheel.sh computes for one group under
    tools/release.py step_macos_build and tools/mac_slot.py run, spelled out here
    independently of bincache.macos_merged_group."""
    env = dict({"MOJOLEARN_BINCACHE_DIR": str(tmp_path / "release/macos-bincache"),
                "MOJOLEARN_PACKAGE_BYTE_LM": "1", "MOJOLEARN_COMPILE_JOBS": "1", "MOJOLEARN_BUILD_JOBS": "4",
                "MOJOLEARN_RELEASE_MODES": "identical", "MOJOLEARN_MERGED_BINDINGS": "1",
                "MOJOLEARN_MERGED_KEY": "f" * 64, "MOJOLEARN_SLOT_TOKEN": "slot-token",
                "MOJOLEARN_BINCACHE_SEED_DIRS": str(tmp_path / "mac")}, **release_env)
    if mode == "host":
        kind, output = "host", f"python/mojolearn/host/{stem}.so"
        env.pop("MOJOLEARN_GPU_ARCHS", None)
        env.pop("MOJOLEARN_HOST_OUTDIR", None)
        env.update(MOJOLEARN_NUMERIC_MODE="identical", MOJOLEARN_TARGET_COLUMN="cpu")
    else:
        kind, output = "gpu", f"python/mojolearn/{mode}/{stem}.so"
        env["MOJOLEARN_NUMERIC_MODE"] = mode
    env.update(MOJOLEARN_BINCACHE_OUTPUTS=output, MOJOLEARN_BINCACHE_SHELL="sh")
    return bincache.key_of(bincache.local_fields(repo, repo / "bindings/merged_build.sh",
                                                 [kind, "--group", stem], env, [output], "sh"))


def test_default_layout_is_the_release_compile_on_macos_only():
    assert dev.default_layout("darwin") == "release"
    assert dev.default_layout("linux") == "development"


def test_release_environment_takes_the_release_compile_settings_and_refuses_another_job_count():
    env = dev.release_environment("identical", {}, "darwin")
    assert (env["MOJOLEARN_COMPILE_JOBS"], env["MOJOLEARN_PACKAGE_BYTE_LM"]) == ("1", "1")
    assert dev.release_environment("identical", {"MOJOLEARN_COMPILE_JOBS": "1"}, "darwin")
    with pytest.raises(dev.BuildError, match="MOJOLEARN_COMPILE_JOBS=4 would key another compile"):
        dev.release_environment("identical", {"MOJOLEARN_COMPILE_JOBS": "4"}, "darwin")
    with pytest.raises(dev.BuildError, match="MOJOLEARN_PACKAGE_BYTE_LM=0 would key another compile"):
        dev.release_environment("host", {"MOJOLEARN_PACKAGE_BYTE_LM": "0"}, "darwin")
    host = dev.release_environment("host", {"MOJOLEARN_GPU_ARCHS": "x"}, "darwin")
    assert "MOJOLEARN_GPU_ARCHS" not in host and host["MOJOLEARN_TARGET_COLUMN"] == "cpu"


def test_release_layout_is_refused_off_macos(mac, tmp_path):
    with pytest.raises(dev.BuildError, match="macOS wheel's compile"):
        dev.release_plan(mac, tmp_path / "cache", tmp_path / "mac", ["identical"], {}, facts=FACTS,
                         plat="linux", c_facts=C_FACTS)


def test_release_plan_keys_are_the_macos_wheel_keys(mac, tmp_path):
    # The development shell carries variables the release never sets and no
    # keyed input reads; the release carries its own runner switches.
    value = release_plan(mac, tmp_path, env={"MOJOLEARN_RUNPOD_KEY_FILE": "/k", dev.GROUP_WORKERS_ENV: "3",
                                             "MOJOLEARN_MERGED_PYTHON": "/some/python"})
    assert value["schema"] == dev.RELEASE_SCHEMA and value["layout"] == "release"
    assert {(r["mode"], r["stem"]) for r in value["groups"]} == RELEASE_GROUPS
    native = next(r for r in value["groups"] if r["stem"] == "_mojolearn_native")
    assert native["members"] == [A, BYTE]
    for row in value["groups"]:
        assert row["key"] == wheel_key(mac, tmp_path, row["mode"], row["stem"])
        assert row["action"] == "build" and row["compile_jobs"] == 1 and not row["auto_compile_jobs"]
        assert row["command"][1:] == ["tools/bincache.py", "build", "./bindings/merged_build.sh",
                                      "host" if row["mode"] == "host" else "gpu", "--group", row["stem"]]
    # The keys still separate a different compile: another job count, another checkout.
    assert native["key"] != wheel_key(mac, tmp_path, "identical", "_mojolearn_native", MOJOLEARN_COMPILE_JOBS="2")
    assert native["key"] != wheel_key(mac, tmp_path, "identical", "_mojolearn_native",
                                      MOJOLEARN_BUILD_DEFINES="-D X=1")


def release_outputs(repo, value, data=b"release-bytes"):
    """Write each group's package file and archive it in the one Mac cache under its key."""
    for row in value["groups"]:
        write(repo, row["output"], data.decode() + row["stem"])
        mac_cache = Path(row["mac_cache"])
        mac_cache.mkdir(parents=True, exist_ok=True)
        bincache.pack(mac_cache / (row["key"] + ".tar.gz"), row["key"], row["fields"]["bincache"], repo,
                      [row["output"]], {})


def test_cached_release_groups_plan_as_reuse_and_activation_indexes_each_set(mac, tmp_path):
    value = release_plan(mac, tmp_path)
    release_outputs(mac, value)
    again = release_plan(mac, tmp_path)
    assert {r["action"] for r in again["groups"]} == {"reuse"}
    assert again["counts"]["reuse"] == 3
    for row in again["support"]:
        cache_row(tmp_path / "cache", row)
    stale = write(mac, "python/mojolearn/identical/_mojolearn_native.so", "other bytes")
    assert dev.activate_release(again, mac, tmp_path / "cache")["missing"] == ["identical:_mojolearn_native"]
    release_outputs(mac, value)
    assert stale.read_text() == "release-bytes_mojolearn_native"
    result = dev.activate_release(again, mac, tmp_path / "cache")
    assert result["state"] == "activated" and result["activated"] == ["host", "identical"]
    index = json.loads((mac / "python/mojolearn/identical/_mojolearn_merged.json").read_text())
    assert index == {"format": merged.INDEX_FORMAT,
                     "files": {"_mojolearn_native.so": [A, BYTE], "_mojolearn_native_m1.so": [B]}}
    host = json.loads((mac / "python/mojolearn/host/_mojolearn_merged.json").read_text())
    assert host["files"] == {"_mojolearn_native_host.so": [HOST]}
    write(mac, f"python/mojolearn/identical/{A}.so", "a shadowing per-binding file")
    with pytest.raises(dev.BuildError, match="shadow"):
        dev.activate_release(again, mac, tmp_path / "cache")


def fake_bincache_child(record):
    def run(argv, root, env, log):
        record.append((list(argv), {k: v for k, v in env.items() if k.startswith("MOJOLEARN_")}))
        if argv[:2] == ["sh", "tokenizer/tools/gen_unicode_table.sh"]:
            return 0
        assert argv[1:3] == ["tools/bincache.py", "build"]
        child = {k: v for k, v in env.items() if k != dev.GUARD_ENV}
        return bincache.cmd_build_local(str(Path(root) / argv[3]), argv[4:], child, child["MOJOLEARN_BINCACHE_DIR"])
    return run


def test_compile_release_group_requires_the_plan_key_and_counts_only_compiles(mac, tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(dev, "run_logged", fake_bincache_child(calls))
    value = release_plan(mac, tmp_path)
    row = next(r for r in value["groups"] if r["mode"] == "host")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    built = dev.compile_release_group(row, mac, tmp_path / "cache", {}, evidence)
    assert built["state"] == "built" and built["cached"] and built["command"] == row["command"]
    assert built["bincache_key"] == row["key"]
    env = calls[-1][1]
    assert env["MOJOLEARN_BINCACHE_DIR"] == str((tmp_path / "mac").resolve())
    assert (env["MOJOLEARN_COMPILE_JOBS"], env["MOJOLEARN_PACKAGE_BYTE_LM"]) == ("1", "1")
    assert (env["MOJOLEARN_TARGET_COLUMN"], env["MOJOLEARN_BINCACHE_OUTPUTS"]) == ("cpu", row["output"])
    assert "MOJOLEARN_GPU_ARCHS" not in env
    placed = dev.compile_release_group(row, mac, tmp_path / "cache", {}, evidence)
    assert placed["state"] == "reused" and "command" not in placed, "a placement is not a compile attempt"
    assert not dev.peak_path(tmp_path / "cache", "host", row["stem"]).exists() or \
        json.loads(dev.peak_path(tmp_path / "cache", "host", row["stem"]).read_text())["key"] == row["key"]
    moved = dict(row, key="0" * 64)
    failed = dev.compile_release_group(moved, mac, tmp_path / "cache", {}, evidence)
    assert failed["exit_code"] == 1 and "keyed this group" in failed["reason"]


def test_limited_release_group_retains_peak_without_cache_provenance(mac, tmp_path, monkeypatch):
    value = release_plan(mac, tmp_path)
    row = next(r for r in value["groups"] if r["mode"] == "host")
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    cache = tmp_path / "cache"
    observed = 5 * GIB

    def limited_child(command, cwd, env, log):
        Path(log).with_suffix(".log.execution.json").write_text(json.dumps({
            "status": "rss-limit", "exit_code": 124,
            "peak_rss_bytes": observed, "reason": "measured RSS exceeded reservation",
        }))
        return 124

    monkeypatch.setattr(dev, "run_logged", limited_child)
    failed = dev.compile_release_group(row, mac, cache, {}, evidence)
    assert failed["state"] == "failed" and failed["exit_code"] == 124
    assert failed["bincache_outcome"] == ""
    assert failed["peak_rss_bytes"] == observed
    assert dev.recorded_need(cache, row["mode"], row["stem"]) == int(observed * dev.PEAK_HEADROOM["limited"])


def test_main_release_layout_compiles_once_into_the_one_mac_cache_and_the_release_keys_hit(
        mac, tmp_path, monkeypatch):
    monkeypatch.setattr(dev.os, "environ", {})
    monkeypatch.setattr(merged, "default_platform", lambda: "darwin")
    monkeypatch.setattr(bincache, "first_line", lambda *_: "synthetic-source-commit")
    monkeypatch.setattr(dev, "physical_memory", lambda: 100 * 2 ** 30)
    calls = []
    monkeypatch.setattr(dev, "run_logged", fake_bincache_child(calls))
    for row in release_plan(mac, tmp_path)["support"]:
        cache_row(tmp_path / "cache", row)
    argv = ["build", "--sets", "identical,host", "--cache", str(tmp_path / "cache"),
            "--mac-cache", str(tmp_path / "mac")]
    assert dev.main(argv + ["--evidence", str(tmp_path / "first")]) == 0
    first = json.loads((tmp_path / "first/result.json").read_text())
    assert first["schema"] == dev.RELEASE_SCHEMA and first["state"] == "activated"
    assert first["compile_attempts"] == 3
    assert {(r["mode"], r["stem"], r["state"]) for r in first["results"]} == \
        {(m, s, "built") for m, s in RELEASE_GROUPS}
    # The release's macOS build of the same commit in the same checkout: every key is in the one cache.
    for mode, stem in RELEASE_GROUPS:
        assert (tmp_path / "mac" / (wheel_key(mac, tmp_path, mode, stem) + ".tar.gz")).is_file()
    assert dev.main(argv + ["--evidence", str(tmp_path / "second")]) == 0
    second = json.loads((tmp_path / "second/result.json").read_text())
    assert second["compile_attempts"] == 0
    assert {r["state"] for r in second["results"]} == {"reused"}
    # --layout development keeps the per-binding groups and their own cache.
    assert dev.main(["plan", "--layout", "development", "--sets", "identical", "--cache",
                     str(tmp_path / "cache")]) == 0
