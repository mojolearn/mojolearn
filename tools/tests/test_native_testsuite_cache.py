"""Tool tests for tools/ops/native_testsuite_cache.py: what changes a native-check program's key, and the cache store.

No compiler runs: the tree is a temp directory with two TestSuite files, a fake `.pixi/envs/default/bin/mojo` and the
generated files `native_testsuite.sh build` leaves in its gen dir. The end-to-end reuse (a program built in one
checkout, copied into another and run there) was shown with Mojo 1.2.0 on 2026-11-26; see the module docstring.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("native_testsuite_cache", ROOT / "tools/ops/native_testsuite_cache.py")
cache = importlib.util.module_from_spec(_spec)
sys.modules["native_testsuite_cache"] = cache
_spec.loader.exec_module(cache)

FLAGS = ["-D", "MOJOLEARN_NUMERIC_IDENTICAL=1", "-D", "MOJOLEARN_COLUMN_CPU=1", "--target-cpu", "x86-64-v3"]
ENV = {"PATH": os.environ.get("PATH", "")}


def _tree(root: Path) -> Path:
    """Two checks (ta, tb) importing one shared module and one module of their own; plan: one group each."""
    files = {
        "checks/__init__.mojo": "",
        "checks/native_tests/__init__.mojo": "",
        "core/__init__.mojo": "",
        "core/shared.mojo": "def shared() -> Int:\n    return 1\n",
        "core/only_a.mojo": "def a() -> Int:\n    return 2\n",
        "core/only_b.mojo": "def b() -> Int:\n    return 3\n",
        "checks/native_tests/ta.mojo": "# SPDX-License-Identifier: Apache-2.0\n# native-check: cpu\n"
                                       "from core.shared import shared\nfrom core.only_a import a\n",
        "checks/native_tests/tb.mojo": "# SPDX-License-Identifier: Apache-2.0\n# native-check: cpu\n"
                                       "from core.shared import shared\nfrom core.only_b import b\n",
        "build/tsgen/plan.txt": "group 0 ta\ngroup 1 tb\n",
        "build/tsgen/ts_group_0.mojo": "# driver 0\n",
        "build/tsgen/ts_group_1.mojo": "# driver 1\n",
        "build/tsgen/ts_ta.mojo": "# copy of ta\n",
        "build/tsgen/ts_tb.mojo": "# copy of tb\n",
    }
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    mojo = root / ".pixi/envs/default/bin/mojo"
    mojo.parent.mkdir(parents=True)
    mojo.write_text("#!/bin/sh\necho 'Mojo 1.2.0 (test)'\n")
    mojo.chmod(0o755)
    return root / "build/tsgen"


def _keys(root: Path, flags=FLAGS, env=ENV) -> dict[str, str]:
    return {f"{kind} {ident}": key for kind, ident, key in
            cache.unit_keys(root.resolve(), "cpu", root.resolve() / "build/tsgen", list(flags), env)}


def test_one_key_per_planned_unit_and_the_same_keys_in_another_checkout(tmp_path):
    _tree(tmp_path / "a")
    _tree(tmp_path / "b")
    a, b = _keys(tmp_path / "a"), _keys(tmp_path / "b")
    assert sorted(a) == ["group 0", "group 1"] and len(set(a.values())) == 2
    assert a == b  # no path in the key: a second checkout of the same source reuses the programs


def test_a_source_edit_changes_only_the_keys_of_the_programs_that_import_it(tmp_path):
    _tree(tmp_path)
    before = _keys(tmp_path)
    (tmp_path / "core/only_a.mojo").write_text("def a() -> Int:\n    return 20\n")
    after = _keys(tmp_path)
    assert after["group 0"] != before["group 0"] and after["group 1"] == before["group 1"]
    (tmp_path / "core/shared.mojo").write_text("def shared() -> Int:\n    return 10\n")
    shared = _keys(tmp_path)
    assert shared["group 0"] != after["group 0"] and shared["group 1"] != after["group 1"]


def test_the_generated_driver_and_member_copy_are_inputs(tmp_path):
    gen = _tree(tmp_path)
    before = _keys(tmp_path)
    (gen / "ts_group_1.mojo").write_text("# driver 1, regenerated\n")
    assert _keys(tmp_path)["group 1"] != before["group 1"]
    (gen / "ts_ta.mojo").write_text("# copy of ta, regenerated\n")
    assert _keys(tmp_path)["group 0"] != before["group 0"]


def test_flags_toolchain_and_build_environment_change_every_key(tmp_path):
    _tree(tmp_path)
    base = _keys(tmp_path)
    changed = {
        "define": _keys(tmp_path, flags=FLAGS + ["-D", "MOJOLEARN_OTHER=1"]),
        "environment": _keys(tmp_path, env=dict(ENV, MOJOLEARN_SOMETHING="1")),
    }
    mojo = tmp_path / ".pixi/envs/default/bin/mojo"
    mojo.write_text("#!/bin/sh\necho 'Mojo 1.2.1 (test)'\n")
    changed["compiler"] = _keys(tmp_path)
    for why, keys in changed.items():
        assert all(keys[u] != base[u] for u in base), why


def test_a_fault_build_and_a_checkout_without_a_compiler_get_no_keys(tmp_path):
    _tree(tmp_path)
    assert _keys(tmp_path, flags=FLAGS + ["-D", "MOJOLEARN_SABOTAGE_X=1"]) == {}
    assert _keys(tmp_path, env=dict(ENV, MOJOLEARN_FAULT_INJECT="1")) == {}
    (tmp_path / ".pixi/envs/default/bin/mojo").unlink()
    assert _keys(tmp_path) == {}


def test_store_and_reuse_and_a_damaged_entry_is_a_miss(tmp_path):
    root, key = tmp_path / "c", "ab" + "0" * 62
    src = tmp_path / "prog"
    src.write_bytes(b"\x7fELF program")
    assert cache.get(root, key, tmp_path / "out") is False  # empty cache
    assert cache.put(root, key, src) is True
    dst = tmp_path / "bin/ts/prog"
    assert cache.get(root, key, dst) is True
    assert dst.read_bytes() == src.read_bytes() and os.access(dst, os.X_OK)
    (root / key[:2] / key / "bin").write_bytes(b"\x7fELF")  # truncated
    assert cache.get(root, key, tmp_path / "again") is False
    assert cache.get(None, key, dst) is False  # NC_TS_CACHE unset


def test_old_entries_are_pruned_and_a_bad_key_is_refused(tmp_path):
    root, src = tmp_path / "c", tmp_path / "prog"
    src.write_bytes(b"x")
    old, new = "aa" + "1" * 62, "bb" + "2" * 62
    cache.put(root, old, src)
    meta = root / old[:2] / old / "meta.json"
    os.utime(meta, (1, 1))
    cache.put(root, new, src, days=14)
    assert not meta.exists() and (root / new[:2] / new / "meta.json").is_file()
    try:
        cache.entry(root, "../../etc")
        raise AssertionError("a key that is not a sha256 was accepted")
    except SystemExit:
        pass


def test_keys_command_prints_one_line_per_unit(tmp_path, capsys):
    gen = _tree(tmp_path)
    old = os.environ.copy()
    try:
        for k in [k for k in os.environ if k.startswith(("MOJOLEARN_", "MOJO_", "MODULAR_"))]:
            del os.environ[k]
        assert cache.main(["keys", "--column", "cpu", "--gen", str(gen), "--flags", " ".join(FLAGS),
                           "--repo", str(tmp_path)]) == 0
    finally:
        os.environ.clear()
        os.environ.update(old)
    rows = [line.split() for line in capsys.readouterr().out.splitlines()]
    assert [r[:2] for r in rows] == [["group", "0"], ["group", "1"]] and all(len(r[2]) == 64 for r in rows)


def test_the_build_script_reuses_before_compiling_and_stores_after(tmp_path):
    text = (ROOT / "tools/ops/native_testsuite.sh").read_text()
    unit = text[text.index("compile_unit() {"):text.index("export -f compile_unit")]
    get, build, put = (unit.index("native_testsuite_cache.py get"), unit.index("mojo build"),
                       unit.index("native_testsuite_cache.py put"))
    assert get < build < put and "[ $rc -eq 0 ] && [ -n \"$key\" ]" in unit  # only a successful compile is kept
    assert 'NC_TS_CACHE:-' in text and "native_testsuite_cache.py keys" in text
    run = text[text.index("\nrun)"):]
    assert "LD_LIBRARY_PATH=\"$PWD/.pixi/envs/default/lib" in run
    for job in ("tools/ops/cpu_box_checks_job.sh", "tools/ops/native_checks_gpu_job.sh"):
        assert "NC_TS_CACHE" in (ROOT / job).read_text(), job


def test_the_repository_checks_resolve_every_import():
    # An unresolved import widens a program's key to the whole tree (no reuse for it). On 2026-11-26 none of the
    # 797 check files had one; this keeps the first 40 (by name) honest without parsing the whole tree.
    import native_checks_report as ncr  # on sys.path through the module under test
    picked = sorted(ncr.testsuite_paths(ROOT).items())[:40]
    sources = cache.Sources(ROOT, [ROOT / rel for _, (rel, _) in picked])
    unresolved = {name: sources.closure(ROOT / rel)[1] for name, (rel, _) in picked}
    assert not any(unresolved.values()), {n: u for n, u in unresolved.items() if u}
    files, _ = sources.closure(ROOT / picked[0][1][0])
    assert all(len(h) == 64 for h in sources.hashes(files).values())


def test_the_github_build_restores_the_cache_before_building_and_saves_it_after():
    import yaml
    steps = yaml.safe_load((ROOT / ".github/workflows/native-checks.yml").read_text())["jobs"]["build"]["steps"]
    uses = [s.get("uses", "") for s in steps]
    restore = next(i for i, u in enumerate(uses) if u.startswith("actions/cache/restore@"))
    save = next(i for i, u in enumerate(uses) if u.startswith("actions/cache/save@"))
    build = next(i for i, s in enumerate(steps) if s.get("run", "").endswith("native_checks_steps.sh build"))
    assert restore < build < save
    path = steps[restore]["with"]["path"]
    assert steps[build]["env"]["NC_TS_CACHE"] == path == steps[save]["with"]["path"]
    assert steps[restore]["with"]["key"] == steps[save]["with"]["key"] and steps[restore]["with"]["restore-keys"]
