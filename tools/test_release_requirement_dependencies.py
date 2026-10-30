#!/usr/bin/env python3
"""Dependency scoping and frozen-source integrity, without builds or GPUs."""
import copy
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import release_requirement_dependencies as d


def binding(name, source, vendor="cuda", arch="sm_89", target="linux-64"):
    key = "/".join((target, vendor, arch, "identical", name))
    return dict(key=key, target=target, vendor=vendor, arch=arch, tier="identical", name=name,
        identity=dict(name=name, target=target, vendor=vendor, arch=arch,
            closure=dict(scope="closure", digest=source), toolchain={"packages": ["mojo-pin"]},
            flags={"identical": True}, builders={"build.sh": "binding-filtered-view"}, image="pinned"),
        sources=["core/" + name + ".mojo"], unresolved=[])


def snapshot():
    return dict(revision="b" * 40, bindings=[binding("one", "one-bytes"), binding("two", "two-bytes")],
        runtime=dict(identity={"runtime": "pinned"}, unresolved=None),
        toolchains={"linux-64": {"packages": ["mojo-pin"]}},
        adapters=dict(sources={"adapters/torch/registration.cpp": "cpp-bytes"}, unresolved=[]),
        lanes={"one": dict(sources={"core/shared.mojo": "shared", "python/mojolearn/one.py": "wrapper-one"},
                           bindings=["one"], precision_limitations=[]),
               "two": dict(sources={"core/shared.mojo": "shared", "python/mojolearn/two.py": "wrapper-two"},
                           bindings=["two"], precision_limitations=[])},
        extra_sources={"recipes/one.json": "recipe"})


def spec(kind):
    return dict(id=kind + "-nvidia", kind=kind, scope="same contract", stage="pre",
        coverage=["full coverage"], source_inputs={"broad": ["*.mojo", "python/**/*.py"]},
        configuration=["build_flags", "release_modes"])


def manifest(expanded, revision):
    rows = {}
    for row in expanded["requirements"]:
        inputs = dict(row.get("resolved_inputs", {}))
        inputs["contract"] = {k: row[k] for k in ("id", "scope", "coverage", "kind", "stage")}
        rows[row["id"]] = dict(inputs=inputs, fingerprint=d.digest(inputs), source_revision=revision,
            kind=row["kind"], coverage=row["coverage"], missing_inputs=row.get("unresolved_inputs", []))
    return dict(source_revision=revision, requirements=rows)


class Dependencies(unittest.TestCase):
    def setUp(self):
        self.config = dict(build_flags={}, release_modes=["identical"],
                           dependency_context=dict(compiler_sha256={"linux-64": "c" * 64}))

    def expand(self, kinds=("compile", "identity"), snap=None, config=None):
        with patch.object(d, "frozen_snapshot", return_value=snap or snapshot()):
            return d.expand_catalog(".", "HEAD", dict(source_inputs={"all": ["**"]},
                requirements=[spec(k) for k in kinds]), config or self.config)

    def test_binding_and_adapter_units_ignore_global_source_globs(self):
        rows = self.expand()["requirements"]
        self.assertEqual(len(rows), 6)  # two native bindings, runtime, adapters, two lanes
        self.assertTrue(all(not r["inherit_source_inputs"] and not r["source_inputs"] for r in rows))
        self.assertTrue(all(not r["configuration"] for r in rows))
        cpp = next(r for r in rows if "/adapters/" in r["id"])
        self.assertIn("adapters/torch/registration.cpp", cpp["resolved_inputs"]["adapter_sources"])

    def test_selected_flags_change_only_affected_compile_and_identity_units(self):
        key = snapshot()["bindings"][0]["key"]
        config = copy.deepcopy(self.config)
        config["dependency_context"]["selected_binding_build_flags"] = {
            key: {"MOJOLEARN_MOJO_BUILD_FLAGS": "--optimization-level 0"}}
        before = manifest(self.expand(config=config), "a")
        config["dependency_context"]["selected_binding_build_flags"][key] = {
            "MOJOLEARN_MOJO_BUILD_FLAGS": "--optimization-level 3"}
        after = manifest(self.expand(config=config), "b")
        changed = {rid for rid, row in before["requirements"].items()
                   if row["fingerprint"] != after["requirements"][rid]["fingerprint"]}
        self.assertEqual(changed, {"compile-nvidia/" + key, "identity-nvidia/one"})

    def test_selected_flags_are_complete_not_merged_with_global_flags(self):
        key = snapshot()["bindings"][0]["key"]
        config = copy.deepcopy(self.config)
        config["build_flags"] = {"MOJOLEARN_MOJO_BUILD_FLAGS": "-O3"}
        config["dependency_context"]["selected_binding_build_flags"] = {key: {}}
        rows = {r["id"]: r["resolved_inputs"] for r in self.expand(config=config)["requirements"]}
        self.assertEqual(rows["compile-nvidia/" + key]["unit"]["build_flags"], {})
        self.assertEqual(rows["compile-nvidia/" + snapshot()["bindings"][1]["key"]]["unit"]["build_flags"],
                         config["build_flags"])
        self.assertEqual(rows["identity-nvidia/one"]["execution_environment"]["one"]["bindings"][key]
                         ["selected_build_flags"], {})

    def test_selected_flags_refuse_unknown_bindings_and_nonsemantic_values(self):
        key = snapshot()["bindings"][0]["key"]
        invalid = [None, {"unknown": {}}, {key: {"MOJOLEARN_COMPILE_JOBS": "10"}},
                   {key: {"MOJOLEARN_NUMERIC_MODE": "fast"}},
                   {key: {"MOJOLEARN_MOJO_BUILD_FLAGS": 0}}, {key: {"MOJOLEARN_API_TOKEN": "secret"}}]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                config = copy.deepcopy(self.config)
                config["dependency_context"]["selected_binding_build_flags"] = value
                self.expand(config=config)

    def test_selected_compiler_workers_are_pinned_without_mutating_source_snapshot(self):
        snap = snapshot()
        key = snap["bindings"][0]["key"]
        snap["bindings"][0]["identity"]["flags"]["compile_jobs"] = "1"
        config = copy.deepcopy(self.config)
        config["dependency_context"]["selected_binding_compile_jobs"] = {key: "10"}
        expanded = self.expand(snap=snap, config=config)
        before = manifest(expanded, "a")
        self.assertEqual(snap["bindings"][0]["identity"]["flags"]["compile_jobs"], "1")
        row = next(r for r in expanded["requirements"] if r["id"] == "compile-nvidia/" + key)
        self.assertEqual(row["resolved_inputs"]["binding"]["flags"]["compile_jobs"], "10")
        config["dependency_context"]["selected_binding_compile_jobs"][key] = "32"
        after = manifest(self.expand(snap=snap, config=config), "b")
        changed = {rid for rid, row in before["requirements"].items()
                   if row["fingerprint"] != after["requirements"][rid]["fingerprint"]}
        self.assertEqual(changed, {"compile-nvidia/" + key, "identity-nvidia/one"})

    def test_selected_compiler_workers_refuse_unknown_or_invalid_allocations(self):
        key = snapshot()["bindings"][0]["key"]
        for mapping in ({"unknown": "10"}, {key: "0"}, {key: "-1"}, {key: "1.5"},
                        {key: "01"}, {key: 10}, {key: True}, None):
            with self.subTest(mapping=mapping), self.assertRaises(ValueError):
                config = copy.deepcopy(self.config)
                config["dependency_context"]["selected_binding_compile_jobs"] = mapping
                self.expand(config=config)

    def test_cpu_compile_uses_explicit_execution_target(self):
        snap = snapshot()
        host = binding("host_one", "bytes", vendor="host", arch="", target="osx-arm64")
        host["tier"] = "host"
        linux = copy.deepcopy(host)
        linux.update(target="linux-64", key="linux-64/host/host_one")
        linux["identity"]["target"] = "linux-64"
        snap["bindings"] = [host, linux]
        row = dict(spec("compile"), id="compile-cpu")
        config = dict(self.config, dependency_context=dict(cpu_target="osx-arm64"))
        with patch.object(d, "frozen_snapshot", return_value=snap):
            expanded = d.expand_catalog(".", "HEAD", dict(requirements=[row]), config)
        self.assertEqual(len(expanded["requirements"]), 1)
        result = expanded["requirements"][0]
        self.assertEqual(result["resolved_inputs"]["binding"]["target"], "osx-arm64")
        self.assertEqual(result["parent_requirement"], "compile-cpu")
        self.assertNotIn("runtime", result["id"])

    def test_cpu_compile_without_target_blocks_instead_of_guessing_linux(self):
        for target in (None, "unknown"):
            row = dict(spec("compile"), id="compile-cpu")
            config = dict(self.config, dependency_context=dict(cpu_target=target))
            with patch.object(d, "frozen_snapshot", return_value=snapshot()):
                expanded = d.expand_catalog(".", "HEAD", dict(requirements=[row]), config)
            self.assertEqual(len(expanded["requirements"]), 1)
            self.assertEqual(expanded["requirements"][0]["unresolved_inputs"],
                             ["dependency_context.cpu_target is required"])

    def test_apple_adapter_uses_actual_apple_arch_and_builder_view(self):
        import release_reuse as rr
        arches = {row.arch for row in rr.bindings(rr.MACOS, ["identical"]) if not row.host}
        self.assertEqual(arches, {"apple"})
        snap = snapshot()
        snap["bindings"] = [binding("one", "bytes", vendor="metal", arch="apple", target="osx-arm64")]
        snap["adapters"]["builder_views"] = {"osx-arm64": {"tools/build_macos.sh": "filtered-adapter-builder"}}
        with patch.object(d, "frozen_snapshot", return_value=snap):
            expanded = d.expand_catalog(".", "HEAD", dict(requirements=[dict(spec("compile"), id="compile-apple")]), self.config)
        adapter = next(row for row in expanded["requirements"] if "/adapters/" in row["id"])
        self.assertEqual(adapter["id"], "compile-apple/adapters/apple")
        self.assertFalse(adapter["unresolved_inputs"])
        self.assertEqual(adapter["resolved_inputs"]["environment"]["arch"], "apple")
        self.assertEqual(adapter["resolved_inputs"]["environment"]["builders"],
                         {"tools/build_macos.sh": "filtered-adapter-builder"})

    def test_partial_lane_binding_context_does_not_hide_unmapped_helper(self):
        snap = snapshot()
        snap["lanes"]["one"]["bindings"].append("unmapped_helper")
        expanded = self.expand(("identity",), snap)
        one = next(row for row in expanded["requirements"] if row["id"] == "identity-nvidia/one")
        self.assertIn("no native binding context mapped for binding: unmapped_helper", one["unresolved_inputs"])
        environment = one["resolved_inputs"]["execution_environment"]["one"]
        self.assertTrue(environment["bindings"])
        two = next(row for row in expanded["requirements"] if row["id"] == "identity-nvidia/two")
        self.assertFalse(two["unresolved_inputs"])

    def test_isolated_wrapper_changes_only_its_lane_not_compilation(self):
        before = manifest(self.expand(), "a" * 40)
        changed = snapshot()
        changed["lanes"]["one"]["sources"]["python/mojolearn/one.py"] = "edited"
        after = manifest(self.expand(snap=changed), "b" * 40)
        moved = [k for k, v in before["requirements"].items()
                 if v["fingerprint"] != after["requirements"][k]["fingerprint"]]
        self.assertEqual(moved, ["identity-nvidia/one"])

    def test_one_binding_edit_preserves_other_compile_identity(self):
        before = manifest(self.expand(("compile",)), "a" * 40)
        changed = snapshot()
        changed["bindings"][0]["identity"]["closure"]["digest"] = "changed"
        after = manifest(self.expand(("compile",), changed), "b" * 40)
        moved = [k for k, v in before["requirements"].items()
                 if v["fingerprint"] != after["requirements"][k]["fingerprint"]]
        self.assertEqual(len(moved), 1)
        self.assertTrue(moved[0].endswith("/one"))

    def test_revision_is_only_provenance(self):
        before = self.expand()
        changed = snapshot()
        changed["revision"] = "c" * 40
        after = self.expand(snap=changed)
        self.assertEqual(before["requirements"], after["requirements"])
        self.assertNotEqual(before["dependency_provenance"], after["dependency_provenance"])

    def test_new_edge_and_new_unit_are_not_lost(self):
        changed = snapshot()
        changed["lanes"]["one"]["sources"]["core/new.mojo"] = "new"
        changed["bindings"].append(binding("new", "new"))
        result = self.expand(snap=changed)
        one = next(r for r in result["requirements"] if r["id"] == "identity-nvidia/one")
        self.assertIn("core/new.mojo", one["resolved_inputs"]["native_sources"])
        self.assertTrue(any(r["id"].endswith("/new") for r in result["requirements"]))

    def test_unknown_closure_blocks_unit(self):
        changed = snapshot()
        changed["bindings"][0]["unresolved"] = ["unresolved import closure"]
        row = next(r for r in self.expand(("compile",), changed)["requirements"] if r["id"].endswith("/one"))
        self.assertEqual(row["unresolved_inputs"], ["unresolved import closure"])

    def test_conservative_lane_precision_is_explicit_without_blanket_rerun(self):
        changed = snapshot()
        changed["lanes"]["one"]["precision_limitations"] = [{"route": "unknown"}]
        row = self.expand(("identity",), changed)["requirements"][0]
        self.assertFalse(row["unresolved_inputs"])
        self.assertEqual(row["dependency_limitations"]["one"], [{"route": "unknown"}])

    def test_no_recipe_mapping_is_explicit_blocker(self):
        row = self.expand(("performance",))["requirements"][0]
        self.assertEqual(row["unresolved_inputs"], ["full-workload recipes have not been mapped"])
        self.assertIn(d.RECIPE_INDEX, row["provenance_sources"])

    def test_recipe_and_actual_workload_metadata_are_hashed(self):
        config = copy.deepcopy(self.config)
        recipe = dict(id="one-full", lanes=["one"], source_inputs=["recipes/one.json"],
            dataset=dict(id="data", version="1", sha256="d" * 64, rows=1000), settings={},
            boundary="preparation through consumed outputs", phases=["fit", "inference"],
            environment={"gpu": "same hardware"})
        config["dependency_context"] = dict(workloads={"nvidia": [recipe]})
        row = self.expand(("performance",), config=config)["requirements"][0]
        self.assertFalse(row["unresolved_inputs"])
        self.assertEqual(row["resolved_inputs"]["recipe"], recipe)
        self.assertEqual(row["coverage"], ["full coverage", "lane:one", "phase:fit", "phase:inference"])
        self.assertEqual(row["resolved_inputs"]["recipe_sources"], {"recipes/one.json": "recipe"})

    def test_duplicate_recipe_ids_refused(self):
        config = dict(self.config, dependency_context=dict(workloads={"nvidia": [dict(id="x", lanes=[])] * 2}))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.expand(("performance",), config=config)

    def test_fast_is_not_an_expansion_campaign(self):
        with self.assertRaisesRegex(ValueError, "identical"):
            self.expand(config=dict(self.config, release_modes=["fast", "identical"]))

    def _pair(self):
        old = manifest(self.expand(("identity",)), "a" * 40)
        changed = snapshot()
        for lane in changed["lanes"].values():
            lane["sources"]["core/shared.mojo"] = "edited one definition"
        new = manifest(self.expand(("identity",), changed), "b" * 40)
        selection = dict(lanes=["one"], unattributed=[], precision_limitations=[],
            base_revision=old["source_revision"], source_revision=new["source_revision"],
            changed_paths=["core/shared.mojo"], selector_sources={"tools/lane_select.py": "selector"})
        return old, new, dict(selections={"nvidia": selection})

    def test_named_edit_reuses_other_lane_with_original_fingerprints(self):
        old, new, selected = self._pair()
        with patch.object(d, "frozen_snapshot", return_value=selected), patch.object(d, "_revision", side_effect=lambda root, rev: rev):
            plan = d.equivalence_proofs(".", old, new)
            self.assertEqual(plan["units"]["identity-nvidia/one"]["decision"], "affected")
            row = plan["units"]["identity-nvidia/two"]
            self.assertEqual(row["decision"], "retained")
            proof = row["reuse_proof"]
            self.assertNotEqual(proof["old_fingerprint"], proof["new_fingerprint"])
            self.assertFalse(d.validate_equivalence_proof(".", proof,
                old["requirements"]["identity-nvidia/two"], new["requirements"]["identity-nvidia/two"]))

    def test_caller_authored_selector_or_fingerprint_is_rejected(self):
        old, new, selected = self._pair()
        with patch.object(d, "frozen_snapshot", return_value=selected), patch.object(d, "_revision", side_effect=lambda root, rev: rev):
            proof = d.equivalence_proofs(".", old, new)["units"]["identity-nvidia/two"]["reuse_proof"]
            for field in ("old_fingerprint", "new_fingerprint", "unit_id"):
                forged = dict(proof, **{field: "forged"})
                forged.pop("digest")
                forged["digest"] = d.digest(forged)
                self.assertTrue(d.validate_equivalence_proof(".", forged,
                    old["requirements"]["identity-nvidia/two"], new["requirements"]["identity-nvidia/two"]))
            selected["selections"]["nvidia"]["lanes"].append("two")
            self.assertTrue(d.validate_equivalence_proof(".", proof,
                old["requirements"]["identity-nvidia/two"], new["requirements"]["identity-nvidia/two"]))

    def test_unknown_selector_never_means_rerun_everything(self):
        old, new, selected = self._pair()
        selected["selections"]["nvidia"]["unattributed"] = ["new/unknown.mojo"]
        with patch.object(d, "frozen_snapshot", return_value=selected), patch.object(d, "_revision", side_effect=lambda root, rev: rev):
            plan = d.equivalence_proofs(".", old, new)
        self.assertTrue(all(row["decision"] == "blocked" for row in plan["units"].values()))

    def test_limited_selected_lane_does_not_block_unaffected_lane(self):
        old, new, selected = self._pair()
        selected["selections"]["nvidia"]["precision_limitations"] = [dict(lane="one", paths=["core/shared.mojo"])]
        with patch.object(d, "frozen_snapshot", return_value=selected), patch.object(d, "_revision", side_effect=lambda root, rev: rev):
            plan = d.equivalence_proofs(".", old, new)
        self.assertEqual(plan["units"]["identity-nvidia/one"]["decision"], "blocked")
        self.assertEqual(plan["units"]["identity-nvidia/two"]["decision"], "retained")

    def test_changed_coverage_or_flags_cannot_reuse_named_edit_proof(self):
        old, new, selected = self._pair()
        for key in ("contract", "execution_environment"):
            edited = copy.deepcopy(new)
            edited["requirements"]["identity-nvidia/two"]["inputs"][key]["changed"] = True
            with patch.object(d, "frozen_snapshot", return_value=selected), patch.object(d, "_revision", side_effect=lambda root, rev: rev):
                plan = d.equivalence_proofs(".", old, edited)
            self.assertEqual(plan["units"]["identity-nvidia/two"]["decision"], "affected")

    def test_unresolved_inputs_never_reuse_even_same_fingerprint(self):
        old, new, selected = self._pair()
        new = copy.deepcopy(old)
        new["requirements"]["identity-nvidia/one"]["missing_inputs"] = ["toolchain"]
        with patch.object(d, "frozen_snapshot", return_value=selected), patch.object(d, "_revision", side_effect=lambda root, rev: rev):
            plan = d.equivalence_proofs(".", old, new)
        self.assertEqual(plan["units"]["identity-nvidia/one"]["decision"], "blocked")

    def test_missing_evidence_requires_restore_not_rerun(self):
        old, new, selected = self._pair()
        with patch.object(d, "frozen_snapshot", return_value=selected), patch.object(d, "_revision", side_effect=lambda root, rev: rev):
            plan = d.equivalence_proofs(".", old, new,
                retained_evidence={"identity-nvidia/two": [{"path": "/absent/retained-evidence.json", "sha256": "a" * 64}]})
        row = plan["units"]["identity-nvidia/two"]
        self.assertEqual(row["decision"], "retained")
        self.assertEqual(row["evidence"]["status"], "restore-required")

    def test_outside_tree_sources_are_rejected(self):
        for path in ("../secret", "/absolute", "tools/*.py"):
            with self.assertRaisesRegex(ValueError, "repository-relative"):
                d._file_hash(".", path)


class FrozenSource(unittest.TestCase):
    def test_requested_revision_not_current_or_dirty_source(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "repo"
            root.mkdir()
            def git(*args):
                return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL, text=True).strip()
            git("init")
            git("config", "user.email", "test@example.test")
            git("config", "user.name", "Dependency Test")
            (root / "tools").mkdir()
            for name in ("release_reuse.py", "lane_select.py", "bincache.py"):
                (root / "tools" / name).write_text("from pathlib import Path\nROOT=Path(__file__).resolve().parent.parent\n")
            (root / "recipe.json").write_text("frozen recipe")
            git("add", ".")
            git("commit", "-m", "frozen source")
            frozen = git("rev-parse", "HEAD")
            (root / "recipe.json").write_text("new committed recipe")
            git("add", ".")
            git("commit", "-m", "changed source")
            (root / "recipe.json").write_text("dirty source")
            request = dict(extra_paths=["recipe.json", "absent.json"])
            got = d.frozen_snapshot(root, frozen, request, log_dir=Path(temp) / "logs")
            self.assertEqual(got["revision"], frozen)
            self.assertEqual(got["extra_sources"]["recipe.json"], d.hashlib.sha256(b"frozen recipe").hexdigest())
            self.assertIsNone(got["extra_sources"]["absent.json"])
            self.assertEqual((root / "recipe.json").read_text(), "dirty source")
            self.assertEqual(len(git("worktree", "list", "--porcelain").split("worktree ")) - 1, 1)
            self.assertEqual(d.frozen_snapshot(root, frozen, request, log_dir=Path(temp) / "logs"), got)


@unittest.skipUnless(sys.platform == "darwin", "macOS clonefile source-view path")
class CurrentFrozenSource(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.root = self.base / "repo"
        self.root.mkdir()
        self.git("init")
        self.git("config", "user.email", "test@example.test")
        self.git("config", "user.name", "Dependency Test")
        (self.root / "tools").mkdir()
        for name in ("release_reuse.py", "lane_select.py", "bincache.py"):
            (self.root / "tools" / name).write_text("from pathlib import Path\nROOT=Path(__file__).resolve().parent.parent\n")
        (self.root / "recipe.json").write_text("frozen recipe")
        (self.root / "entry.sh").write_text("#!/bin/sh\nexit 0\n")
        (self.root / "entry.sh").chmod(0o755)
        (self.root / "recipe-link.json").symlink_to("recipe.json")
        self.git("add", ".")
        self.git("commit", "-m", "frozen source")
        self.revision = self.git("rev-parse", "HEAD")
        self.request = dict(extra_paths=["recipe.json", "recipe-link.json", "entry.sh", "generated.mojo"])
        self.logs = self.base / "logs"

    def git(self, *args):
        return subprocess.check_output(["git", "-C", str(self.root), *args],
                                       stderr=subprocess.DEVNULL, text=True).strip()

    def snapshot(self):
        return d.frozen_snapshot(self.root, self.revision, self.request, log_dir=self.logs)

    def assert_no_retained_result_or_worktree(self):
        self.assertFalse(list(self.logs.glob("*.json")))
        self.assertEqual(len(self.git("worktree", "list", "--porcelain").split("worktree ")) - 1, 1)

    def test_current_tree_uses_tracked_only_clones_with_modes_and_symlinks(self):
        (self.root / "generated.mojo").write_text("not committed")
        (self.root / "tools" / "__pycache__").mkdir()
        (self.root / "tools" / "__pycache__" / "stale.pyc").write_bytes(b"not source")
        worker = d._run_snapshot_worker
        observed = []

        def inspect(tree, temporary, request, log):
            observed.append(tree)
            self.assertNotEqual(tree, self.root)
            self.assertNotEqual((tree / "recipe.json").stat().st_ino, (self.root / "recipe.json").stat().st_ino)
            self.assertTrue((tree / "entry.sh").stat().st_mode & stat.S_IXUSR)
            self.assertTrue((tree / "recipe-link.json").is_symlink())
            self.assertEqual(os.readlink(tree / "recipe-link.json"), "recipe.json")
            self.assertFalse((tree / "generated.mojo").exists())
            self.assertFalse((tree / "tools" / "__pycache__").exists())
            return worker(tree, temporary, request, log)

        with patch.object(d, "_run_snapshot_worker", side_effect=inspect):
            result = self.snapshot()
        self.assertEqual(len(observed), 1)
        expected = d.hashlib.sha256(b"frozen recipe").hexdigest()
        self.assertEqual(result["extra_sources"]["recipe.json"], expected)
        self.assertEqual(result["extra_sources"]["recipe-link.json"], expected)
        self.assertIsNone(result["extra_sources"]["generated.mojo"])
        self.assertTrue((self.root / "generated.mojo").is_file())
        self.assertFalse(observed[0].exists())
        self.assertEqual(len(self.git("worktree", "list", "--porcelain").split("worktree ")) - 1, 1)

    def test_dirty_current_source_is_rejected_before_analysis(self):
        (self.root / "recipe.json").write_text("dirty")
        with patch.object(d, "_run_snapshot_worker") as worker:
            with self.assertRaisesRegex(ValueError, "differs from requested revision"):
                self.snapshot()
            worker.assert_not_called()
        self.assert_no_retained_result_or_worktree()

    def test_assume_unchanged_does_not_hide_dirty_bytes(self):
        self.git("update-index", "--assume-unchanged", "recipe.json")
        (self.root / "recipe.json").write_text("hidden dirty bytes")
        self.assertEqual(self.git("status", "--porcelain"), "")
        with self.assertRaisesRegex(ValueError, "differs from requested revision"):
            self.snapshot()
        self.assert_no_retained_result_or_worktree()

    def test_staged_change_is_rejected(self):
        (self.root / "added.json").write_text("staged addition")
        self.git("add", "added.json")
        with self.assertRaisesRegex(ValueError, "staged changes"):
            self.snapshot()
        self.assert_no_retained_result_or_worktree()

    def test_changed_executable_mode_is_rejected(self):
        (self.root / "entry.sh").chmod(0o644)
        with self.assertRaisesRegex(ValueError, "file mode changed"):
            self.snapshot()
        self.assert_no_retained_result_or_worktree()

    def test_symlink_replaced_with_regular_file_is_rejected(self):
        link = self.root / "recipe-link.json"
        link.unlink()
        link.write_text("recipe.json")
        with self.assertRaisesRegex(ValueError, "symlink changed"):
            self.snapshot()
        self.assert_no_retained_result_or_worktree()

    def test_original_source_edit_does_not_invalidate_frozen_worker(self):
        worker = d._run_snapshot_worker

        def mutate(tree, temporary, request, log):
            (self.root / "recipe.json").write_text("changed while worker was running")
            return worker(tree, temporary, request, log)

        with patch.object(d, "_run_snapshot_worker", side_effect=mutate):
            result = self.snapshot()
        self.assertEqual(result["revision"], self.revision)
        self.assertEqual(result["extra_sources"]["recipe.json"],
                         d.hashlib.sha256(b"frozen recipe").hexdigest())

    def test_original_source_edit_then_restore_keeps_frozen_result(self):
        worker = d._run_snapshot_worker

        def mutate(tree, temporary, request, log):
            source = self.root / "recipe.json"
            before = source.stat()
            source.write_text("temporary edit")
            source.write_text("frozen recipe")
            os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
            return worker(tree, temporary, request, log)

        with patch.object(d, "_run_snapshot_worker", side_effect=mutate):
            result = self.snapshot()
        self.assertEqual(result["revision"], self.revision)
        self.assertEqual(result["extra_sources"]["recipe.json"],
                         d.hashlib.sha256(b"frozen recipe").hexdigest())

    def test_original_head_change_does_not_relabel_frozen_result(self):
        worker = d._run_snapshot_worker

        def mutate(tree, temporary, request, log):
            self.git("commit", "--allow-empty", "-m", "head moved")
            return worker(tree, temporary, request, log)

        with patch.object(d, "_run_snapshot_worker", side_effect=mutate):
            result = self.snapshot()
        self.assertEqual(result["revision"], self.revision)
        self.assertNotEqual(self.git("rev-parse", "HEAD"), result["revision"])

    def test_symlink_to_original_checkout_cannot_escape_frozen_view(self):
        link = self.root / "recipe-link.json"
        link.unlink()
        link.symlink_to(self.root / "recipe.json")
        self.git("add", "recipe-link.json")
        self.git("commit", "-m", "external source link")
        self.revision = self.git("rev-parse", "HEAD")
        with patch.object(d, "_run_snapshot_worker") as worker:
            with self.assertRaisesRegex(ValueError, "symlink escapes frozen view"):
                self.snapshot()
            worker.assert_not_called()
        self.assert_no_retained_result_or_worktree()

    def test_view_write_and_chmod_cannot_change_original_source(self):
        worker = d._run_snapshot_worker
        mode = (self.root / "recipe.json").stat().st_mode

        def mutate(tree, temporary, request, log):
            (tree / "recipe.json").write_text("analyzer unexpectedly wrote")
            (tree / "recipe.json").chmod(0o600)
            self.assertEqual((self.root / "recipe.json").read_text(), "frozen recipe")
            self.assertEqual((self.root / "recipe.json").stat().st_mode, mode)
            return worker(tree, temporary, request, log)

        with patch.object(d, "_run_snapshot_worker", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "differs from requested revision"):
                self.snapshot()
        self.assert_no_retained_result_or_worktree()

    def test_frozen_view_edit_then_restore_is_still_rejected(self):
        worker = d._run_snapshot_worker

        def mutate(tree, temporary, request, log):
            source = tree / "recipe.json"
            before = source.stat()
            source.write_text("temporary edit")
            source.write_text("frozen recipe")
            os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
            return worker(tree, temporary, request, log)

        with patch.object(d, "_run_snapshot_worker", side_effect=mutate):
            with self.assertRaisesRegex(ValueError, "frozen dependency view changed"):
                self.snapshot()
        self.assert_no_retained_result_or_worktree()

    def test_worker_failure_removes_only_its_temporary_worktree(self):
        with patch.object(d, "_run_snapshot_worker", side_effect=ValueError("worker failure")):
            with self.assertRaisesRegex(ValueError, "worker failure"):
                self.snapshot()
        self.assert_no_retained_result_or_worktree()


if __name__ == "__main__":
    unittest.main()


def test_version_literal_view_rejects_other_changes():
    before = b'# same\n__version__ = "0.8.58"\n'
    after = b'# same\n__version__ = "0.8.59"\n'
    assert d._version_literal_view(before) == d._version_literal_view(after)
    assert d._version_literal_view(before) != d._version_literal_view(after.replace(b'same', b'changed'))
    for source in (b'__version__ = version()\n', b'__version__ = "0.8.59"\nexecute()\n',
                   b'__version__ = "0.8.59"; other = 1\n', b'__version__ = "arbitrary"\n'):
        assert d._version_literal_view(source) is None


def test_scope_annotation_preserves_inputs_and_separate_multi_device_obligations():
    import pytest
    snap = snapshot()
    fixture = Dependencies(); fixture.setUp()
    old = manifest(fixture.expand(('identity',), snap), 'b' * 40)
    for wanted in old['requirements'].values():
        wanted['required'] = True
    old['requirements'].update({name: dict(required=True, kind='multi', inputs={}, fingerprint=d.digest({}))
                               for name in ('multi-gpu-nvidia', 'multi-gpu-amd')})
    rows = {
        'one': dict(kind='cpu-host-route-only', claim_devices=1, oracle='self', anchor='in-cell-self',
                    required_columns=['cpu'], excluded={'nvidia': 'host only'}, separate_requirements=[]),
        'two': dict(kind='multi-device-driver', claim_devices=2, oracle='independent', anchor='in-cell-independent',
                    required_columns=[], excluded={'nvidia': 'two physical devices required'},
                    separate_requirements=['multi-gpu-nvidia', 'multi-gpu-amd'])}
    scope = dict(schema='mojolearn.release-identity-applicability.v1', source_revision='b' * 40,
                 lanes=rows, digest='scope-digest')
    with patch.object(d, 'identity_scope_snapshot', return_value=scope):
        new = d.scope_identity_manifest('.', old)
        for name in old['requirements']:
            assert old['requirements'][name]['inputs'] == new['requirements'][name]['inputs']
            assert old['requirements'][name]['fingerprint'] == new['requirements'][name]['fingerprint']
        assert not new['requirements']['identity-nvidia/one']['required']
        assert not new['requirements']['identity-nvidia/two']['required']
        assert new['requirements']['multi-gpu-nvidia']['required']
        assert new['requirements']['multi-gpu-amd']['required']
        assert old['requirements']['identity-nvidia/one']['required']
        old['requirements']['multi-gpu-amd']['required'] = False
        deferred = d.scope_identity_manifest('.', old)
        assert deferred['requirements']['multi-gpu-amd']['required'] is False
        assert deferred['requirements']['identity-nvidia/two']['dependency_provenance']['identity_applicability']['separate_requirements'] == ['multi-gpu-nvidia', 'multi-gpu-amd']


class _Row:
    def __init__(self, tier, name):
        self.tier, self.name = tier, name


def test_frozen_bindings_uses_current_mode_registry():
    calls = []

    class Current:
        @staticmethod
        def bindings(target, modes=None):
            calls.append((target, modes))
            return [_Row("identical", "a"), _Row("host", "h")]
    rows = d.frozen_bindings(Current, "linux-64")
    assert calls == [("linux-64", ["identical"])]
    assert [r.name for r in rows] == ["a", "h"]


def test_frozen_bindings_filters_one_argument_historical_registry():
    class Historical:
        @staticmethod
        def bindings(target):
            return [_Row("fast", "f"), _Row("identical", "a"), _Row("host", "h"), _Row("fast", "g")]
    assert [r.name for r in d.frozen_bindings(Historical, "linux-64")] == ["a", "h"]


def test_frozen_bindings_refuses_unknown_registry_signature():
    import pytest

    class Unknown:
        @staticmethod
        def bindings(target, tiers):
            return []
    with pytest.raises(ValueError, match="unrecognized frozen binding registry"):
        d.frozen_bindings(Unknown, "linux-64")


# The published 0.8.55 source: its registry takes only target and lists fast,
# identical and host rows. Its wheels ship 33 identical bindings per GPU set
# and on Metal, and 42 host bindings per CPU target.
HISTORICAL_0855 = "a8255828597a3bd19ac7bb5f657a38d09b8c732f"


def test_frozen_bindings_reads_the_published_0855_registry(tmp_path):
    import pytest
    repo = Path(__file__).resolve().parents[1]
    if subprocess.run(["git", "-C", str(repo), "cat-file", "-e", HISTORICAL_0855 + "^{commit}"],
                      capture_output=True).returncode:
        pytest.skip("published 0.8.55 source is not in this clone")
    archive = subprocess.run(["git", "-C", str(repo), "archive", HISTORICAL_0855, "tools", "packaging",
                              "python/mojolearn"], check=True, capture_output=True).stdout
    subprocess.run(["tar", "-x", "-C", str(tmp_path)], input=archive, check=True)
    probe = (
        "import collections, inspect, json, sys\n"
        "sys.path[:0] = [sys.argv[1] + '/tools']\n"
        "import release_reuse as rr\n"
        "sys.path.insert(0, sys.argv[2])\n"
        "import release_requirement_dependencies as d\n"
        "out = dict(params=list(inspect.signature(rr.bindings).parameters))\n"
        "for t in (rr.LINUX, rr.MACOS):\n"
        "    rows = d.frozen_bindings(rr, t)\n"
        "    allrows = [b for b in rr.bindings(t) if b.tier != 'fast']\n"
        "    out[t] = dict(tiers=dict(collections.Counter('/'.join((b.vendor, b.arch, b.tier)) for b in rows)),\n"
        "        same=[b.key for b in rows] == [b.key for b in allrows])\n"
        "print(json.dumps(out))\n")
    result = subprocess.run([sys.executable, "-I", "-c", probe, str(tmp_path), str(Path(__file__).resolve().parent)],
                            check=True, capture_output=True, text=True, cwd=tmp_path)
    out = json.loads(result.stdout)
    assert out["params"] == ["target"]
    assert out["linux-64"]["same"] and out["osx-arm64"]["same"]
    assert out["linux-64"]["tiers"] == {"cuda/sm_89/identical": 33, "cuda/sm_80/identical": 33,
                                        "hip/gfx942/identical": 33, "host//host": 42}
    assert out["osx-arm64"]["tiers"] == {"metal/apple/identical": 33, "host//host": 42}
