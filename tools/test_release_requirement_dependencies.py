#!/usr/bin/env python3
"""Dependency scoping and frozen-source integrity, without builds or GPUs."""
import copy
import json
from pathlib import Path
import subprocess
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
        self.config = dict(build_flags={}, release_modes=["identical"])

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


if __name__ == "__main__":
    unittest.main()
