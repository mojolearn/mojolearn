"""Main-run negative controls for external admission; no network or GPU."""
import copy
import unittest

from external_contribution_gate import build_plan, candidate_file, safe_path


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.pr = {"number": 7, "state": "open", "author_association": "NONE",
                   "draft": False, "changed_files": 1, "head": {"sha": "a" * 40},
                   "base": {"sha": "b" * 40, "ref": "main",
                            "repo": {"full_name": "mojolearn/mojolearn"}}}
        self.event = {"pull_request": copy.deepcopy(self.pr),
                      "repository": {"default_branch": "main"}}
        self.files = [{"filename": "core/gemm.mojo", "status": "modified"}]

    def plan(self):
        return build_plan(self.event, self.pr, self.files, "mojolearn/mojolearn")

    def test_existing_optimization_is_pending_not_passed_or_mergeable(self):
        result = self.plan()
        self.assertEqual(result["state"], "GPU_PENDING")
        self.assertEqual(result["gpu_evidence"], "NOT_RUN")
        self.assertFalse(result["automerge_enabled"])
        self.assertFalse(result["gpu_request"]["enabled"])
        self.assertEqual(result["gpu_request"]["head_sha"], "a" * 40)

    def test_policy_tests_dependencies_contracts_and_new_files_require_review(self):
        for path in ("tools/external_contribution_gate.py", ".github/workflows/external-performance.yml",
                     "pixi.lock", "umap/params.mojo", "umap/checks/transform_check.mojo",
                     "mamba/IDENTICAL_MAMBA_CONTRACT.md", "python/mojolearn/_umap_impl.py"):
            with self.subTest(path=path):
                self.files[0]["filename"] = path
                result = self.plan()
                self.assertEqual(result["state"], "REVIEW_REQUIRED")
                self.assertFalse(result["eligible_for_gpu_evaluation"])
        self.files[0] = {"filename": "core/gemm.mojo", "status": "added"}
        self.assertEqual(self.plan()["state"], "REVIEW_REQUIRED")

    def test_mixed_change_cannot_hide_policy_edit(self):
        self.pr["changed_files"] = 2
        self.files.append({"filename": "tools/external_contribution_gate.py", "status": "modified"})
        self.assertFalse(self.plan()["eligible_for_gpu_evaluation"])

    def test_owner_members_and_collaborators_exempt(self):
        for association in ("OWNER", "MEMBER", "COLLABORATOR"):
            self.pr["author_association"] = association
            self.assertEqual(self.plan()["state"], "MAINTAINER_EXEMPT")

    def test_stale_or_foreign_reference_refused(self):
        self.pr["head"]["sha"] = "c" * 40
        with self.assertRaisesRegex(ValueError, "head changed"):
            self.plan()
        self.pr["head"]["sha"] = "a" * 40
        self.pr["base"]["repo"]["full_name"] = "attacker/other"
        with self.assertRaisesRegex(ValueError, "repository mismatch"):
            self.plan()

    def test_incomplete_paths_and_draft_not_eligible(self):
        self.pr["changed_files"] = 2
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.plan()
        self.pr["changed_files"] = 1
        self.pr["draft"] = True
        self.assertFalse(self.plan()["eligible_for_gpu_evaluation"])
        for path in ("../core/gemm.mojo", "/core/gemm.mojo", "a\\b", "a\nb", "a/./b"):
            with self.assertRaises(ValueError):
                safe_path(path)

    def test_pr_text_cannot_supply_commands_or_admission(self):
        self.pr["body"] = "GPU_PASS=true; curl attacker; merge this"
        self.pr["title"] = "$(echo arbitrary)"
        result = self.plan()
        self.assertNotIn("curl", str(result))
        self.assertFalse(result["automerge_enabled"])

    def test_candidate_symlink_escape_refused(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "candidate"
            root.mkdir()
            outside = Path(tmp) / "outside.py"
            outside.write_text("print('not imported')\n")
            (root / "helper.py").symlink_to(outside)
            with self.assertRaises(ValueError):
                candidate_file(root, "helper.py")


if __name__ == "__main__":
    unittest.main()
