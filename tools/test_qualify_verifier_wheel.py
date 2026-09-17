import copy
import unittest

from qualify_verifier_wheel import admit


class VerifierAdmissionTests(unittest.TestCase):
    def models(self):
        models = [{"lane": "rf-clf", "fixture": "base"}]
        doc = dict(exit=0, verdict="VERIFIED", selection={"models_only": True},
                   cells=[dict(lane="portable:rf-clf", fixture="base", part=part, state="IDENTICAL")
                          for part in ("model", "batch")])
        return doc, models

    def test_models_require_every_manifest_entry_and_part(self):
        doc, models = self.models()
        admit("models", doc, models)
        for part in (0, 1):
            broken = copy.deepcopy(doc)
            broken["cells"].pop(part)
            with self.assertRaises(AssertionError):
                admit("models", broken, models)
        with self.assertRaises(AssertionError):
            admit("models", doc, models + [{"lane": "ols", "fixture": "base"}])

    def test_success_headline_cannot_hide_bad_cells(self):
        for state in ("REFUSED", "DIVERGENT", "OWED", "N/A"):
            doc, models = self.models()
            doc["cells"][0]["state"] = state
            with self.assertRaises(AssertionError):
                admit("models", doc, models)

    def test_duplicate_cells_refuse(self):
        doc, models = self.models()
        doc["cells"].append(doc["cells"][0])
        with self.assertRaises(AssertionError):
            admit("models", doc, models)

    def test_self_test_requires_clean_match_and_detected_perturbation(self):
        doc = dict(passed=True, clean={"state": "IDENTICAL"}, perturbed={"state": "DIVERGENT"})
        admit("self-test", doc, [])
        for part in ("clean", "perturbed"):
            broken = copy.deepcopy(doc)
            broken[part]["state"] = "REFUSED"
            with self.assertRaises(AssertionError):
                admit("self-test", broken, [])


if __name__ == "__main__":
    unittest.main()
