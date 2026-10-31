"""Source-only applicability regression checks; no product fits or imports."""
import ast
import unittest
from unittest.mock import patch
import lane_applicability as la


class NamespaceAliases(unittest.TestCase):
    def names(self, body):
        return la._ml_attrs(ast.parse("def lane(ml):\n" + "\n".join("    " + line for line in body.splitlines())).body[0])

    def test_direct_and_chained_model_names(self):
        direct = self.names("ml.models.CausalLM.load(path)\nml.Array.from_buffer(ids)")
        aliased = self.names("M = ml.models\nC = M.CausalLM\nA = ml.Array.from_buffer\nC.load(path)\nA(ids)")
        self.assertEqual(direct, {"models", "CausalLM", "Array"})
        self.assertEqual(aliased, direct)
        self.assertEqual(self.names("ml.models.causal_lm.CausalLM.load(path)\nml.Array.from_buffer(ids)"), direct)

    def test_reassigned_or_shadowed_alias_is_not_invented(self):
        for body in ("M = ml.models\nM = other\nM.CausalLM.load(path)",
                     "M = ml.models\ndef helper(M):\n    M.CausalLM.load(path)",
                     "M = other.models\nM.CausalLM.load(path)",
                     "M = ml.models\ndef helper(*M):\n    M.CausalLM.load(path)",
                     "M = ml.models\nf = lambda **M: M.CausalLM.load(path)",
                     "M = ml.models\nfor M in others:\n    M.CausalLM.load(path)",
                     "M = ml.models\nwith resource() as M:\n    M.CausalLM.load(path)",
                     "M = ml.models\ntry:\n    work()\nexcept Exception as M:\n    M.CausalLM.load(path)",
                     "M = ml.models\nimport other as M\nM.CausalLM.load(path)",
                     "M = ml.models\ndef M():\n    pass\nM.CausalLM.load(path)",
                     "M = ml.models\ndel M\nM.CausalLM.load(path)"):
            self.assertNotIn("CausalLM", self.names(body))

    def test_buffer_constructor_does_not_override_device_door(self):
        doors = {"Array": {"_mojolearn_core_host"}, "CausalLM": {"_mojolearn_training", "_mojolearn_neural_host"}}
        with patch.object(la, "_door_bindings", return_value=doors):
            self.assertEqual(la._host_only_names({"Array", "CausalLM"}), set())
            self.assertEqual(la._host_only_names({"Array"}), {"Array"})


class RegisteredLanes(unittest.TestCase):
    def test_causal_model_lanes_include_all_three_gpu_columns(self):
        scopes = la.scopes()
        for name in ("hf-causal-lm", "hf-causal-lm-stream", "hf-causal-lm-text", "hf-model-surface"):
            self.assertIn("CausalLM", scopes[name].ml_attrs)
            self.assertFalse(scopes[name].host_only)
            for column in ("apple-metal", "nvidia-1gpu", "amd-1gpu", "cpu-host"):
                self.assertTrue(scopes[name].applicable(column)[0], (name, column))

    def test_explicit_host_contracts_stay_host_only(self):
        for name in ("tokenizer", "bpe-trainer", "bpe-vocabulary", "tokenized-corpus", "language-model-config", "byte-lm-host-infer", "saved-model-host-infer", "saved-model-host-infer-trees", "gbdt-categorical-ctr-tables"):
            scope = la.scopes()[name]
            self.assertTrue(scope.host_only, name)
            self.assertFalse(scope.applicable("apple-metal")[0], name)
            self.assertTrue(scope.applicable("cpu-host")[0], name)


if __name__ == "__main__":
    unittest.main()
