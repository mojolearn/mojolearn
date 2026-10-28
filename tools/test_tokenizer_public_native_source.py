"""Source contracts only; these do not qualify compiled tokenizer runtime."""
import ast
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
NATIVE = ROOT / "bindings/tokenizer_public_native.mojo"


class TokenizerPublicSource(unittest.TestCase):
    def test_every_public_operation_has_one_module_global_native_call(self):
        paths = ("python/mojolearn/tokenizer.py", "python/mojolearn/models/tokenizer.py")
        count = 0
        for path in paths:
            tree = ast.parse((ROOT / path).read_text())
            for cls in tree.body:
                if not isinstance(cls, ast.ClassDef) or cls.name.startswith("_"):
                    continue
                for method in cls.body:
                    if not isinstance(method, ast.FunctionDef):
                        continue
                    if method.name.startswith("_") and method.name not in {"__init__", "__repr__"}:
                        continue
                    body = method.body
                    if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                        body = body[1:]
                    self.assertEqual(len(body), 1, (path, cls.name, method.name))
                    self.assertIsInstance(body[0], ast.Return)
                    call = body[0].value
                    self.assertIsInstance(call, ast.Call)
                    self.assertIsInstance(call.func, ast.Attribute)
                    self.assertIsInstance(call.func.value, ast.Name)
                    self.assertIn(call.func.value.id, {"_NATIVE", "_TOKENIZER_NATIVE"})
                    self.assertEqual(call.func.attr, "tokenizer_public_call")
                    self.assertEqual(sum(isinstance(node, ast.Call) for node in ast.walk(call)), 1)
                    self.assertIn('operation == "' + call.args[0].value + '"', NATIVE.read_text())
                    count += 1
        self.assertEqual(count, 40)

    def test_public_alias_and_standalone_pattern_functions(self):
        source = (ROOT / "python/mojolearn/tokenizer.py").read_text()
        self.assertIn('"GPT2Tokenizer": "BpeTokenizer"', source)
        model = ast.parse((ROOT / "python/mojolearn/models/tokenizer.py").read_text())
        for node in model.body:
            if isinstance(node, ast.FunctionDef) and node.name in {"pretokenize", "pattern_name"}:
                self.assertEqual(sum(isinstance(n, ast.Call) for n in ast.walk(node)), 1)

    def test_arithmetic_routes_and_negative_controls_stay_in_existing_owners(self):
        source = NATIVE.read_text()
        for owner in ("bpe_encode_into(owner[]", "bpe_encode_batch_into(owner[]",
                      "bpe_decode_seq_bytes(owner[]", "bpe_train_fill(handle,",
                      "vocab_encode_ids(owner[]", "vocab_decode_ids(owner[]",
                      "render_ranks(vocab)", "render_tokenizer_json(vocab,"):
            self.assertIn(owner, source)
        self.assertIn('getenv("MOJOLEARN_BPE_TRAINER_SABOTAGE", "")', source)
        self.assertIn('context[5][1]', source)  # retained optional native fast route

    def test_admission_precedes_buffer_acquisition_and_instance_storage(self):
        source = NATIVE.read_text()
        owner = source.split("def _owner(", 1)[1].split("def _new(", 1)[0]
        self.assertLess(owner.index("py_type(value) is not cls"), owner.index("_fields(value)"))
        self.assertIn('native_instance_item', source)
        capture = source.split("def _bytes(", 1)[1].split("def _py_bytes(", 1)[0]
        self.assertLess(capture.index("admit_native_buffer_exporter(raw)"), capture.index("AdmittedBorrowedBuffer(raw)"))
        self.assertIn("pin.release()", capture)
        for forbidden in ("tokenizer._handle", "tokenizer.normalizer", "Python.import_module", "operator.index", "isinstance", "__fspath__"):
            self.assertNotIn(forbidden, source)

    def test_finished_outputs_and_unicode_exception_ownership_are_native(self):
        source = NATIVE.read_text()
        for expected in ("PyBytes_FromStringAndSize", "PyUnicode_AsUTF8String", "stash_python_error()",
                         "optimizer_python_error(error)", "_trained_finish(", "with open(String(path), \"w\")"):
            self.assertIn(expected, source)

    def test_legacy_generation_consumers_have_a_connected_native_entry(self):
        source = NATIVE.read_text()
        for entry in ("tokenizer_vocab_encode", "tokenizer_vocab_decode",
                      "tokenizer_vocab_pretokenize", "tokenizer_vocab_fingerprint"):
            self.assertIn("def " + entry + "(", source)
        fingerprint = source.split("def tokenizer_vocab_fingerprint(", 1)[1].split("def _vocab_config_pattern(", 1)[0]
        self.assertIn("_owner(tokenizer, context[0])", fingerprint)
        self.assertIn("vocab_fingerprint_hex(owner[].tok.value(), tail)", fingerprint)
        self.assertNotIn("py_call(_native_function", fingerprint)
        connection = (ROOT / "python/mojolearn/models/tokenizer.py").read_text()
        self.assertIn('tokenizer_public_call("connect_vocab", (), _TOKENIZER_CONTEXT)', connection)

    def test_corpus_helpers_have_complete_native_boundaries(self):
        expected = {"_identity_from_tokens_text", "_identity_of_ranks_file", "_identity_of_handle",
                    "_native_trainer", "_train_native", "_read_trained", "_train_corpus",
                    "_gather_rows", "_encode_corpus"}
        tree = ast.parse((ROOT / "python/mojolearn/tokenizer.py").read_text())
        found = set()
        for method in ast.walk(tree):
            if not isinstance(method, ast.FunctionDef) or method.name not in expected:
                continue
            body = method.body
            if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                body = body[1:]
            self.assertEqual(len(body), 1, method.name)
            self.assertIsInstance(body[0], ast.Return)
            calls = [node for node in ast.walk(body[0]) if isinstance(node, ast.Call)]
            self.assertEqual(len(calls), 1, method.name)
            self.assertEqual(calls[0].func.attr, "tokenizer_public_call")
            found.add(method.name)
        self.assertEqual(found, expected)

    def test_corpus_native_storage_and_original_typed_algorithms_are_preserved(self):
        source = NATIVE.read_text()
        corpus = source.split("def tokenizer_bpe_identity_file(", 1)[1].split("def tokenizer_public_call(", 1)[0]
        for retained in ("bpe_encode_corpus_into(owner[]", "bpe_train_corpus_fill(handle,",
                         "tokens_gather_rows_into(src, dst,", 'native_array_zeros("i", total)',
                         'neural_array_snapshot(ids_all, context)', 'neural_array_snapshot(output, context)',
                         '"mojolearn.bpe-trainer.v1"', 'PyUnicode_AsASCIIString'):
            self.assertIn(retained, corpus)
        for forbidden in ("hashlib", 'Python.import_module', 'py_call(module.', 'py_call(native_loaded_module("array").array',
                          "tokenizer._m", "vocabulary.render", "source.__buffer__"):
            self.assertNotIn(forbidden, corpus)
        typed = corpus.split("def tokenizer_gather_rows_native(", 1)[1].split("def tokenizer_gather_rows(", 1)[0]
        self.assertNotIn("PythonObject", typed)
        self.assertNotIn("py_call", typed)


if __name__ == "__main__":
    unittest.main()
