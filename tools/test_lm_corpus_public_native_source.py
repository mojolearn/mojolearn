"""Source controls only: no compiled corpus runtime or vendor qualification."""
import ast
import importlib.util
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "python/mojolearn/lm_corpus.py"
NATIVE = ROOT / "bindings/lm_corpus_public_native.mojo"
METADATA = ROOT / "bindings/lm_corpus_metadata.mojo"


class CorpusPublicSource(unittest.TestCase):
    def test_all_record_operations_are_single_native_calls(self):
        tree = ast.parse(PUBLIC.read_text())
        count = 0
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef):
                continue
            body = node.body
            if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
                body = body[1:]
            self.assertEqual(len(body), 1, node.name)
            self.assertIsInstance(body[0], ast.Return, node.name)
            calls = [n for n in ast.walk(body[0]) if isinstance(n, ast.Call)]
            self.assertEqual(len(calls), 1, node.name)
            call = calls[0]
            self.assertEqual(ast.unparse(call.func), "_NATIVE.lm_corpus_public_call")
            self.assertIn('operation == "' + call.args[0].value + '"', NATIVE.read_text())
            count += 1
        self.assertEqual(count, 21)

    def test_preparation_owns_cache_and_existing_tokenizer_routes(self):
        source = NATIVE.read_text()
        for entry in ("tokenizer_bpe_train_corpus(data,", "tokenizer_bpe_encode_corpus(tokenizer,",
                      "tokenizer_gather_rows_native(Int(py=source._addr)",
                      "corpus_atomic_begin(final)", "corpus_atomic_commit(temporary, final)",
                      "corpus_atomic_abort(temporary)", "corpus_hash_file(_join(final,"):
            self.assertIn(entry, source)
        for forbidden in ("Python.import_module", "tokenizer._encode_corpus", "tokenizer.identity",
                          "py_call(progress", "py_call(owner.ids", "fill(tmp)", "hashlib", "threading"):
            self.assertNotIn(forbidden, source)
        self.assertIn("_progress(args[8])", source)
        self.assertIn("Python progress callbacks", source)

    def test_cache_identity_and_document_recipe_are_retained(self):
        source = NATIVE.read_text()
        for constant in ("mojolearn.byte-lm.tokens.v1", "mojolearn.byte-lm.corpus.v1",
                         "mojolearn.bpe-vocabulary.v1", "train-range-modulo.v1"):
            self.assertIn(constant, source)
        for value in ('"-v" + String(size)', '"-f" + String(frequency)', '"-s" + String(hi - lo)',
                      '"-d" + String(document_bytes)', '"user-" + _prefix(identity["sha256"])',
                      'not inserted and not recognized (allow_endoftext=False)',
                      'ending at the last 0x0A at or before the limit'):
            self.assertIn(value, source)

    def test_mmap_protocol_admission_precedes_capture(self):
        source = METADATA.read_text()
        array = source.split("def corpus_array(", 1)[1].split("def corpus_empty(", 1)[0]
        self.assertLess(array.index("admit_native_buffer_exporter(store)"), array.index("py_call(bi.memoryview, store)"))
        mapped = source.split("def corpus_mapped(", 1)[1].split("def corpus_ids_snapshot(", 1)[0]
        self.assertLess(mapped.index("corpus_mmap_hooks(cls)"), mapped.index("mapping = py_call(cls,"))
        self.assertIn('hooks[keys[i]] is not expected[keys[i]]', mapped)
        for name in ("__new__", "__init__", "__del__", "__buffer__", "__release_buffer__"):
            self.assertIn('"' + name + '"', source)
        self.assertIn('connect_native_mmap_exporter(context["mmap_class"])', source)
        self.assertIn('_corpus_connect_module("mmap", _corpus_mmap, ("mmap",))', PUBLIC.read_text())

    def test_mutable_records_are_admitted_before_instance_storage(self):
        source = METADATA.read_text()
        fields = source.split("def corpus_fields(", 1)[1].split("def corpus_new(", 1)[0]
        self.assertLess(fields.index('"__getattr__"'), fields.index("return _fields(owner)"))
        self.assertLess(fields.index("py_contains(namespace, names[j])"), fields.index("return _fields(owner)"))
        self.assertIn("native integer; Python conversion hooks are unsupported", source)
        self.assertIn("neural_array_snapshot(value, context)", source)

    def test_checkpoint_check_only_captures_vocabulary_metadata(self):
        source = NATIVE.read_text().split("def _expected_vocabulary(", 1)[1].split("def _require_vocabulary(", 1)[0]
        self.assertIn("var fields = _fields(model)", source)
        self.assertIn('py_contains(_class_fields(bases[j]), "__get__")', source)
        self.assertIn("return corpus_plain(vocabulary)", source)
        self.assertNotIn("corpus_plain(model)", source)
        self.assertNotIn("model.data_schedule", source)
        self.assertNotIn("hasattr", source)

    def test_iterator_is_lazy_native_and_closes_after_failure(self):
        source = NATIVE.read_text()
        make = source.split("def _prefetch(", 1)[1].split("def _iterator_next(", 1)[0]
        advance = source.split("def _iterator_next(", 1)[1].split("def _batch_describe(", 1)[0]
        self.assertNotIn("_batch_ids(", make)
        self.assertIn("LmCorpusIteratorHandle(owner, start, start + steps)", make)
        self.assertIn("_batch_method(handle[].owner, batch_fields, context)", advance)
        self.assertIn("_batch_ids(handle[].owner, PythonObject(step), context, batch_fields)", advance)
        self.assertIn("except error:\n        handle[].closed = True", advance)
        self.assertIn('ExceptionType("PyExc_StopIteration")', source)

    def test_compile_probe_reaches_complete_public_dispatch(self):
        probe = (ROOT / "experiments/peer3_neural_tokenizer_20261027/compile_corpus_public.mojo").read_text()
        self.assertIn("return lm_corpus_public_c(owner, arguments)", probe)
        self.assertIn("lm_corpus_public_call(String(args[0]), args[1])", NATIVE.read_text())

    def test_new_native_shells_have_no_untagged_compute_loops(self):
        spec = importlib.util.spec_from_file_location("corpus_shell_guard", ROOT / "tools/hooks/no_compute_in_shells.py")
        guard = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(guard)
        for path in (NATIVE, METADATA):
            self.assertEqual(guard.scan(path.read_text()), [], str(path))


if __name__ == "__main__":
    unittest.main()
