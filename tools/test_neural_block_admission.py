"""Source checks only; public hook and numeric checks require compiled bindings."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from boundary_keys_plain import plain  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = (ROOT / "bindings/neural_block_public_door.mojo").read_text()
ADMISSION = (ROOT / "bindings/neural_block_admission_door.mojo").read_text()
# The held-key door reads by name through plain() (tools/boundary_keys_plain.py).
ARRAY_ADMISSION = plain("bindings/native_array_snapshot_door.mojo", (ROOT / "bindings/native_array_snapshot_door.mojo").read_text())


class NeuralAdmissionSource(unittest.TestCase):
    def test_public_owner_admits_before_scalar_and_binding_selection(self):
        # neural_block_owner delegates to the full owner build (db970cfce4).
        # Both owner forms share one setup (snapshot first); only the parallel
        # worker's holder is an object, the plan build's owner is typed.
        setup = PUBLIC.split("def _neural_block_owner_setup(")[1].split("\ndef ", 1)[0]
        body = PUBLIC.split("def _neural_block_owner_full(")[1].split("\ndef ", 1)[0]
        typed = PUBLIC.split("def _neural_block_owner_typed(")[1].split("\ndef ", 1)[0]
        door = PUBLIC.split("def neural_block_owner(")[1].split("\ndef ", 1)[0]
        execution = PUBLIC.split("def neural_block_execution(")[1].split("\ndef ", 1)[0]
        self.assertIn("return _neural_block_owner_full(mods, block, context)", door)
        self.assertLess(setup.index("neural_block_snapshot(mods, block, context)"), setup.index('String(fields["_native_kind"])'))
        for part in (setup, body, typed):
            self.assertNotIn('object.__getattribute__', part)
            self.assertIn("_neural_block_owner_setup(", part if part is not setup else "_neural_block_owner_setup(")
        self.assertIn('"_native_block_fields"', body)
        self.assertNotIn('module("types")', typed)
        self.assertNotIn("py_setattr", typed)
        self.assertIn("causal_lm_admit(mods, admission,", typed)
        self.assertIn("_neural_block_owner_typed(mods, block, context)", execution)
        self.assertNotIn("_neural_block_owner_full", execution)

    def test_array_metadata_uses_native_slots_and_retains_owner(self):
        # The first overload is the per-call NativeModules wrapper; check the real body.
        body = ARRAY_ADMISSION.split("def native_array_snapshot(mut mods: NativeModules,")[1]
        self.assertIn("native_array_snapshot as neural_array_snapshot", ADMISSION)
        self.assertIn('slots[names[i]].__get__', body)
        self.assertIn('AdmittedBorrowedBuffer(store)', body)
        self.assertIn('slots["_base"].__set__, result, value', body)
        self.assertIn('offset + count * itemsize > layout.count * layout.itemsize', body)
        self.assertIn('py_type(original_view) is not b.memoryview', body)
        self.assertIn('view_layout.address != Int(py=values["_addr"])', body)
        self.assertIn('slots["_mv"].__set__, result, retained_view', body)

    def test_public_input_protocols_guarded(self):
        self.assertNotIn("operator.index", PUBLIC)
        self.assertNotIn("block._W_NAMES", PUBLIC)
        self.assertIn("admit_native_buffer_exporter(value)", PUBLIC)
        self.assertIn("neural_native_scalar(mods, batch, context, True)", PUBLIC)
        self.assertIn("admit_block_f32(mods, grad_output", PUBLIC)

    def test_native_dictionary_descriptor_guard_reused(self):
        body = ADMISSION.split("def neural_record_fields(")[1].split("\ndef ", 1)[0]
        self.assertIn("var fields = _fields(mods, value)", body)
        self.assertNotIn("getattr", body.split("var fields = _fields(mods, value)", 1)[1])
        self.assertIn('"__get__"', body)

    def test_state_publication_preserves_counters_and_failure_status(self):
        body = PUBLIC.split("def neural_block_forward_binding(")[1].split("def neural_training_entries(")[0]
        self.assertEqual(body.count("_neural_publish_state(state_value, captured)"), 2)
        self.assertIn("_neural_state_capture(mods, owner, state_value)", body)


if __name__ == "__main__":
    unittest.main()
