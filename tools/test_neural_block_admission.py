"""Source checks only; public hook and numeric checks require compiled bindings."""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = (ROOT / "bindings/neural_block_public.mojo").read_text()
ADMISSION = (ROOT / "bindings/neural_block_admission.mojo").read_text()


class NeuralAdmissionSource(unittest.TestCase):
    def test_public_owner_admits_before_scalar_and_binding_selection(self):
        body = PUBLIC.split("def neural_block_owner(")[1].split("def neural_block_init_binding(")[0]
        self.assertLess(body.index("neural_block_snapshot(block, context)"), body.index('String(fields["_native_kind"])'))
        self.assertNotIn('object.__getattribute__', body)
        self.assertIn('"_native_block_fields"', body)

    def test_array_metadata_uses_native_slots_and_retains_owner(self):
        body = ADMISSION.split("def neural_array_snapshot(")[1].split("def neural_admit_weights(")[0]
        self.assertIn('slots[names[i]].__get__', body)
        self.assertIn('AdmittedBorrowedBuffer(store)', body)
        self.assertIn('slots["_base"].__set__, result, value', body)
        self.assertIn('offset + count * itemsize > layout.count * layout.itemsize', body)

    def test_public_input_protocols_guarded(self):
        self.assertNotIn("operator.index", PUBLIC)
        self.assertNotIn("block._W_NAMES", PUBLIC)
        self.assertIn("admit_native_buffer_exporter(value)", PUBLIC)
        self.assertIn("neural_native_scalar(batch, context, True)", PUBLIC)
        self.assertIn("neural_block_f32(grad_output", PUBLIC)

    def test_native_dictionary_descriptor_guard_reused(self):
        body = ADMISSION.split("def neural_record_fields(")[1].split("def neural_array_slots(")[0]
        self.assertIn("var fields = _fields(value)", body)
        self.assertNotIn("getattr", body.split("var fields = _fields(value)", 1)[1])
        self.assertIn('"__get__"', body)

    def test_state_publication_preserves_counters_and_failure_status(self):
        body = PUBLIC.split("def neural_block_forward_binding(")[1].split("def neural_training_entries(")[0]
        self.assertEqual(body.count("_neural_publish_state(state_value, captured)"), 2)
        self.assertIn("_neural_state_capture(owner, state_value)", body)


if __name__ == "__main__":
    unittest.main()
