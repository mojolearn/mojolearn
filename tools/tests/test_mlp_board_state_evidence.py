"""CPU-only evidence tests; no bindings, training or GPU launches."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('mlp_board', Path(__file__).parents[1] / 'bench_board_neural.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


def state():
    return dict(schema='test-v1', architecture=[8, 16, 3], numeric_mode='identical',
                parameter_order=list(b.MLP_NAMES),
                weights={n: np.arange(4, dtype=np.float32).reshape(2, 2) for n in b.MLP_NAMES},
                optimizer=dict(kind='AdamW', step=2, m=np.ones(16, np.float32),
                               v=np.ones(16, np.float32), flags=np.ones(4, np.int32)),
                config={'lr': .001}, data_schedule={'seed': 7})


class EvidenceTests(unittest.TestCase):
    def test_every_state_field_affects_digest(self):
        arrays = b.mlp_state_evidence(state())
        original = b.mlp_state_digest(arrays)
        self.assertEqual(len(arrays), 9)
        for key in arrays:
            changed = {k: v.copy() for k, v in arrays.items()}
            changed[key].flat[0] += 1
            self.assertNotEqual(original, b.mlp_state_digest(changed), key)
        changed = copy.deepcopy(state()); changed['config']['lr'] = .002
        self.assertNotEqual(original, b.mlp_state_digest(b.mlp_state_evidence(changed)))

    def test_names_dtype_and_shape_are_bound(self):
        arrays = b.mlp_state_evidence(state()); original = b.mlp_state_digest(arrays)
        for variant in ('name', 'shape', 'dtype'):
            changed = arrays.copy(); key = 'state.weight.weight1'
            if variant == 'name': changed[key + 'x'] = changed.pop(key)
            if variant == 'shape': changed[key] = changed[key].reshape(4)
            if variant == 'dtype': changed[key] = changed[key].view(np.int32)
            self.assertNotEqual(original, b.mlp_state_digest(changed))

    def test_saved_snapshot_recomputes_round_digest_without_model_call(self):
        class Model:
            reads = 0
            def state_dict(self):
                self.reads += 1
                return state()
        runner = b.OursMLP.__new__(b.OursMLP)
        runner.model = Model(); runner.np = np; runner.lane = 'mlp-train-step'
        runner.losses = [1.0, .8]; runner._state_evidence = None
        digest = runner.digest()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'state.npz'; np.savez(path, **runner.outputs())
            with np.load(path, allow_pickle=False) as saved:
                actual = {k: saved[k] for k in saved.files if k.startswith('state.')}
                self.assertEqual(digest, b.mlp_state_digest(actual))
        self.assertEqual(runner.model.reads, 1)

    def test_nonfinite_state_refused(self):
        arrays = b.mlp_state_evidence(state()); arrays['state.optimizer.m'][0] = np.nan
        with self.assertRaises(ValueError): b.mlp_state_digest(arrays)

    def test_snapshot_is_independent(self):
        original = state(); arrays = b.mlp_state_evidence(original)
        digest = b.mlp_state_digest(arrays); original['weights']['weight1'][0, 0] = 99
        self.assertEqual(digest, b.mlp_state_digest(arrays))


if __name__ == '__main__': unittest.main()
