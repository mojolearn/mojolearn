"""Root-only retained-evidence sabotage tests; no GPU or measurement."""
import json
from pathlib import Path
import tempfile
import unittest

from nvidia_feature_finish_validate import validate


class FeatureEvidenceTests(unittest.TestCase):
    def test_missing_duplicate_failed_jobs_and_missing_quality_refuse(self):
        names = ['vendor-venv', 'vendor-wheels', 'vendor-freeze', 'vendor-cuda',
                 'mamba-corpus', 'mamba23-backward', 'nan-forbidden',
                 'mamba2-native-dump', 'mamba3-native-dump',
                 'umap-finite-params', 'umap-controls-fast', 'umap-controls-identical',
                 'umap-identity', 'compare-gbdt', 'compare-umap']
        for mode in ('fast', 'identical'):
            names += [f'build-{binding}-{mode}' for binding in ('gbdt', 'metrics', 'mamba')]
            names += [f'{surface}-{mode}' for surface in ('gbdt-boundary', 'umap-api', 'mamba-api')]
        good = ''.join(f'{name}\t0\n' for name in names)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'results.tsv').write_text(good)
            (root / 'exit_code').write_text('0\n')
            for lane in ('gbdt', 'umap'):
                target = root / f'{lane}-three-arm'
                target.mkdir()
                metadata = {arm: {'inputs': ['a' * 64], 'mode': arm,
                                  'binding_sha256': 'b' * 64}
                            for arm in ('fast', 'identical', 'external')}
                records = [{'arm': arm, 'round': r, 'ms': 1.0, 'warmup': r == 0,
                            'hashes': ['c' * 64]} for arm in metadata for r in range(8)]
                (target / 'results.json').write_text(json.dumps({
                    'status': 'PASSED', 'metadata': metadata, 'args': {'rounds': 7},
                    'records': records,
                    'accuracy': [{'arm': r['arm'], 'round': r['round'], 'passed': True} for r in records]}))
            self.assertEqual(validate(root)['status'], 'PASSED')
            for bad in (good.split('\n', 1)[1], good + 'compare-gbdt\t0\n',
                        good.replace('mamba23-backward\t0', 'mamba23-backward\t124')):
                (root / 'results.tsv').write_text(bad)
                with self.assertRaises(ValueError):
                    validate(root)
            (root / 'results.tsv').write_text(good)
            target = root / 'umap-three-arm/results.json'
            result = json.loads(target.read_text())
            result['accuracy'] = []
            target.write_text(json.dumps(result))
            with self.assertRaises(ValueError):
                validate(root)

    # The main fixture also demonstrates that a status string alone is not
    # accepted: every mode, round, byte witness and quality arm is required.


if __name__ == '__main__':
    unittest.main()
