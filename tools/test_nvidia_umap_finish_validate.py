"""Root-only evidence sabotage checks. Stdlib fixtures; no NumPy/GPU execution."""
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest
import zipfile

from nvidia_umap_finish_validate import validate


POLICY = dict(k=10, minimum_trustworthiness=0.85,
              maximum_trustworthiness_loss_vs_reference=0.08,
              minimum_retention=0.15, minimum_control_trustworthiness_margin=0.15)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def npy(shape, word=0):
    """A real little-endian float32 NPY, assembled without NumPy."""
    header = repr({'descr': '<f4', 'fortran_order': False, 'shape': tuple(shape)}).encode()
    header += b' ' * ((-(10 + len(header) + 1)) % 64) + b'\n'
    count = 1
    for size in shape:
        count *= size
    return b'\x93NUMPY\x01\x00' + struct.pack('<H', len(header)) + header + struct.pack('<I', word) * count


def archive(path, arrays):
    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, 'w', zipfile.ZIP_DEFLATED) as output:
        for name, value in arrays.items():
            output.writestr(name + '.npy', value)


def witness(shape):
    return {'shape': list(shape), 'dtype': '<f4', 'raw_c_order_sha256': 'a' * 64}


def fixture(root):
    jobs = ['vendor-venv', 'vendor-wheels', 'vendor-freeze', 'external-runtime',
            'guard-checks', 'admission-checks', 'compare-umap', 'digits-capture', 'digits-quality']
    for mode in ('fast', 'identical'):
        jobs += [f'{name}-{mode}' for name in ('build-metrics', 'umap-api', 'retain-metrics')]
    (root / 'results.tsv').write_text(''.join(f'{name}\t0\n' for name in jobs))
    (root / 'exit_code').write_text('0\n')
    runtime = {'arm': 'external', 'lane': 'umap', 'cuda': 12090,
               'cupy': '14.2.0', 'cuml': '26.8.0', 'device': 'fixture-NVIDIA',
               'synchronization': 'cupy.cuda.runtime.deviceSynchronize',
               'loaded_cuda_libraries': ['/fixture/libcuda.so.1', '/fixture/libcuml.so', '/fixture/libcublas.so']}
    save(root / 'external-runtime/probe.json',
         {'status': 'READY', 'returncode': 0, 'metadata': {'ready': runtime}})
    quality_root = root / 'digits-quality'
    archive(quality_root / 'inputs.npz', {'train': npy((1024, 64)), 'query': npy((256, 64))})
    dataset = {'pin_verified': True, 'data_sha256': 'd' * 64,
               'expected_data_sha256': 'd' * 64,
               'inputs_file_sha256': sha(quality_root / 'inputs.npz'),
               'train': witness((1024, 64)), 'query': witness((256, 64)),
               'train_indices': witness((1024,)), 'query_indices': witness((256,))}
    (root / 'digits-pin').mkdir()
    (root / 'digits-pin/inputs.npz').write_bytes((quality_root / 'inputs.npz').read_bytes())
    save(root / 'digits-pin/dataset.json', dataset)
    arms, quality, metadata = {}, {}, {}
    for arm in ('reference', 'fast', 'identical'):
        archive(quality_root / f'{arm}-embeddings.npz',
                {'fit': npy((1024, 2)), 'transform': npy((256, 2))})
        record = {'status': 'PASS', 'artifact_sha256': sha(quality_root / f'{arm}-embeddings.npz'),
                  'fit': witness((1024, 2)), 'transform': witness((256, 2))}
        if arm != 'reference':
            binding = root / f'bindings/metrics-{arm}.so'
            binding.parent.mkdir(exist_ok=True)
            binding.write_bytes(('synthetic retained binding ' + arm).encode())
            record.update(native_vendor='cuda', numeric_mode_witness={'fast': 0, 'identical': 1}[arm],
                          binding_sha256=sha(binding))
            metadata[arm] = {'mode': arm, 'native_vendor': 'cuda', 'binding_sha256': sha(binding),
                             'inputs': ['a' * 64], 'shapes': [[1024, 3], [1024, 3]]}
        arms[arm] = record
        quality[arm] = {stage: {'passed': True, 'trustworthiness': .95, 'retention': .5,
                                'control_trustworthiness': .5} for stage in ('fit', 'transform')}
    archive(quality_root / 'identical-repeat-embeddings.npz',
            {'fit': npy((1024, 2)), 'transform': npy((256, 2))})
    arms['identical']['repeatability'] = {
        'fit_raw_bytes_equal': True, 'transform_raw_bytes_equal': True,
        'frozen_model_unchanged': True, 'inputs_unchanged': True,
        'artifact_sha256': sha(quality_root / 'identical-repeat-embeddings.npz')}
    save(quality_root / 'results.json', {'schema': 'umap.digits.real-data-quality.v1',
         'status': 'PASS', 'policy': POLICY, 'dataset': dataset, 'arms': arms, 'quality': quality})
    metadata['external'] = dict(runtime, inputs=['a' * 64], shapes=[[1024, 3], [1024, 3]])
    records = [{'arm': arm, 'round': index, 'ms': 1., 'warmup': index == 0, 'hashes': ['c' * 64]}
               for arm in metadata for index in range(8)]
    comparison_inputs = root / 'umap-three-arm/inputs.npz'
    archive(comparison_inputs, {'a': npy((1024, 3)), 'b': npy((1024, 3))})
    save(root / 'umap-three-arm/results.json', {
        'status': 'PASSED', 'metadata': metadata,
        'inputs': {'file': 'inputs.npz', 'file_sha256': sha(comparison_inputs),
                   'raw_sha256': ['a' * 64]},
        'args': {'rounds': 7, 'umap_rows': 1024, 'umap_epochs': 50}, 'records': records,
        'accuracy': [{'arm': row['arm'], 'round': row['round'], 'passed': True} for row in records]})


class UmapEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        fixture(self.root)

    def change(self, relative, mutate):
        path = self.root / relative
        record = json.loads(path.read_text())
        mutate(record)
        save(path, record)

    def refused(self):
        with self.assertRaises((ValueError, OSError, KeyError)):
            validate(self.root)

    def test_complete_fixture_admitted(self):
        self.assertEqual(validate(self.root)['status'], 'PASSED')

    def test_missing_external_probe_refused(self):
        (self.root / 'external-runtime/probe.json').unlink()
        self.refused()

    def test_refused_external_probe_cannot_be_hidden_by_successful_job(self):
        self.change('external-runtime/probe.json', lambda r: r.update(status='REFUSED', returncode=1))
        self.refused()

    def test_wrong_probe_lane_refused(self):
        self.change('external-runtime/probe.json', lambda r: r['metadata']['ready'].update(lane='knn'))
        self.refused()

    def test_missing_runtime_map_refused(self):
        self.change('external-runtime/probe.json', lambda r: r['metadata']['ready'].update(loaded_cuda_libraries=[]))
        self.refused()

    def test_changed_neighborhood_k_refused(self):
        self.change('digits-quality/results.json', lambda r: r['policy'].update(k=5))
        self.refused()

    def test_small_quality_shape_refused_even_when_pin_agrees(self):
        for path in ('digits-quality/results.json', 'digits-pin/dataset.json'):
            self.change(path, lambda r: (r.get('dataset', r))['train'].update(shape=[64, 64]))
        self.refused()

    def test_small_timing_rows_refused(self):
        self.change('umap-three-arm/results.json', lambda r: r['args'].update(umap_rows=64))
        self.refused()

    def test_changed_timing_epochs_refused(self):
        self.change('umap-three-arm/results.json', lambda r: r['args'].update(umap_epochs=10))
        self.refused()

    def test_mismatched_binary_refused(self):
        self.change('digits-quality/results.json', lambda r: r['arms']['fast'].update(binding_sha256='b' * 64))
        self.refused()

    def test_wrong_native_vendor_refused(self):
        self.change('digits-quality/results.json', lambda r: r['arms']['identical'].update(native_vendor='hip'))
        self.refused()

    def test_tampered_repeat_refused_even_with_updated_container_hash(self):
        path = self.root / 'digits-quality/identical-repeat-embeddings.npz'
        archive(path, {'fit': npy((1024, 2), word=1), 'transform': npy((256, 2))})
        self.change('digits-quality/results.json',
                    lambda r: r['arms']['identical']['repeatability'].update(artifact_sha256=sha(path)))
        self.refused()

    def test_tampered_comparison_input_archive_refused(self):
        archive(self.root / 'umap-three-arm/inputs.npz',
                {'a': npy((1024, 3), word=1), 'b': npy((1024, 3))})
        self.refused()

    def test_worker_input_hash_must_match_retained_manifest(self):
        self.change('umap-three-arm/results.json', lambda r: r['inputs'].update(raw_sha256=['b' * 64]))
        self.refused()

    def test_failed_job_refused(self):
        path = self.root / 'results.tsv'
        path.write_text(path.read_text().replace('digits-quality\t0', 'digits-quality\t124'))
        self.refused()

    def test_missing_job_refused(self):
        path = self.root / 'results.tsv'
        path.write_text(path.read_text().replace('external-runtime\t0\n', ''))
        self.refused()


if __name__ == '__main__':
    unittest.main()
