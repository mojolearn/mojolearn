#!/usr/bin/env python3
"""Admit only complete targeted UMAP evidence; no model execution."""
import hashlib
import json
import math
from pathlib import Path
import sys
import zipfile

from nvidia_feature_finish_validate import validate_comparison


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(path):
    path = Path(path)
    required = {'vendor-venv', 'vendor-wheels', 'vendor-freeze', 'external-runtime',
                'guard-checks', 'admission-checks', 'compare-umap',
                'digits-capture', 'digits-quality'}
    for mode in ('fast', 'identical'):
        required.update(f'{job}-{mode}' for job in ('build-metrics', 'umap-api', 'retain-metrics'))
    rows = [line.split('\t') for line in (path / 'results.tsv').read_text().splitlines()]
    require(len(rows) == len(required) and {r[0] for r in rows} == required,
            'Missing or duplicated targeted jobs')
    require(all(len(r) == 2 and r[1] == '0' for r in rows), 'Failed or skipped targeted job')
    require((path / 'exit_code').read_text().strip() == '0', 'Targeted campaign failed')
    probe = json.loads((path / 'external-runtime/probe.json').read_text())
    require(probe['status'] == 'READY' and probe['returncode'] == 0
            and 'failed' not in probe['metadata'], 'External runtime readiness failed')
    ready = probe['metadata']['ready']
    require(ready['arm'] == 'external' and ready['lane'] == 'umap'
            and ready['cupy'] and ready['cuml'] and type(ready['cuda']) is int and ready['cuda'] > 0
            and ready['synchronization'] == 'cupy.cuda.runtime.deviceSynchronize',
            'Missing external CUDA/CuPy/cuML readiness')
    libraries = ready['loaded_cuda_libraries']
    require(isinstance(libraries, list)
            and all(any(Path(p).name.startswith(lib) for p in libraries)
                    for lib in ('libcuml', 'libcublas')), 'Missing loaded runtime libraries')
    comparison = validate_comparison(path, 'umap')
    require(comparison['args']['umap_rows'] == 1024 and comparison['args']['umap_epochs'] == 50,
            'Comparison did not exercise the bounded FAST GPU workload')
    inputs = comparison['inputs']
    require(inputs['file'] == 'inputs.npz' and inputs['file_sha256']
            == sha(path / 'umap-three-arm/inputs.npz'), 'Changed comparison input archive')
    require(all(record['inputs'] == inputs['raw_sha256'] for record in comparison['metadata'].values()),
            'Comparison workers did not use retained inputs')
    root = path / 'digits-quality'
    result = json.loads((root / 'results.json').read_text())
    pin = json.loads((path / 'digits-pin/dataset.json').read_text())
    require(result['schema'] == 'umap.digits.real-data-quality.v1' and result['status'] == 'PASS',
            'Real-data quality is not admitted')
    require(result['policy'] == dict(k=10, minimum_trustworthiness=.85,
                                    maximum_trustworthiness_loss_vs_reference=.08,
                                    minimum_retention=.15, minimum_control_trustworthiness_margin=.15),
            'Neighborhood policy differs from declared admission')
    dataset = result['dataset']
    require(dataset['pin_verified'] is True and dataset['data_sha256'] == dataset['expected_data_sha256']
            == pin['data_sha256'], 'Missing prior dataset pin')
    for key in ('train', 'query', 'train_indices', 'query_indices'):
        require(dataset[key] == pin[key], 'Dataset/split differs from prior capture')
    require(dataset['train']['shape'] == [1024, 64] and dataset['query']['shape'] == [256, 64],
            'Wrong real-data workload dimensions')
    require(pin['inputs_file_sha256'] == sha(path / 'digits-pin/inputs.npz'), 'Changed dataset pin archive')
    require(dataset['inputs_file_sha256'] == sha(root / 'inputs.npz'), 'Changed retained dataset')
    require(set(result['arms']) == {'reference', 'fast', 'identical'}, 'Missing real-data arm')
    require(set(result['quality']) == set(result['arms']), 'Missing arm quality')
    for arm, record in result['arms'].items():
        require(record['status'] == 'PASS', 'Incomplete real-data arm')
        require(record['artifact_sha256'] == sha(root / f'{arm}-embeddings.npz'), 'Changed embedding file')
        require(record['fit']['shape'] == [1024, 2] and record['transform']['shape'] == [256, 2],
                'Wrong retained embedding dimensions')
        if arm != 'reference':
            require(record['native_vendor'] == 'cuda' and record['numeric_mode_witness']
                    == {'fast': 0, 'identical': 1}[arm], 'Missing native CUDA/mode witness')
            binding_sha = sha(path / f'bindings/metrics-{arm}.so')
            require(record['binding_sha256'] == binding_sha
                    == comparison['metadata'][arm]['binding_sha256'], 'Binding provenance differs')
        require(set(result['quality'][arm]) == {'fit', 'transform'}, 'Missing fit/transform quality')
        for stage, quality in result['quality'][arm].items():
            reference = result['quality']['reference'][stage]
            require(quality['passed'] is True
                    and all(math.isfinite(quality[k]) for k in ('trustworthiness', 'retention', 'control_trustworthiness'))
                    and quality['trustworthiness'] >= .85 and quality['retention'] >= .15
                    and quality['trustworthiness'] >= reference['trustworthiness'] - .08
                    and quality['trustworthiness'] - quality['control_trustworthiness'] >= .15,
                    'Real-data neighborhood admission failed')
    repeat = result['arms']['identical']['repeatability']
    require(all(repeat[key] is True for key in ('fit_raw_bytes_equal', 'transform_raw_bytes_equal',
                                               'frozen_model_unchanged', 'inputs_unchanged')),
            'Missing IDENTICAL repeatability')
    require(repeat['artifact_sha256'] == sha(root / 'identical-repeat-embeddings.npz'), 'Changed repeated embeddings')
    with zipfile.ZipFile(root / 'identical-embeddings.npz') as first, \
            zipfile.ZipFile(root / 'identical-repeat-embeddings.npz') as second:
        for array in ('fit.npy', 'transform.npy'):
            require(first.read(array) == second.read(array), 'IDENTICAL retained embedding bytes differ')
    return {'status': 'PASSED', 'scope': 'NVIDIA UMAP API fixtures, one CUDA timing comparator and pinned digits quality; no cross-vendor or universal certificate'}


if __name__ == '__main__':
    print(json.dumps(validate(sys.argv[1]), indent=2))
