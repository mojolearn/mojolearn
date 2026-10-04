# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Fail-closed runtime validation of offline PTX IDENTICAL admission records.

This validates a release decision, not its underlying experimental evidence.
Only the offline evidence checker may establish complete applicable coverage.
No supported configuration or admission record is shipped by this module.
"""
import re

ADMISSION_FILE = 'PTX_IDENTITY_ADMISSION.json'
SCHEMA = 'mojolearn.ptx-identity-admission.v1'
COVERAGE_CONTRACT = 'mojolearn.cross-vendor-identical.v1'
SHARED_FIXTURES = ('base', 'denormal', 'odd')
NVIDIA_FIXTURES = ('base', 'ties', 'hashed', 'wide', 'denormal', 'denormal_ftz', 'dupes', 'odd', 'negative')
SHARED_PARTS = ('train', 'infer', 'model', 'batchgrad', 'batchscale', 'ragged', 'stepfull')
NVIDIA_PARTS = SHARED_PARTS + ('batch', 'rlpair')


def _require(condition, message):
    if not condition:
        raise ValueError('PTX identity admission: ' + message)


def _digest(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def _names(value):
    return (isinstance(value, list) and bool(value)
            and all(isinstance(v, str) and v.strip() == v and bool(v) for v in value)  # glue: validate metadata identifier strings
            and len(value) == len(set(value)))


def _configuration(value):
    _require(isinstance(value, dict) and set(value) == {
        'device_name', 'compute_capability', 'driver_version', 'cuda_driver_version'},
        'configuration fields are missing or unknown')
    cap = value['compute_capability']
    _require(isinstance(cap, list) and len(cap) == 2
             and all(type(v) is int and v >= 0 for v in cap) and cap >= [8, 0],  # glue: validate architecture metadata integers
             'unsupported compute capability')
    _require(isinstance(value['device_name'], str) and bool(value['device_name'].strip()), 'missing device name')
    _require(isinstance(value['driver_version'], str)
             and re.fullmatch(r'[0-9]+(?:\.[0-9]+)+', value['driver_version']) is not None,
             'unknown driver version')
    _require(type(value['cuda_driver_version']) is int and value['cuda_driver_version'] > 0,
             'missing CUDA driver API version')
    return (value['device_name'], tuple(cap), value['driver_version'], value['cuda_driver_version'])


def validate_admission(doc, *, source_commit, manifest_sha256, configuration=None):
    """Validate immutable release metadata; optionally match this exact runtime.

    Source and manifest come from actual installed files. Packaging supplies no
    configuration, while the loader must supply independently queried hardware.
    """
    _require(isinstance(doc, dict), 'record must be an object')
    _require(doc.get('schema') == SCHEMA and doc.get('qualified') is True
             and doc.get('numeric_mode') == 'identical'
             and doc.get('coverage_contract') == COVERAGE_CONTRACT, 'unqualified record or wrong contract')
    _require(isinstance(source_commit, str) and re.fullmatch('[0-9a-f]{40}', source_commit) is not None
             and doc.get('source_commit') == source_commit, 'source mismatch')
    _require(_digest(manifest_sha256) and doc.get('manifest_sha256') == manifest_sha256, 'manifest mismatch')
    coverage = doc.get('coverage')
    _require(isinstance(coverage, dict) and _digest(coverage.get('inventory_sha256'))
             and _digest(coverage.get('harness_sha256')), 'missing coverage provenance')
    shared, native = coverage.get('shared'), coverage.get('nvidia')
    for name, row, fixtures, parts in (('shared', shared, SHARED_FIXTURES, SHARED_PARTS),  # glue: validate two release coverage metadata records
                                      ('nvidia', native, NVIDIA_FIXTURES, NVIDIA_PARTS)):
        _require(isinstance(row, dict) and _names(row.get('lanes')), name + ' lanes missing or duplicated')
        _require(_names(row.get('fixtures')) and set(row['fixtures']) == set(fixtures), name + ' fixtures incomplete')
        _require(_names(row.get('parts')) and set(row['parts']) == set(parts), name + ' parts incomplete')
        _require(_digest(row.get('comparison_sha256')), name + ' comparison missing')
    _require(set(shared['lanes']) == set(native['lanes']), 'shared and NVIDIA lane scopes differ')
    # Undeclared parts are named exclusions of the NVIDIA scope, never coverage.
    # The shared scope admits none.
    _require('undeclared_exclusions' not in shared, 'shared scope cannot exclude undeclared parts')
    excluded = native.get('undeclared_exclusions', [])
    _require(isinstance(excluded, list), 'undeclared exclusions malformed')
    seen = set()
    for row in excluded:  # glue: validate excluded lane/part metadata records
        _require(isinstance(row, dict) and set(row) == {'lane', 'part'}
                 and row['lane'] in native['lanes'] and row['part'] in native['parts']
                 and (row['lane'], row['part']) not in seen, 'undeclared exclusion unknown or duplicated')
        seen.add((row['lane'], row['part']))
    _require(_names(shared.get('vendors')) and set(shared['vendors']) == {'cuda', 'hip', 'metal'},
             'cross-vendor references incomplete')
    configs = doc.get('configurations')
    _require(isinstance(configs, list) and bool(configs), 'no admitted configurations')
    keys = [_configuration(row) for row in configs]  # glue: validate release hardware metadata entries
    _require(len(keys) == len(set(keys)), 'duplicate configurations')
    if configuration is not None:
        _require(_configuration(configuration) in keys, 'this device/driver configuration is not admitted')
    return doc
