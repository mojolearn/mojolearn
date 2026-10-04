#!/usr/bin/env python3
"""Synthetic release metadata exercises runtime policy; no GPU claim is made."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import test_plugin_loader as fixtures

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location('ptx_admission_under_test', ROOT / 'python/mojolearn/ptx_admission.py')
admission = importlib.util.module_from_spec(spec)
spec.loader.exec_module(admission)
CONFIG = dict(device_name='Synthetic A100', compute_capability=[8, 0], driver_version='580.1.2', cuda_driver_version=13000)


def record(manifest_hash='b' * 64):
    return dict(schema=admission.SCHEMA, qualified=True, numeric_mode='identical', source_commit='a' * 40,
                manifest_sha256=manifest_hash, coverage_contract=admission.COVERAGE_CONTRACT,
                coverage=dict(inventory_sha256='c' * 64, harness_sha256='d' * 64,
                    shared=dict(lanes=['synthetic-lane'], fixtures=list(admission.SHARED_FIXTURES),
                                parts=list(admission.SHARED_PARTS), vendors=['cuda', 'hip', 'metal'], comparison_sha256='e' * 64),
                    nvidia=dict(lanes=['synthetic-lane'], fixtures=list(admission.NVIDIA_FIXTURES),
                                parts=list(admission.NVIDIA_PARTS), comparison_sha256='f' * 64)), configurations=[copy.deepcopy(CONFIG)])


class AdmissionRecord(unittest.TestCase):
    def test_exact_record_and_runtime(self):
        doc = record()
        self.assertIs(admission.validate_admission(doc, source_commit='a' * 40, manifest_sha256='b' * 64,
                                                  configuration=CONFIG), doc)

    def test_incomplete_tampered_and_unknown_records_refuse(self):
        mutations = [lambda d: d.update(qualified=False), lambda d: d.update(source_commit='0' * 40),
                     lambda d: d.update(manifest_sha256='0' * 64), lambda d: d.update(numeric_mode='fast'),
                     lambda d: d.update(configurations=[]), lambda d: d['configurations'].append(copy.deepcopy(CONFIG)),
                     lambda d: d['configurations'][0].update(driver_version='unknown'),
                     lambda d: d['coverage']['shared'].update(vendors=['cuda', 'metal']),
                     lambda d: d['coverage']['shared'].update(lanes=[]),
                     lambda d: d['coverage']['shared'].update(lanes=['synthetic-lane', 'synthetic-lane']),
                     lambda d: d['coverage']['shared'].update(parts=['train']),
                     lambda d: d['coverage']['shared'].update(fixtures=list(admission.NVIDIA_FIXTURES)),
                     lambda d: d['coverage']['nvidia'].update(fixtures=list(admission.SHARED_FIXTURES)),
                     lambda d: d['coverage']['nvidia'].update(lanes=['other']),
                     lambda d: d['coverage'].update(inventory_sha256='')]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                doc = record()
                mutate(doc)
                with self.assertRaises(ValueError):
                    admission.validate_admission(doc, source_commit='a' * 40, manifest_sha256='b' * 64)
        for key, value in [('driver_version', '581.1.2'), ('device_name', 'Other GPU'),
                           ('compute_capability', [9, 0]), ('cuda_driver_version', 13010)]:
            with self.subTest(configuration=key), self.assertRaises(ValueError):
                admission.validate_admission(record(), source_commit='a' * 40, manifest_sha256='b' * 64,
                                              configuration={**CONFIG, key: value})


class NativeFirst(unittest.TestCase):
    setUp = fixtures.PluginLoader.setUp
    dist_info = fixtures.PluginLoader.dist_info
    split_core = fixtures.PluginLoader.split_core
    plugin = fixtures.PluginLoader.plugin
    sets = fixtures.PluginLoader.sets
    baseline = fixtures.ArchitecturePayloadLoader.baseline

    def bundled(self):
        root, binary = self.baseline()
        os.environ.pop('MOJOLEARN_CUDA_PATH')
        os.environ.pop('MOJOLEARN_EXPERIMENTAL_PTX')
        self.plugin('cuda', ['sm_89', 'sm_90a'])
        manifest_hash = hashlib.sha256((root / self.B.gpu_plugins.BASELINE_MANIFEST).read_bytes()).hexdigest()
        raw = json.dumps(record(manifest_hash)).encode()
        (root / admission.ADMISSION_FILE).write_bytes(raw)
        bundle = dict(manifest_sha256=manifest_hash, admission_sha256=hashlib.sha256(raw).hexdigest())
        gp = self.B.gpu_plugins
        marker_path = self.site / f'mojolearn_nvidia-{self.version}.dist-info' / gp.PLUGIN_MARKER
        marker_path.write_text(json.dumps(gp.plugin_marker('cuda', self.version, ['sm_89', 'sm_90a'], bundled_ptx=bundle)))
        self.B._probe_box = lambda: fixtures.probe(self.B, cuda=True)
        self.B._device_arch = lambda vendor: ('sm_80', 'synthetic driver')
        self.B._ptx_runtime_configuration = lambda: copy.deepcopy(CONFIG)
        return root, binary

    def test_native_priority_never_reads_admission(self):
        root, _ = self.bundled()
        (root / admission.ADMISSION_FILE).unlink()
        self.B._device_arch = lambda vendor: ('sm_89', 'synthetic driver')
        self.assertTrue(self.B._layout()[1].endswith('cuda_native/sm_89'))
        self.assertIsNone(self.B.baseline_selection_receipt())

    def test_no_native_selects_exact_admitted_bytes(self):
        root, binary = self.bundled()
        self.assertEqual(self.B._layout(), ('vendor', str(root)))
        self.assertEqual(self.B.baseline_selection_receipt()['requested'], 'native-first')
        self.assertTrue(self.B.baseline_selection_receipt()['identical_qualified'])
        self.B._record_baseline_load(str(binary))
        self.assertEqual(len(self.B.baseline_selection_receipt()['loaded_files']), 1)
        binary.write_bytes(b'tampered after validation')
        with self.assertRaises(self.B.GpuPluginError):
            self.B._record_baseline_load(str(binary))

    def test_unqualified_missing_tampered_and_unknown_fail_closed(self):
        root, binary = self.bundled()
        admission_path = root / admission.ADMISSION_FILE
        old = admission_path.read_bytes()
        for mutation in ('missing', 'changed', 'unknown-driver', 'payload-changed', 'fast'):
            with self.subTest(mutation=mutation):
                self.B._LAYOUT = None
                if mutation == 'missing':
                    admission_path.unlink()
                elif mutation == 'changed':
                    admission_path.write_bytes(old + b' ')
                elif mutation == 'unknown-driver':
                    self.B._ptx_runtime_configuration = lambda: {**CONFIG, 'driver_version': '999.1'}
                elif mutation == 'payload-changed':
                    binary.write_bytes(b'changed')
                with patch.dict(os.environ, {'MOJOLEARN_NUMERIC_MODE': 'fast' if mutation == 'fast' else 'identical'}):
                    with patch.object(self.B, 'host_binding_built', return_value=True):
                        with patch.object(self.B, '_select_cpu_only') as cpu:
                            with self.assertRaises(self.B.GpuPluginError):
                                self.B.select()
                            cpu.assert_not_called()
                admission_path.write_bytes(old)
                self.B._ptx_runtime_configuration = lambda: copy.deepcopy(CONFIG)
                binary.write_bytes(b'fake embedded PTX, not executable')

    def test_native_errors_do_not_trigger_baseline(self):
        self.bundled()
        with patch.object(self.B, '_pick_arch', side_effect=ImportError('native probe broken')):
            with patch.object(self.B, '_admitted_baseline_base') as fallback:
                with self.assertRaisesRegex(ImportError, 'native probe broken'):
                    self.B._layout()
                fallback.assert_not_called()

    def test_no_bundle_is_an_explicit_gpu_refusal(self):
        self.split_core()
        self.plugin('cuda', ['sm_89', 'sm_90a'])
        self.B._probe_box = lambda: fixtures.probe(self.B, cuda=True)
        self.B._device_arch = lambda vendor: ('sm_80', 'synthetic driver')
        with self.assertRaisesRegex(self.B.GpuPluginError, 'no admitted PTX'):
            self.B._layout()


class RuntimeHardware(unittest.TestCase):
    def query(self, *, count=1, output=None):
        from types import SimpleNamespace
        def set_int(pointer, value):
            pointer._obj.value = value
            return 0
        def set_uuid(buffer, device):
            buffer.raw = bytes(range(16))
            return 0
        library = SimpleNamespace(cuInit=lambda _: 0, cuDeviceGetCount=lambda p: set_int(p, count),
            cuDeviceGet=lambda p, ordinal: set_int(p, ordinal), cuDeviceGetUuid_v2=set_uuid,
            cuDriverGetVersion=lambda p: set_int(p, 13000))
        B = fixtures.fresh_backend()
        expected_uuid = 'GPU-00010203-0405-0607-0809-0a0b0c0d0e0f'
        if output is None:
            output = expected_uuid + ', Synthetic A100, 8.0, 580.1.2\n'
        with patch('ctypes.CDLL', return_value=library):
            with patch('subprocess.check_output', return_value=output) as query:
                result = B._ptx_runtime_configuration()
        self.assertIn('--id=' + expected_uuid, query.call_args.args[0])
        return result

    def test_correlates_cuda_uuid_and_exact_driver(self):
        self.assertEqual(self.query(), CONFIG)

    def test_ambiguous_visible_devices_and_wrong_uuid_refuse(self):
        with self.assertRaisesRegex(ValueError, 'one visible CUDA'):
            self.query(count=2)
        with self.assertRaisesRegex(ValueError, 'witness differ'):
            self.query(output='GPU-other, Synthetic A100, 8.0, 580.1.2\n')
        with self.assertRaisesRegex(ValueError, 'ambiguous'):
            self.query(output='first\nsecond\n')
