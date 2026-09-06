"""Small synthetic release-admission tests; main agent alone executes these."""
import base64
import copy
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import check_linux_release_qualification as gate


def write_json(path, value):
    path.write_text(json.dumps(value))


def make_wheel(root, vendors=('hip', 'cuda'), omit=None, corrupt=False):
    root.mkdir(parents=True, exist_ok=True)
    wrapper = root / 'python/mojolearn/__init__.py'
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text('# current wrapper\n')
    files = {'mojolearn/__init__.py': wrapper.read_bytes()}
    for vendor in vendors:
        arch = 'gfx942' if vendor == 'hip' else 'sm_89'
        for mode in gate.surface.MODES:
            prefix = f'mojolearn/{vendor}/{arch}/' + (mode + '/' if mode != 'fast' else '')
            for binding in gate.surface.BINDINGS:
                name = prefix + binding + '.so'
                if name != omit:
                    files[name] = name.encode()
    record = 'mojolearn-0.6.0.dist-info/RECORD'
    rows = io.StringIO()
    writer = csv.writer(rows)
    for name, data in sorted(files.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b'=').decode()
        writer.writerow([name, 'sha256=' + digest, len(data)])
    writer.writerow([record, '', ''])
    files[record] = rows.getvalue().encode()
    if corrupt:
        files['mojolearn/__init__.py'] += b'# not in RECORD\n'
    wheel = root / 'mojolearn-0.6.0-py3-none-manylinux_2_28_x86_64.whl'
    with zipfile.ZipFile(wheel, 'w') as archive:
        for name, data in files.items():
            archive.writestr(name, data)
    return wheel


class ReleaseAdmissionTests(unittest.TestCase):
    def test_complete_dual_wheel_record_and_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = make_wheel(root)
            extensions, sets = gate.inspect_wheel(wheel, root)
            self.assertEqual(len(extensions), 90)
            self.assertEqual(len(sets), 6)
            self.assertEqual(set(sets.values()), {15})

    def test_vendor_candidates_missing_modes_and_corrupt_record_refuse(self):
        cases = [dict(vendors=('hip',)), dict(vendors=('cuda',)),
                 dict(omit='mojolearn/hip/gfx942/identical/_mojolearn_gbdt.so'),
                 dict(corrupt=True)]
        for options in cases:
            with self.subTest(options=options), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                wheel = make_wheel(root, **options)
                with self.assertRaises(ValueError):
                    gate.inspect_wheel(wheel, root)

    def test_native_inventory_catches_added_source_but_excludes_release_tools(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'a.mojo').write_text('native')
            (root / 'tools').mkdir()
            (root / 'tools/linux_surface_qualification.sh').write_text('qualification')
            first = gate.native_inventory(root)
            (root / 'tools/check_linux_release_qualification.py').write_text('admission only')
            (root / 'archive').mkdir()
            (root / 'archive/old.mojo').write_text('old')
            self.assertEqual(gate.native_inventory(root), first)
            (root / 'new.mojo').write_text('new native code')
            self.assertNotEqual(gate.native_inventory(root), first)

    def vendor_fixture(self, root):
        wheel = make_wheel(root)
        extensions, sets = gate.inspect_wheel(wheel, root)
        inventory = gate.native_inventory(root)
        wheel_sha = gate.digest_file(wheel)
        source_sha = gate.inventory_digest(inventory)
        out = root / 'qualification/hip'
        out.mkdir(parents=True)
        proof = {'schema': 'mojolearn.linux.build-provenance.v1', 'complete': True,
                 'action': 'build', 'build_exit': 0, 'source_commit': 'a' * 40,
                 'source_inventory': inventory, 'source_sha256': source_sha,
                 'extensions': {'mojolearn/' + p: h for p, h in extensions.items()
                                if p.startswith('hip/')}}
        write_json(out / 'build-provenance.json', proof)
        audit = {'sha256': wheel_sha, 'qualification_vendor': 'hip',
                 'advertised_vendors': ['cuda', 'hip'], 'extension_hashes': extensions,
                 'sets': sets, 'source_sha256': source_sha,
                 'build_provenance_sha256': gate.digest_file(out / 'build-provenance.json')}
        write_json(out / 'wheel-audit.json', audit)
        records = {}
        package = '/vanished/qualification/venv/lib/python3.11/site-packages/mojolearn'
        for surface in gate.surface.SURFACES:
            for mode, code in gate.surface.MODES.items():
                name = surface + '-' + mode
                prefix = 'hip/gfx942/' + (mode + '/' if mode != 'fast' else '')
                bindings = {}
                for binding in gate.surface.BINDINGS:
                    member = prefix + binding + '.so'
                    bindings[binding] = {'path': package + '/' + member,
                                         'sha256': extensions[member], 'mode_code': code}
                installed = {'vendor': 'hip', 'mode': mode, 'wheel_sha256': wheel_sha,
                             'package': package + '/__init__.py', 'installed_bindings': bindings}
                path = out / (name + '.installed.json')
                write_json(path, installed)
                records[name] = gate.digest_file(path)
                if surface == 'umap-quality':
                    write_json(out / (name + '.json'), {})
        qualification = {'vendor': 'hip', 'wheel_sha256': wheel_sha,
                         'source_sha256': source_sha, 'installed_records': records}
        return out, qualification, wheel_sha, inventory, extensions, sets

    def test_all_24_installed_records_and_three_quality_modes_rechecked(self):
        with tempfile.TemporaryDirectory() as directory:
            out, qualification, *args = self.vendor_fixture(Path(directory))
            with patch.object(gate.surface, 'retained', return_value=(qualification, {})), \
                    patch.object(gate.surface, 'check_quality') as quality:
                gate.check_vendor(out, 'hip', *args)
                self.assertEqual(quality.call_count, 3)

    def test_stale_hash_inventory_and_missing_job_refuse(self):
        for defect in ('wheel', 'native', 'missing_job', 'build_proof'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as directory:
                out, qualification, wheel_sha, inventory, extensions, sets = self.vendor_fixture(Path(directory))
                if defect == 'wheel':
                    qualification['wheel_sha256'] = 'b' * 64
                elif defect == 'native':
                    inventory = inventory + [['new.mojo', 'b' * 64]]
                elif defect == 'missing_job':
                    qualification['installed_records'].pop('ordered-rmse-fast')
                else:
                    (out / 'build-provenance.json').write_text('{}')
                with patch.object(gate.surface, 'retained', return_value=(qualification, {})), \
                        patch.object(gate.surface, 'check_quality'), self.assertRaises(ValueError):
                    gate.check_vendor(out, 'hip', wheel_sha, inventory, extensions, sets)

    def test_nonidentical_job_binding_cannot_hide_behind_success_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            out, qualification, *args = self.vendor_fixture(Path(directory))
            path = out / 'ordered-rmse-fast.installed.json'
            installed = json.loads(path.read_text())
            installed['installed_bindings']['_mojolearn_gbdt']['mode_code'] = 1
            write_json(path, installed)
            qualification['installed_records']['ordered-rmse-fast'] = gate.digest_file(path)
            with patch.object(gate.surface, 'retained', return_value=(qualification, {})), \
                    patch.object(gate.surface, 'check_quality'), self.assertRaises(ValueError):
                gate.check_vendor(out, 'hip', *args)

    def test_comparators_are_recomputed_and_failure_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = make_wheel(root)
            for fail in (False, True):
                with self.subTest(fail=fail), patch.object(gate, 'check_vendor'), \
                        patch.object(gate.surface, 'compare', return_value={'status': 'PASSED'}) as umap, \
                        patch.object(gate.compare_ordered_python, 'compare', return_value={
                            'status': 'FAILED' if fail else 'PASSED'}) as ordered:
                    for vendor in ('hip', 'cuda'):
                        out = root / 'qualification' / vendor
                        out.mkdir(parents=True, exist_ok=True)
                        write_json(out / 'qualification.json', {})
                    if fail:
                        with self.assertRaises(ValueError):
                            gate.check(wheel, root / 'qualification', root)
                    else:
                        self.assertEqual(gate.check(wheel, root / 'qualification', root)['jobs_per_vendor'], 24)
                    umap.assert_called_once()
                    ordered.assert_called_once()


if __name__ == '__main__':
    unittest.main()
