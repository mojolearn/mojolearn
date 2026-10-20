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


def merged_index_bytes(files):
    """A set directory's _mojolearn_merged.json for {'<stem>.so': [members]}."""
    return json.dumps({'format': 1, 'files': files}, sort_keys=True).encode()


def make_wheel(root, vendors=('hip', 'cuda'), omit=None, corrupt=False, merged=False, extra=None, index_edit=None):
    """A synthetic dual-vendor wheel in the per-binding layout, or with
    `merged` in the merged layout (one file per group of the plan plus each
    set directory's index). `extra` adds raw members; `index_edit(prefix,
    files)` may rewrite an index's {file: members} before it is written."""
    root.mkdir(parents=True, exist_ok=True)
    wrapper = root / 'python/mojolearn/__init__.py'
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text('# current wrapper\n')
    files = {'mojolearn/__init__.py': wrapper.read_bytes()}
    for vendor in vendors:
        arch = 'gfx942' if vendor == 'hip' else 'sm_89'
        for mode in gate.surface.MODES:
            prefix = f'mojolearn/{vendor}/{arch}/' + (mode + '/' if mode != 'fast' else '')
            if merged:
                plan = gate.surface.expected_set_files(mode, False, 'merged')
                for stem in plan:
                    if prefix + stem != omit:
                        files[prefix + stem] = (prefix + stem).encode()
                index = index_edit(prefix, copy.deepcopy(plan)) if index_edit else plan
                if prefix + gate.surface.MERGED_INDEX != omit:
                    files[prefix + gate.surface.MERGED_INDEX] = merged_index_bytes(index)
                continue
            for binding in gate.surface.expected_bindings(mode):
                name = prefix + binding + '.so'
                if name != omit:
                    files[name] = name.encode()
    files.update(extra or {})
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
            per_arch = sum(len(gate.surface.expected_bindings(m)) for m in gate.surface.MODES)
            self.assertEqual(len(extensions), 2 * per_arch)
            self.assertEqual(len(sets), 2 * len(gate.surface.MODES))  # 2 arches x {fast, identical}
            self.assertEqual(sorted(sets.values()), sorted(len(gate.surface.expected_bindings(m))
                                                           for m in gate.surface.MODES for _ in 'ab'))

    def test_complete_dual_wheel_merged_layout(self):
        # The merged layout: one file per group of the plan and one index per
        # set directory; the indexes are collected beside the .so map.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = make_wheel(root, merged=True)
            indexes = {}
            extensions, sets = gate.inspect_wheel(wheel, root, index_out=indexes)
            per_arch = sum(len(gate.surface.expected_set_files(m, False, 'merged')) for m in gate.surface.MODES)
            self.assertEqual(len(extensions), 2 * per_arch)
            self.assertTrue(all(p.endswith('.so') for p in extensions))
            self.assertEqual(len(indexes), 2 * len(gate.surface.MODES))
            self.assertEqual(gate.wheel_layout(indexes), 'merged')
            for path, row in indexes.items():
                self.assertTrue(path.endswith('/' + gate.surface.MERGED_INDEX))
                members = sorted(n for m in row['files'].values() for n in m)
                mode = 'identical' if '/identical/' in path else 'fast'
                self.assertEqual(members, sorted(gate.surface.expected_bindings(mode)))

    def test_merged_layout_defects_refuse(self):
        member = 'mojolearn/hip/gfx942/identical/'
        stem = next(iter(gate.surface.expected_set_files('identical', False, 'merged')))
        def drop_gbdt(prefix, files):
            for m in files.values():
                if '_mojolearn_gbdt' in m:
                    m.remove('_mojolearn_gbdt')
            return files
        def twice(prefix, files):
            files['_mojolearn_twice.so'] = ['_mojolearn_gbdt']
            return files
        cases = {
            'own file shadows the merged file': dict(extra={member + '_mojolearn_gbdt.so': b'own'}),
            'index lacks a member': dict(index_edit=drop_gbdt),
            'member in two files': dict(index_edit=twice),
            'merged file missing': dict(omit=member + stem),
            'index missing': dict(omit=member + gate.surface.MERGED_INDEX),
            'index outside a set': dict(extra={'mojolearn/hip/gfx942/other/' + gate.surface.MERGED_INDEX: b'{}'}),
        }
        for why, options in cases.items():
            with self.subTest(why=why), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                wheel = make_wheel(root, merged=True, **options)
                with self.assertRaises(ValueError):
                    gate.inspect_wheel(wheel, root, index_out={})

    def test_mixed_layouts_refuse(self):
        # One vendor's sets merged, the other's per binding: one layout per wheel.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            merged = make_wheel(root / 'm', merged=True)
            with zipfile.ZipFile(merged) as archive:
                hip = {n: archive.read(n) for n in archive.namelist() if n.startswith('mojolearn/hip/')}
            wheel = make_wheel(root, vendors=('cuda',), extra=hip)
            with self.assertRaises(ValueError):
                gate.inspect_wheel(wheel, root, index_out={})

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

    def vendor_fixture(self, root, merged=False):
        wheel = make_wheel(root, merged=merged)
        indexes = {}
        extensions, sets = gate.inspect_wheel(wheel, root, index_out=indexes)
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
        if merged:
            proof.update(binding_layout='merged',
                         merged_indexes={'mojolearn/' + p: r for p, r in indexes.items() if p.startswith('hip/')})
        write_json(out / 'build-provenance.json', proof)
        audit = {'sha256': wheel_sha, 'qualification_vendor': 'hip',
                 'advertised_vendors': ['cuda', 'hip'], 'extension_hashes': extensions,
                 'sets': sets, 'source_sha256': source_sha,
                 'build_provenance_sha256': gate.digest_file(out / 'build-provenance.json')}
        if merged:
            audit.update(merged_indexes=indexes, binding_layout='merged')
        write_json(out / 'wheel-audit.json', audit)
        records = {}
        package = '/vanished/qualification/venv/lib/python3.11/site-packages/mojolearn'
        for surface, mode in sorted(gate.surface.expected_jobs(audit)):
            for code in (gate.surface.MODES[mode],):
                name = surface + '-' + mode
                prefix = 'hip/gfx942/' + (mode + '/' if mode != 'fast' else '')
                bindings = {}
                for binding in gate.surface.expected_bindings(mode):
                    # Each member loads from its own file or its directory's merged file.
                    member = gate.surface.member_file(prefix, binding, indexes)
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
        if merged:
            return out, qualification, wheel_sha, inventory, extensions, sets, indexes
        return out, qualification, wheel_sha, inventory, extensions, sets

    def test_merged_installed_members_load_from_the_indexed_file(self):
        with tempfile.TemporaryDirectory() as directory:
            out, qualification, wheel_sha, inventory, extensions, sets, indexes = self.vendor_fixture(
                Path(directory), merged=True)
            with patch.object(gate.surface, 'retained', return_value=(qualification, {})), \
                    patch.object(gate.surface, 'check_quality'):
                gate.check_vendor(out, 'hip', wheel_sha, inventory, extensions, sets, indexes=indexes)
                # The audit's indexes must be the wheel's.
                with self.assertRaisesRegex(ValueError, 'merged indexes'):
                    gate.check_vendor(out, 'hip', wheel_sha, inventory, extensions, sets)
            # A member claiming a per-binding file the merged set does not ship refuses.
            path = out / 'smoke-fast.installed.json'
            installed = json.loads(path.read_text())
            row = installed['installed_bindings']['_mojolearn_gbdt']
            row['path'] = row['path'].rsplit('/', 1)[0] + '/_mojolearn_gbdt.so'
            write_json(path, installed)
            qualification['installed_records']['smoke-fast'] = gate.digest_file(path)
            with patch.object(gate.surface, 'retained', return_value=(qualification, {})), \
                    patch.object(gate.surface, 'check_quality'), self.assertRaises(ValueError):
                gate.check_vendor(out, 'hip', wheel_sha, inventory, extensions, sets, indexes=indexes)

    def test_all_installed_records_and_identical_quality_rechecked(self):
        # DEVIATION 2490: UMAP quality is an identical-only job, rechecked once.
        with tempfile.TemporaryDirectory() as directory:
            out, qualification, *args = self.vendor_fixture(Path(directory))
            with patch.object(gate.surface, 'retained', return_value=(qualification, {})), \
                    patch.object(gate.surface, 'check_quality') as quality:
                gate.check_vendor(out, 'hip', *args)
                self.assertEqual(quality.call_count, 1)

    def test_runtime_architecture_requires_device_and_selection_in_every_job(self):
        # Use the full release-linux3 inert fixture (DEVIATION 2290, 2490): 11
        # jobs (smoke in three tiers, eight identical-only) and 17 IDENTICAL bindings.
        # Re-seal mutated evidence so this checks actual architecture admission,
        # not merely stale hashes. Neither this fixture nor gate loads natives.
        from test_release061_end_to_end import fixture, seal_evidence
        for target in ('smoke-fast', 'byte-lm-identical'):
            for defect in (None, 'device_architecture', 'selected_architecture', 'architecture_override_absent'):
                with self.subTest(target=target, defect=defect), tempfile.TemporaryDirectory() as directory:
                    root = Path(directory)
                    wheel, qualification = fixture(root)
                    out = qualification / 'hip/gfx942'
                    if defect:
                        path = out / (target + '.installed.json')
                        row = json.loads(path.read_text())
                        row[defect] = False if defect == 'architecture_override_absent' else 'gfx90a'
                        write_json(path, row)
                        seal_evidence(out)
                    extensions, sets = gate.inspect_wheel(wheel, root, flat_python=True, byte_lm=True)
                    args = (out, 'hip', gate.digest_file(wheel), gate.native_inventory(root), extensions, sets)
                    if defect:
                        with self.assertRaisesRegex(ValueError, 'claimed native architecture'):
                            gate.check_vendor(*args, arch='gfx942')
                    else:
                        result = gate.check_vendor(*args, arch='gfx942')
                        self.assertEqual(len(result['installed_records']), 10)  # smoke x 2 modes, 7 identical surfaces, byte-lm

    def test_stale_hash_inventory_and_missing_job_refuse(self):
        for defect in ('wheel', 'native', 'missing_job', 'build_proof'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as directory:
                out, qualification, wheel_sha, inventory, extensions, sets = self.vendor_fixture(Path(directory))
                if defect == 'wheel':
                    qualification['wheel_sha256'] = 'b' * 64
                elif defect == 'native':
                    inventory = inventory + [['new.mojo', 'b' * 64]]
                elif defect == 'missing_job':
                    qualification['installed_records'].pop('ordered-rmse-identical')
                else:
                    (out / 'build-provenance.json').write_text('{}')
                with patch.object(gate.surface, 'retained', return_value=(qualification, {})), \
                        patch.object(gate.surface, 'check_quality'), self.assertRaises(ValueError):
                    gate.check_vendor(out, 'hip', wheel_sha, inventory, extensions, sets)

    def test_hopper_driver_capability_matches_only_its_supported_binary(self):
        import test_release061_end_to_end as end_to_end
        cases = [('sm_90a', 'sm_90', True), ('sm_90a', 'sm_90a', True),
                 ('sm_90a', 'sm_89', False), ('sm_90a', 'sm_90b', False),
                 ('sm_89', 'sm_89a', False), ('sm_89', 'sm_90', False)]
        for arch, device, accepted in cases:
            with self.subTest(arch=arch, device=device), tempfile.TemporaryDirectory() as directory, \
                    patch.object(end_to_end, 'ARCHES', frozenset({'cuda/sm_89', 'cuda/sm_90a', 'hip/gfx942'})):
                root = Path(directory)
                wheel, qualification = end_to_end.fixture(root)
                out = qualification / 'cuda' / arch
                for path in out.glob('*.installed.json'):
                    row = json.loads(path.read_text())
                    row['device_architecture'] = device
                    write_json(path, row)
                end_to_end.seal_evidence(out)
                extensions, sets = gate.inspect_wheel(wheel, root, flat_python=True, byte_lm=True)
                args = (out, 'cuda', gate.digest_file(wheel), gate.native_inventory(root), extensions, sets)
                if accepted:
                    result = gate.check_vendor(*args, arch=arch)
                    self.assertEqual(len(result['installed_records']), 10)  # smoke x 2 modes, 7 identical surfaces, byte-lm
                else:
                    with self.assertRaisesRegex(ValueError, 'claimed native architecture'):
                        gate.check_vendor(*args, arch=arch)

    def test_nonidentical_job_binding_cannot_hide_behind_success_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            out, qualification, *args = self.vendor_fixture(Path(directory))
            path = out / 'smoke-fast.installed.json'
            installed = json.loads(path.read_text())
            installed['installed_bindings']['_mojolearn_gbdt']['mode_code'] = 1
            write_json(path, installed)
            qualification['installed_records']['smoke-fast'] = gate.digest_file(path)
            with patch.object(gate.surface, 'retained', return_value=(qualification, {})), \
                    patch.object(gate.surface, 'check_quality'), self.assertRaises(ValueError):
                gate.check_vendor(out, 'hip', *args)

    def test_nested_corpus_fingerprint_required_and_current(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            out = root / 'qualification'
            out.mkdir()
            snapshot = {}
            for case in gate.surface.CORPUS_CASES:
                corpus = root / 'mamba/corpus' / case
                corpus.mkdir(parents=True)
                for name in ('x.f32', 'reference.f64'):
                    path = corpus / name
                    path.write_bytes(b'fixture')
                    snapshot[path.relative_to(root).as_posix()] = gate.digest_file(path)
            record = out / 'qualification-sources.json'
            write_json(record, snapshot)
            gate.check_corpora(out, root)
            for case in gate.surface.CORPUS_CASES[1:]:
                missing = {p: h for p, h in snapshot.items()
                           if not p.startswith('mamba/corpus/' + case + '/')}
                write_json(record, missing)
                with self.subTest(case=case), self.assertRaises(ValueError):
                    gate.check_corpora(out, root)
            write_json(record, snapshot)
            (root / 'mamba/corpus/mamba3/m3_base_b2_l4_d32/reference.f64').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                gate.check_corpora(out, root)

    def test_comparators_are_recomputed_and_failure_refuses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            wheel = make_wheel(root)
            for fail in (False, True):
                with self.subTest(fail=fail), patch.object(gate, 'check_vendor'), \
                        patch.object(gate, 'check_corpora'), \
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
                        self.assertEqual(gate.check(wheel, root / 'qualification', root)['jobs_per_vendor'], 9)  # smoke x 2 modes, 7 identical surfaces (no byte-lm off the release profile)
                    umap.assert_called_once()
                    ordered.assert_called_once()


if __name__ == '__main__':
    unittest.main()
