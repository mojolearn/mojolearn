"""File-only packer fixtures, authored without execution; root runs checks."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('pack_wheel', Path(__file__).with_name('pack_wheel.py'))
packer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(packer)


class ReleaseInventory(unittest.TestCase):
    def test_python_payload_includes_imported_model_subpackage(self):
        entries = packer.python_package_entries()
        for name in ("__init__", "causal_lm", "config", "safetensors", "tokenizer"):
            path = f"mojolearn/models/{name}.py"
            self.assertEqual(entries[path].read_bytes(), (packer.PKG / "models" / (name + ".py")).read_bytes())
        self.assertFalse(any("/tests/" in path for path in entries))

    def fixture(self, root, merged=False):
        """Three release sets and their proofs. `merged`: each set directory
        in the merged layout (bindings/merged_groups.json plan files and the
        index), the proofs naming the merged files and their indexes."""
        # DEVIATION 2290: the fixture root declares its own version; the tests
        # read it back through the packer's shared reader and pin no number.
        version_file = root / 'python/mojolearn/_version.py'
        version_file.parent.mkdir(parents=True)
        version_file.write_text('__version__ = "9.9.9"\n')
        source = root / 'source.py'
        source.write_bytes(b'fixture source\n')
        inventory = [['source.py', packer.sha(source).hex()]]
        source_sha = hashlib.sha256(json.dumps(inventory, separators=(',', ':')).encode()).hexdigest()
        sets, proofs = [], []
        index_name = packer.MERGED_INDEX
        for vendor, arch in sorted(packer.RELEASE_061_SETS):
            files, indexes = {}, {}
            for mode in packer.TIERS:
                if merged:
                    sub = '' if mode == 'fast' else mode + '/'
                    plan = packer.merged_layout.plan(mode, True)
                    for f in plan:
                        rel = f'{vendor}/{arch}/{sub}{f}'
                        binary = root / rel
                        binary.parent.mkdir(parents=True, exist_ok=True)
                        binary.write_bytes(rel.encode())  # inert bytes; never loaded
                        files[rel] = binary
                    index = root / f'{vendor}/{arch}/{sub}{index_name}'
                    index.write_text(json.dumps(dict(format=1, files=plan), sort_keys=True))
                    files[f'{vendor}/{arch}/{sub}{index_name}'] = index
                    indexes[sub + index_name] = index
                    continue
                # FROM THE PACKER'S OWN DEFINITION, never a second list. This
                # read `EXT_NAMES + byte_lm`, which was right until DEVIATION
                # 2490 moved every non-tree binding into identical and narrowed
                # EXT_NAMES to the three tree lanes. The fixture then built 4
                # identical-tier files where the packer required 17, and this
                # file's first test had been failing on main ever since; the
                # other three assert refusals, so they passed for the wrong
                # reason and hid it.
                names = packer.tier_names(mode, True)
                for name in names:
                    rel = f'{vendor}/{arch}/' + ('' if mode == 'fast' else mode + '/') + name + '.so'
                    binary = root / rel
                    binary.parent.mkdir(parents=True, exist_ok=True)
                    binary.write_bytes(rel.encode())  # inert bytes; never loaded
                    files[rel] = binary
            # DEVIATION 2680: the sixth element is this leg's host bindings,
            # basename -> path ({} or None when the leg built none). The
            # release profile requires every binding the manifest ships
            # (since 0.8.6), so these fixtures build one inert file per
            # manifest name, the same bytes on every leg; the absent case is
            # the generic profile's and is not what release_inventory packs.
            hosts = {}
            if merged:
                # one merged host file serves every host binding, beside its index
                plan = packer.merged_layout.plan('host', True)
                for f, names in plan.items():
                    binary = root / f'{vendor}/{arch}/host/{f}'
                    binary.parent.mkdir(parents=True, exist_ok=True)
                    binary.write_bytes(f'inert host {f}'.encode())
                    hosts.update({name: binary for name in names})
                index = root / f'{vendor}/{arch}/host/{index_name}'
                index.write_text(json.dumps(dict(format=1, files=plan), sort_keys=True))
                indexes['host/' + index_name] = index
            else:
                for name in packer.HOST_NAMES:
                    binary = root / f'{vendor}/{arch}/host/{name}.so'
                    binary.parent.mkdir(parents=True, exist_ok=True)
                    binary.write_bytes(f'inert host {name}'.encode())
                    hosts[name] = binary
            # The seventh element (2026-09-23) is the set's reuse.json record,
            # None for a set a leg built with nothing taken from a published wheel;
            # the eighth and ninth the binding layout and its indexes.
            sets.append(packer.SetDir(vendor, arch, files, {}, {}, hosts, None,
                                      'merged' if merged else 'per-binding', indexes))
            proof = root / f'{vendor}-{arch}.json'
            proof.write_text(json.dumps(dict(
                schema='mojolearn.linux.build-provenance.v1', complete=True,
                build_exit=0, action='build', source_commit='a' * 40,
                source_inventory=inventory, source_sha256=source_sha,
                extensions={'mojolearn/' + n: packer.sha(p).hex() for n, p in files.items() if n.endswith('.so')},
                **(dict(binding_layout='merged', merged_indexes={
                    f'mojolearn/{vendor}/{arch}/{rel}': packer.index_record(path)
                    for rel, path in indexes.items()}) if merged else {}))))
            proofs.append(proof)
        return sets, proofs

    def test_exact_three_architectures_remain_runtime_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sets, proofs = self.fixture(root)
            version = packer.read_version(root)
            result = packer.release_inventory(sets, proofs, version, root)
            self.assertEqual(result['version'], version)
            self.assertEqual(result['assembly_profile'], packer.RELEASE_PROFILE)
            # Three tree lanes in fast, and all
            # twenty-three identical-tier names (the three trees, the nineteen
            # identical-only bindings, and the byte LM), so 29 per architecture
            # across three architectures. Counted from tier_names rather than
            # written out, so the number cannot drift from the packer again.
            # 23 until workstream D (2026-09-14) added kernel_methods, mixture,
            # hdbscan and resample; 27 until ivf and embedding the same day.
            # 44 since classical FAST (2026-09-25): fifteen classical names
            # join the fast tier (43 while FAST svm was Apple only).
            # Plus the expansion lanes' GPU bindings (lane/algos-prep,
            # 2026-09-27), read from the manifest's fragments: a classical one
            # is in fast and identical, a neural one in identical alone.
            per_arch = sum(len(packer.tier_names(mode, True)) for mode in packer.TIERS)
            expansion = (2 * len(packer.host_surface.expansion_gpu_bindings("classical"))
                         + len(packer.host_surface.expansion_gpu_bindings("identical-only")))
            # Four neural blocks now also ship FAST (py/Apple takeover).
            self.assertEqual(per_arch, 45 + expansion)
            self.assertEqual(len(result['extensions']), per_arch * 3)
            self.assertTrue(result['optional_native']['_mojolearn_byte_lm']['included'])
            self.assertEqual(result['optional_native']['_mojolearn_byte_lm']['unsupported_modes'],
                             ['fast'])
            self.assertEqual(set(result['runtime_coverage'].values()), {'PENDING_INSTALLED_ARTIFACT'})

    def test_wrong_version_or_missing_architecture_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sets, proofs = self.fixture(root)
            declared = packer.read_version(root)
            # DEVIATION 2290: any version other than the root's declared one is refused.
            for version, selected in [(declared + '.post1', sets), (declared, sets[:2])]:
                with self.assertRaises(SystemExit):
                    packer.release_inventory(selected, proofs, version, root)

    def test_changed_binary_duplicate_proof_and_stale_source_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sets, proofs = self.fixture(root)
            version = packer.read_version(root)
            with self.assertRaises(SystemExit):
                packer.release_inventory(sets, [proofs[0]] * 3, version, root)
            binary = next(iter(sets[0].files.values()))
            original = binary.read_bytes()
            binary.write_bytes(b'changed')
            with self.assertRaises(SystemExit):
                packer.release_inventory(sets, proofs, version, root)
            binary.write_bytes(original)
            (root / 'source.py').write_bytes(b'stale')
            with self.assertRaises(SystemExit):
                packer.release_inventory(sets, proofs, version, root)

    def test_missing_or_wrong_mode_byte_lm_refused_even_with_matching_proof(self):
        for misplaced in (False, True):
            with self.subTest(misplaced=misplaced), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                sets, proofs = self.fixture(root)
                vendor, arch, files = sets[0].vendor, sets[0].arch, sets[0].files
                old = f'{vendor}/{arch}/identical/_mojolearn_byte_lm.so'
                binary = files.pop(old)
                if misplaced:
                    files[f'{vendor}/{arch}/_mojolearn_byte_lm.so'] = binary
                proof = json.loads(proofs[0].read_text())
                proof['extensions'] = {'mojolearn/' + n: packer.sha(p).hex() for n, p in files.items()}
                proofs[0].write_text(json.dumps(proof))
                with self.assertRaises(SystemExit):
                    packer.release_inventory(sets, proofs, packer.read_version(root), root)

    def test_merged_sets_inventory_their_files_and_indexes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sets, proofs = self.fixture(root, merged=True)
            result = packer.release_inventory(sets, proofs, packer.read_version(root), root)
            self.assertEqual(result['binding_layout'], 'merged')
            # `extensions` are the shipped binaries: the planned files per mode
            per_arch = sum(len(packer.merged_layout.plan(mode, True)) for mode in packer.TIERS)
            self.assertEqual(len(result['extensions']), per_arch * 3)
            self.assertTrue(all(n.endswith('.so') for n in result['extensions']))
            # one index per merged GPU directory, and the host index once
            gpu = {f'mojolearn/{v}/{a}/{sub}{packer.MERGED_INDEX}'
                   for v, a in packer.RELEASE_061_SETS for sub in ('', 'identical/')}
            self.assertEqual(set(result['merged_indexes']), gpu | {'mojolearn/host/' + packer.MERGED_INDEX})
            for name, row in result['merged_indexes'].items():
                self.assertRegex(row['sha256'], '^[0-9a-f]{64}$')
                members = sorted(n for m in row['files'].values() for n in m)
                if name.startswith('mojolearn/host/'):
                    self.assertEqual(members, sorted(packer.HOST_NAMES))
                else:
                    mode = 'identical' if '/identical/' in name else 'fast'
                    self.assertEqual(members, sorted(packer.tier_names(mode, True)))
            # every host binding's record names the merged host file it loads from
            self.assertEqual({r['archive_path'] for r in result['host_native'].values()},
                             {'mojolearn/host/_mojolearn_native_host.so'})
            self.assertEqual(set(result['host_native']), set(packer.HOST_NAMES))
            per, per_proofs = self.fixture(root / 'per')
            flat = packer.release_inventory(per, per_proofs, packer.read_version(root / 'per'), root / 'per')
            self.assertEqual((flat['binding_layout'], flat['merged_indexes']), ('per-binding', {}))

    def test_merged_proof_must_index_every_binding_and_name_its_layout(self):
        def byte_lm_dropped(doc, key):
            row = doc['merged_indexes'][key + 'identical/' + packer.MERGED_INDEX]
            for members in row['files'].values():
                if '_mojolearn_byte_lm' in members:
                    members.remove('_mojolearn_byte_lm')

        def byte_lm_in_fast(doc, key):
            row = doc['merged_indexes'][key + packer.MERGED_INDEX]
            next(iter(row['files'].values())).append('_mojolearn_byte_lm')

        def stale_index(doc, key):
            doc['merged_indexes'][key + packer.MERGED_INDEX]['sha256'] = '0' * 64

        def no_host_index(doc, key):
            del doc['merged_indexes'][key + 'host/' + packer.MERGED_INDEX]

        def per_binding_layout(doc, key):
            doc['binding_layout'] = 'per-binding'

        for mutate in (byte_lm_dropped, byte_lm_in_fast, stale_index, no_host_index, per_binding_layout):
            with self.subTest(mutate.__name__), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                sets, proofs = self.fixture(root, merged=True)
                doc = json.loads(proofs[0].read_text())
                mutate(doc, f'mojolearn/{sets[0].vendor}/{sets[0].arch}/')
                proofs[0].write_text(json.dumps(doc))
                with self.assertRaises(SystemExit):
                    packer.release_inventory(sets, proofs, packer.read_version(root), root)

    def test_merged_and_per_binding_sets_never_share_an_inventory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            merged, merged_proofs = self.fixture(root, merged=True)
            per, per_proofs = self.fixture(root / 'per')
            with self.assertRaisesRegex(SystemExit, 'one binding layout'):
                packer.release_inventory(merged[:2] + per[2:], merged_proofs[:2] + per_proofs[2:],
                                         packer.read_version(root), root)


if __name__ == '__main__':
    unittest.main()
