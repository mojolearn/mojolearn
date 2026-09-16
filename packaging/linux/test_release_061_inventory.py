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
    def fixture(self, root):
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
        for vendor, arch in sorted(packer.RELEASE_061_SETS):
            files = {}
            for mode in packer.TIERS:
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
            for name in packer.HOST_NAMES:
                binary = root / f'{vendor}/{arch}/host/{name}.so'
                binary.parent.mkdir(parents=True, exist_ok=True)
                binary.write_bytes(f'inert host {name}'.encode())
                hosts[name] = binary
            sets.append((vendor, arch, files, {}, {}, hosts))
            proof = root / f'{vendor}-{arch}.json'
            proof.write_text(json.dumps(dict(
                schema='mojolearn.linux.build-provenance.v1', complete=True,
                build_exit=0, action='build', source_commit='a' * 40,
                source_inventory=inventory, source_sha256=source_sha,
                extensions={'mojolearn/' + n: packer.sha(p).hex() for n, p in files.items()},
                # the leg's host binaries, as tools/linux_surface_qualification.sh records them
                host_extension={f'mojolearn/{vendor}/{arch}/host/{n}.so': packer.sha(p).hex()
                                for n, p in hosts.items()})))
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
            # Three tree lanes in each of fast and deterministic, and all
            # twenty-three identical-tier names (the three trees, the nineteen
            # identical-only bindings, and the byte LM), so 29 per architecture
            # across three architectures. Counted from tier_names rather than
            # written out, so the number cannot drift from the packer again.
            # 23 until workstream D (2026-09-14) added kernel_methods, mixture,
            # hdbscan and resample; 27 until ivf and embedding the same day.
            per_arch = sum(len(packer.tier_names(mode, True)) for mode in packer.TIERS)
            self.assertEqual(per_arch, 29)
            self.assertEqual(len(result['extensions']), per_arch * 3)
            self.assertTrue(result['optional_native']['_mojolearn_byte_lm']['included'])
            self.assertEqual(result['optional_native']['_mojolearn_byte_lm']['unsupported_modes'],
                             ['fast', 'deterministic'])
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
            binary = next(iter(sets[0][2].values()))
            original = binary.read_bytes()
            binary.write_bytes(b'changed')
            with self.assertRaises(SystemExit):
                packer.release_inventory(sets, proofs, version, root)
            binary.write_bytes(original)
            (root / 'source.py').write_bytes(b'stale')
            with self.assertRaises(SystemExit):
                packer.release_inventory(sets, proofs, version, root)

    def test_host_binary_must_match_its_own_leg_proof(self):
        # 0.8.6: a host binary changed after the build is refused against the
        # proof's host_extension, even with no other change anywhere.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sets, proofs = self.fixture(root)
            next(iter(sets[0][5].values())).write_bytes(b'changed host binary')
            with self.assertRaises(SystemExit):
                packer.release_inventory(sets, proofs, packer.read_version(root), root)

    def test_missing_or_wrong_mode_byte_lm_refused_even_with_matching_proof(self):
        for misplaced in (False, True):
            with self.subTest(misplaced=misplaced), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                sets, proofs = self.fixture(root)
                vendor, arch, files, _, _, _ = sets[0]
                old = f'{vendor}/{arch}/identical/_mojolearn_byte_lm.so'
                binary = files.pop(old)
                if misplaced:
                    files[f'{vendor}/{arch}/_mojolearn_byte_lm.so'] = binary
                proof = json.loads(proofs[0].read_text())
                proof['extensions'] = {'mojolearn/' + n: packer.sha(p).hex() for n, p in files.items()}
                proofs[0].write_text(json.dumps(proof))
                with self.assertRaises(SystemExit):
                    packer.release_inventory(sets, proofs, packer.read_version(root), root)


# THE POST-RECORD ALLOWLIST (0.8.6). The final wheel is packed from the
# recorded build proofs after the record landed in the manifest's record lists;
# python/mojolearn/host_surface.py may differ from the build in those lists and
# nowhere else, and every other inventoried file and every binary must match.
MANIFEST = 'python/mojolearn/host_surface.py'
BUILT_MANIFEST = (b'"""The manifest."""\n'
                  b'TRAINING_GPU_COLUMNS = (\n    "bench/results/identity_break/old/apple-m4.json",\n)\n'
                  b'TRAINING_FIX_LANES = ("kmeans-sqrt",)\n\n'
                  b'def wheel_bindings():\n    return ()\n')
RECORDED_MANIFEST = (b'"""The manifest."""\n'
                     b'# the 0.8.6 release record\n'
                     b'TRAINING_GPU_COLUMNS = (\n    "bench/results/identity_break/new/apple-m4.json",\n)\n'
                     b'TRAINING_FIX_LANES = ()\n\n'
                     b'def wheel_bindings():\n    return ()\n')
OUTSIDE_MANIFEST = RECORDED_MANIFEST.replace(b'return ()', b'return ("_mojolearn_extra_host",)')


class PostRecordAllowlist(ReleaseInventory):
    def setUp(self):
        packer.post_record_reader = lambda root, commit, rel: BUILT_MANIFEST

    def tearDown(self):
        packer.post_record_reader = None

    def manifest_fixture(self, root):
        # the build had BUILT_MANIFEST; every proof's inventory names its digest
        sets, proofs = self.fixture(root)
        path = root / MANIFEST
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(BUILT_MANIFEST)
        built = hashlib.sha256(BUILT_MANIFEST).hexdigest()
        for proof_path in proofs:
            proof = json.loads(proof_path.read_text())
            inventory = sorted(proof['source_inventory'] + [[MANIFEST, built]])
            proof['source_inventory'] = inventory
            proof['source_sha256'] = hashlib.sha256(json.dumps(inventory, separators=(',', ':')).encode()).hexdigest()
            proof_path.write_text(json.dumps(proof))
        return sets, proofs

    def test_unchanged_tree_records_no_post_record_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sets, proofs = self.manifest_fixture(root)
            result = packer.release_inventory(sets, proofs, packer.read_version(root), root)
            self.assertEqual(result['post_record_files'], [])

    def test_record_lists_only_admitted_and_named(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sets, proofs = self.manifest_fixture(root)
            (root / MANIFEST).write_bytes(RECORDED_MANIFEST)
            result = packer.release_inventory(sets, proofs, packer.read_version(root), root)
            self.assertEqual(result['post_record_files'], [MANIFEST])

    def test_manifest_change_outside_record_lists_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sets, proofs = self.manifest_fixture(root)
            (root / MANIFEST).write_bytes(OUTSIDE_MANIFEST)
            with self.assertRaises(SystemExit):
                packer.release_inventory(sets, proofs, packer.read_version(root), root)

    def test_built_copy_must_hash_to_the_proof_digest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sets, proofs = self.manifest_fixture(root)
            (root / MANIFEST).write_bytes(RECORDED_MANIFEST)
            packer.post_record_reader = lambda r, c, rel: RECORDED_MANIFEST
            with self.assertRaises(SystemExit):
                packer.release_inventory(sets, proofs, packer.read_version(root), root)

    def test_changed_binary_or_other_source_refused_beside_an_allowed_edit(self):
        for defect in ('binary', 'source', 'host binary'):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                sets, proofs = self.manifest_fixture(root)
                (root / MANIFEST).write_bytes(RECORDED_MANIFEST)
                if defect == 'binary':
                    next(iter(sets[0][2].values())).write_bytes(b'changed')
                elif defect == 'source':
                    (root / 'source.py').write_bytes(b'stale')
                else:
                    next(iter(sets[0][5].values())).write_bytes(b'changed host')
                with self.assertRaises(SystemExit):
                    packer.release_inventory(sets, proofs, packer.read_version(root), root)


if __name__ == '__main__':
    unittest.main()
