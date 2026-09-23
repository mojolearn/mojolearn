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
            # The seventh element (2026-09-23) is the set's reuse.json record,
            # None for a set a leg built with nothing taken from a published wheel.
            sets.append(packer.SetDir(vendor, arch, files, {}, {}, hosts, None))
            proof = root / f'{vendor}-{arch}.json'
            proof.write_text(json.dumps(dict(
                schema='mojolearn.linux.build-provenance.v1', complete=True,
                build_exit=0, action='build', source_commit='a' * 40,
                source_inventory=inventory, source_sha256=source_sha,
                extensions={'mojolearn/' + n: packer.sha(p).hex() for n, p in files.items()})))
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


if __name__ == '__main__':
    unittest.main()
