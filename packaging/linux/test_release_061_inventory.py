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
        source = root / 'source.py'
        source.write_bytes(b'fixture source\n')
        inventory = [['source.py', packer.sha(source).hex()]]
        source_sha = hashlib.sha256(json.dumps(inventory, separators=(',', ':')).encode()).hexdigest()
        sets, proofs = [], []
        for vendor, arch in sorted(packer.RELEASE_061_SETS):
            files = {}
            for mode in packer.TIERS:
                names = packer.EXT_NAMES + (('_mojolearn_byte_lm',) if mode == 'identical' else ())
                for name in names:
                    rel = f'{vendor}/{arch}/' + ('' if mode == 'fast' else mode + '/') + name + '.so'
                    binary = root / rel
                    binary.parent.mkdir(parents=True, exist_ok=True)
                    binary.write_bytes(rel.encode())  # inert bytes; never loaded
                    files[rel] = binary
            sets.append((vendor, arch, files, {}, {}))
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
            result = packer.release_inventory(sets, proofs, '0.6.1', root)
            self.assertEqual(len(result['extensions']), 138)
            self.assertTrue(result['optional_native']['_mojolearn_byte_lm']['included'])
            self.assertEqual(result['optional_native']['_mojolearn_byte_lm']['unsupported_modes'],
                             ['fast', 'deterministic'])
            self.assertEqual(set(result['runtime_coverage'].values()), {'PENDING_INSTALLED_ARTIFACT'})

    def test_wrong_version_or_missing_architecture_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sets, proofs = self.fixture(root)
            for version, selected in [('0.6.0', sets), ('0.6.1', sets[:2])]:
                with self.assertRaises(SystemExit):
                    packer.release_inventory(selected, proofs, version, root)

    def test_changed_binary_duplicate_proof_and_stale_source_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sets, proofs = self.fixture(root)
            with self.assertRaises(SystemExit):
                packer.release_inventory(sets, [proofs[0]] * 3, '0.6.1', root)
            binary = next(iter(sets[0][2].values()))
            original = binary.read_bytes()
            binary.write_bytes(b'changed')
            with self.assertRaises(SystemExit):
                packer.release_inventory(sets, proofs, '0.6.1', root)
            binary.write_bytes(original)
            (root / 'source.py').write_bytes(b'stale')
            with self.assertRaises(SystemExit):
                packer.release_inventory(sets, proofs, '0.6.1', root)

    def test_missing_or_wrong_mode_byte_lm_refused_even_with_matching_proof(self):
        for misplaced in (False, True):
            with self.subTest(misplaced=misplaced), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                sets, proofs = self.fixture(root)
                vendor, arch, files, _, _ = sets[0]
                old = f'{vendor}/{arch}/identical/_mojolearn_byte_lm.so'
                binary = files.pop(old)
                if misplaced:
                    files[f'{vendor}/{arch}/_mojolearn_byte_lm.so'] = binary
                proof = json.loads(proofs[0].read_text())
                proof['extensions'] = {'mojolearn/' + n: packer.sha(p).hex() for n, p in files.items()}
                proofs[0].write_text(json.dumps(proof))
                with self.assertRaises(SystemExit):
                    packer.release_inventory(sets, proofs, '0.6.1', root)


if __name__ == '__main__':
    unittest.main()
