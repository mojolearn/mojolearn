"""No fitting: test exact glue registration and real packer readback gates."""
import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from device_glue import validate, NO_IMAGE
from test_split_wheels import make_sets, pw


class DeviceGlue(unittest.TestCase):
    def rows(self):
        return [[t, n, NO_IMAGE if n == '_mojolearn_x_trees' else 'gfx942']
                for t in ('fast', 'identical')
                for n in ('_mojolearn_x_trees', '_mojolearn_rf', '_mojolearn_gbdt')]

    def test_registered_glue_requires_both_same_tier_delegates(self):
        validate(self.rows())
        for index in (1, 2, 4, 5):
            rows = self.rows(); rows.pop(index)
            with self.assertRaisesRegex(ValueError, 'delegates'):
                validate(rows)
        rows = self.rows(); rows[1][2] = 'sm_89'
        with self.assertRaisesRegex(ValueError, 'delegates'):
            validate(rows)

    def test_other_bindings_and_unshipped_tiers_cannot_claim_no_image(self):
        for tier, name in [('fast', '_mojolearn_gp'), ('deterministic', '_mojolearn_x_trees')]:
            with self.assertRaisesRegex(ValueError, 'unregistered'):
                validate([[tier, name, NO_IMAGE]])

    def test_packer_accepts_honest_glue_and_rejects_contradictory_binary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); paths = make_sets(root, [('hip', 'gfx942')])
            adir = Path(paths[0]) / 'gfx942'
            for tier in pw.TIERS:
                for name in set(pw.tier_names(tier, True)) - set(pw.tier_names(tier)):
                    (adir / ('' if tier == 'fast' else tier) / (name+'.so')).write_bytes(b'fixture')
                    for leaf, value in [('readback.txt', 'hip'), ('arch_readback.txt', 'gfx942')]:
                        with (adir / leaf).open('a') as f:
                            f.write(f'{tier} {name} {value}\n')
            witness = adir / 'arch_readback.txt'
            rows = [line.split() for line in witness.read_text().splitlines()]
            for row in rows:
                if row[1] == '_mojolearn_x_trees':
                    row[2] = NO_IMAGE
                    binary = adir / ('' if row[0] == 'fast' else row[0]) / (row[1] + '.so')
                    binary.write_bytes(b'actual glue without embedded device image')
            witness.write_text('\n'.join(' '.join(r) for r in rows)+'\n')
            self.assertTrue(pw.load_set(paths[0], include_byte_lm=True))
            (adir / '_mojolearn_x_trees.so').write_bytes(b'embedded gfx942 image')
            with self.assertRaisesRegex(SystemExit, 'contradicts binary'):
                pw.load_set(paths[0], include_byte_lm=True)

    def test_unregistered_missing_image_still_fails_packer(self):
        with tempfile.TemporaryDirectory() as tmp:
            paths = make_sets(Path(tmp), [('hip', 'gfx942')]); adir = Path(paths[0]) / 'gfx942'
            witness = adir / 'arch_readback.txt'
            witness.write_text(witness.read_text().replace('fast _mojolearn_gp gfx942', 'fast _mojolearn_gp '+NO_IMAGE))
            with self.assertRaisesRegex(SystemExit, 'unregistered'):
                pw.load_set(paths[0], include_byte_lm=True)


if __name__ == '__main__':
    unittest.main()
