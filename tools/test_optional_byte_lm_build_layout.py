"""Root-only host fixtures for optional native build lists; no compiler calls."""
from pathlib import Path
import re
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]

#: Bindings that build in EVERY tier, and the ones that build in IDENTICAL
#: ONLY. Since DEVIATION 2490 (2026-09-10) the rule is one sentence: the
#: three TREE lanes (gbdt, rf, trees) ship fast and deterministic, every
#: other binding is identical only. Cross-vendor bitwise identity is the
#: product; a fast tier ships only where it has a measured win over the
#: opponent's own CPU (python/mojolearn/_backend.py, `_TIERED`). These
#: tests are what keeps a later edit from quietly putting a lane back into
#: all three.
EVERY_TIER = 3
IDENTICAL_ONLY = 13


class OptionalBuildLayoutTests(unittest.TestCase):
    def rows(self, enabled, function):
        source = (ROOT / 'packaging/linux/build_sets.sh').read_text()
        names = re.search(r'^EXT_NAMES="([^"]+)"$', source, re.M).group(1)
        scripts = re.search(r'^SCRIPTS="\$\{MOJOLEARN_BUILD_SCRIPTS:-([^}]+)\}"$', source, re.M).group(1)
        identical_only_names = re.search(r'^IDENTICAL_ONLY_NAMES="([^"]+)"$', source, re.M).group(1)
        identical_only_scripts = re.search(r'^IDENTICAL_ONLY_SCRIPTS="([^"]+)"$', source, re.M).group(1)
        body = re.search(r'^' + function + r'\(\) \{\n.*?^\}', source, re.M | re.S).group()
        # Execute ONLY the extracted list helper, never the build driver.
        program = ('set -eu\nEXT_NAMES=' + repr(names) + '\nSCRIPTS=' + repr(scripts)
                   + '\nIDENTICAL_ONLY_NAMES=' + repr(identical_only_names)
                   + '\nIDENTICAL_ONLY_SCRIPTS=' + repr(identical_only_scripts)
                   + '\nPACKAGE_BYTE_LM=' + str(enabled) + '\n' + body
                   + '\nfor tier in fast deterministic identical; do ' + function + ' "$tier"; done\n')
        result = subprocess.run(['bash', '-c', program], capture_output=True, text=True,
                                timeout=5, check=True)
        return [line.split() for line in result.stdout.splitlines()]

    def test_only_tree_lanes_build_in_every_tier(self):
        for helper in ('tier_names', 'tier_scripts'):
            rows = self.rows(0, helper)
            self.assertEqual(
                [len(row) for row in rows],
                [EVERY_TIER, EVERY_TIER, EVERY_TIER + IDENTICAL_ONLY],
                helper,
            )
            # fast and deterministic carry the same list, and it is the
            # identical list with the neural lanes removed.
            self.assertEqual(rows[0], rows[1])
            self.assertEqual(rows[2][:EVERY_TIER], rows[0])
            self.assertFalse(any('byte_lm' in entry for row in rows for entry in row))

    def test_no_neural_binding_outside_identical(self):
        for helper in ('tier_names', 'tier_scripts'):
            fast, deterministic, identical = self.rows(1, helper)
            for row, tier in ((fast, 'fast'), (deterministic, 'deterministic')):
                for stem in ('training', 'mamba', 'transformer', 'byte_lm'):
                    self.assertFalse(
                        any(stem in entry for entry in row),
                        f'{stem} must not build in {tier}: {row}',
                    )
            for stem in ('training', 'mamba', 'transformer', 'byte_lm'):
                self.assertTrue(
                    any(stem in entry for entry in identical),
                    f'{stem} must build in identical: {identical}',
                )

    def test_optional_native_exists_once_in_identical_only(self):
        names = self.rows(1, 'tier_names')
        scripts = self.rows(1, 'tier_scripts')
        full = EVERY_TIER + IDENTICAL_ONLY + 1
        self.assertEqual([len(row) for row in names], [EVERY_TIER, EVERY_TIER, full])
        self.assertEqual([len(row) for row in scripts], [EVERY_TIER, EVERY_TIER, full])
        self.assertEqual(names[0], names[1])
        self.assertEqual(names[2][-1], '_mojolearn_byte_lm')
        self.assertEqual(scripts[2][-1], 'build_byte_lm.sh')
        for row in names + scripts:
            self.assertEqual(len(row), len(set(row)))


if __name__ == '__main__':
    unittest.main()
