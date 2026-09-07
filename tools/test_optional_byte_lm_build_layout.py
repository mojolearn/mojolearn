"""Root-only host fixtures for optional native build lists; no compiler calls."""
from pathlib import Path
import re
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]


class OptionalBuildLayoutTests(unittest.TestCase):
    def rows(self, enabled, function):
        source = (ROOT / 'packaging/linux/build_sets.sh').read_text()
        names = re.search(r'^EXT_NAMES="([^"]+)"$', source, re.M).group(1)
        scripts = re.search(r'^SCRIPTS="\$\{MOJOLEARN_BUILD_SCRIPTS:-([^}]+)\}"$', source, re.M).group(1)
        body = re.search(r'^' + function + r'\(\) \{\n.*?^\}', source, re.M | re.S).group()
        # Execute ONLY the extracted list helper, never the build driver.
        program = ('set -eu\nEXT_NAMES=' + repr(names) + '\nSCRIPTS=' + repr(scripts)
                   + '\nPACKAGE_BYTE_LM=' + str(enabled) + '\n' + body
                   + '\nfor tier in fast deterministic identical; do ' + function + ' "$tier"; done\n')
        result = subprocess.run(['bash', '-c', program], capture_output=True, text=True,
                                timeout=5, check=True)
        return [line.split() for line in result.stdout.splitlines()]

    def test_legacy_keeps_fifteen_bindings_and_scripts_per_mode(self):
        for helper in ('tier_names', 'tier_scripts'):
            rows = self.rows(0, helper)
            self.assertEqual([len(row) for row in rows], [15, 15, 15])
            self.assertEqual(rows[0], rows[1])
            self.assertEqual(rows[1], rows[2])
            self.assertFalse(any('byte_lm' in entry for row in rows for entry in row))

    def test_optional_native_exists_once_in_identical_only(self):
        names = self.rows(1, 'tier_names')
        scripts = self.rows(1, 'tier_scripts')
        self.assertEqual([len(row) for row in names], [15, 15, 16])
        self.assertEqual([len(row) for row in scripts], [15, 15, 16])
        self.assertEqual(names[0], names[1])
        self.assertEqual(names[2][:-1], names[0])
        self.assertEqual(names[2][-1], '_mojolearn_byte_lm')
        self.assertEqual(scripts[2][-1], 'build_byte_lm.sh')
        for row in names + scripts:
            self.assertEqual(len(row), len(set(row)))


if __name__ == '__main__':
    unittest.main()
