"""Root-only host fixtures for optional native build lists; no compiler calls."""
from pathlib import Path
import re
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[1]

#: The Linux tier rule, as packaging/linux/build_sets.sh states it. The three
#: TREE lanes (gbdt, rf, trees) build in EVERY tier (DEVIATION 2490,
#: 2026-09-10). CLASSICAL ML (2026-09-25) builds FAST and IDENTICAL, never
#: deterministic: FAST_CLASSICAL_NAMES / FAST_CLASSICAL_SCRIPTS. The neural
#: lanes and svm build IDENTICAL only: IDENTICAL_ONLY_NAMES /
#: IDENTICAL_ONLY_SCRIPTS (FAST svm ships on Apple only; its Linux list is
#: identical only). Cross-vendor bitwise identity is the product; a fast tier
#: ships only where it has a measured win over the opponent's own CPU
#: (python/mojolearn/_backend.py, `_TIERED`). These tests are what keeps a
#: later edit from quietly moving a lane into a tier it does not belong in.
EVERY_TIER = 3
#: 14 since classical FAST (2026-09-25): _mojolearn, estimators, solver,
#: metrics, preprocessing, tsa, linalg, arima, gp, kernel_methods, mixture,
#: hdbscan, resample, ivf.
FAST_CLASSICAL = 14
#: training, mamba, transformer, embedding, svm.
IDENTICAL_ONLY = 5
FAST_ROW = EVERY_TIER + FAST_CLASSICAL
IDENTICAL_ROW = EVERY_TIER + IDENTICAL_ONLY + FAST_CLASSICAL

#: The CPU training binding (DEVIATION 2680, 2026-09-12). It builds in the
#: identical tier only, and it is named in tier_SCRIPTS but deliberately NOT in
#: tier_NAMES -- see test_optional_native_exists_once_in_identical_only for the
#: reason and for what goes wrong when a name-keyed list picks it up.
HOST_NAME = '_mojolearn_byte_lm_host'
HOST_SCRIPT = 'build_byte_lm_host.sh'
#: Since 0.8.6 (the packaging lane) EVERY host family's binding builds in the
#: identical pass, in the manifest's order (build_sets.sh reads
#: `host_surface.py --wheel-families`); the byte LM's is first.
HOST_FAMILIES = subprocess.run(
    ['python3', str(ROOT / 'python/mojolearn/host_surface.py'), '--wheel-families'],
    capture_output=True, text=True, check=True, timeout=30,
).stdout.split()


class OptionalBuildLayoutTests(unittest.TestCase):
    def rows(self, enabled, function):
        source = (ROOT / 'packaging/linux/build_sets.sh').read_text()
        names = re.search(r'^EXT_NAMES="([^"]+)"$', source, re.M).group(1)
        scripts = re.search(r'^SCRIPTS="\$\{MOJOLEARN_BUILD_SCRIPTS:-([^}]+)\}"$', source, re.M).group(1)
        identical_only_names = re.search(r'^IDENTICAL_ONLY_NAMES="([^"]+)"$', source, re.M).group(1)
        identical_only_scripts = re.search(r'^IDENTICAL_ONLY_SCRIPTS="([^"]+)"$', source, re.M).group(1)
        fast_classical_names = re.search(r'^FAST_CLASSICAL_NAMES="([^"]+)"$', source, re.M).group(1)
        fast_classical_scripts = re.search(r'^FAST_CLASSICAL_SCRIPTS="([^"]+)"$', source, re.M).group(1)
        body = re.search(r'^' + function + r'\(\) \{\n.*?^\}', source, re.M | re.S).group()
        # Execute ONLY the extracted list helper, never the build driver.
        program = ('set -eu\nEXT_NAMES=' + repr(names) + '\nSCRIPTS=' + repr(scripts)
                   + '\nIDENTICAL_ONLY_NAMES=' + repr(identical_only_names)
                   + '\nIDENTICAL_ONLY_SCRIPTS=' + repr(identical_only_scripts)
                   + '\nFAST_CLASSICAL_NAMES=' + repr(fast_classical_names)
                   + '\nFAST_CLASSICAL_SCRIPTS=' + repr(fast_classical_scripts)
                   + '\nPACKAGE_BYTE_LM=' + str(enabled)
                   + '\nHOST_FAMILIES=' + repr(' '.join(HOST_FAMILIES)) + '\n' + body
                   + '\nfor tier in fast deterministic identical; do ' + function + ' "$tier"; done\n')
        result = subprocess.run(['bash', '-c', program], capture_output=True, text=True,
                                timeout=5, check=True)
        return [line.split() for line in result.stdout.splitlines()]

    def test_only_tree_lanes_build_in_every_tier(self):
        for helper in ('tier_names', 'tier_scripts'):
            fast, deterministic, identical = self.rows(0, helper)
            self.assertEqual(
                [len(fast), len(deterministic), len(identical)],
                [FAST_ROW, EVERY_TIER, IDENTICAL_ROW],
                helper,
            )
            # deterministic is the tree lanes alone; fast is the tree lanes
            # plus classical; identical is a superset of fast.
            self.assertEqual(fast[:EVERY_TIER], deterministic)
            self.assertEqual(identical[:EVERY_TIER], deterministic)
            self.assertLessEqual(set(fast), set(identical))
            self.assertEqual(fast[EVERY_TIER:], identical[EVERY_TIER + IDENTICAL_ONLY:])
            self.assertFalse(any('byte_lm' in entry for row in (fast, deterministic, identical)
                                 for entry in row))

    def test_classical_is_fast_and_identical_never_deterministic(self):
        for helper in ('tier_names', 'tier_scripts'):
            fast, deterministic, identical = self.rows(0, helper)
            classical = fast[EVERY_TIER:]
            self.assertEqual(len(classical), FAST_CLASSICAL, helper)
            for entry in classical:
                self.assertNotIn(entry, deterministic, helper)
                self.assertIn(entry, identical, helper)
            # FAST svm is Apple-only: on Linux svm builds in identical alone.
            for row, tier in ((fast, 'fast'), (deterministic, 'deterministic')):
                self.assertFalse(any('svm' in entry for entry in row),
                                 f'svm must not build in {tier} on Linux: {row}')
            self.assertTrue(any('svm' in entry for entry in identical), identical)

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
        full = IDENTICAL_ROW + 1
        self.assertEqual([len(row) for row in names], [FAST_ROW, EVERY_TIER, full])
        # tier_SCRIPTS CARRIES ONE MORE ENTRY THAN tier_NAMES, ON PURPOSE, AND
        # THE ASYMMETRY IS THE POINT OF THIS ASSERTION. The CPU training
        # binding builds in the identical tier's pass and is absent from
        # tier_names, because names drive the per-tier read-back loop and the
        # staging move while this binary is neither a tier member nor a vendor
        # member: it answers 'cpu', carries no GPU code, and is staged once
        # beside the tiers in <set>/host/. Folding it into any name-keyed list
        # gets it refused as a non-GPU vendor, so it carries named checks of
        # its own instead -- tools/test_build_sets_host_reductions.py is the
        # one that keeps the read-back REDUCTIONS excluding its row.
        host_scripts = [f'build_{f}_host.sh' for f in HOST_FAMILIES]
        self.assertEqual(host_scripts[0], HOST_SCRIPT)
        self.assertEqual([len(row) for row in scripts],
                         [FAST_ROW, EVERY_TIER, full + len(host_scripts)])
        self.assertNotIn(HOST_NAME, names[2])
        self.assertEqual(scripts[2][full:], host_scripts)
        self.assertEqual(scripts[2][full - 1], 'build_byte_lm.sh')
        for row in (names[0], names[1], scripts[0], scripts[1]):
            self.assertNotIn(HOST_NAME, row)
            self.assertNotIn(HOST_SCRIPT, row)
        self.assertEqual(names[0][:EVERY_TIER], names[1])
        self.assertEqual(names[2][-1], '_mojolearn_byte_lm')
        for row in names + scripts:
            self.assertEqual(len(row), len(set(row)))


if __name__ == '__main__':
    unittest.main()
