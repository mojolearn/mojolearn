"""The native-source rule has three copies that must agree (0.8.6): the
canonical native_inventory (tools/check_linux_release_qualification.py), the
build snapshot in tools/linux_surface_qualification.sh, and the NVIDIA release
archive filter in tools/gemm_remote_leg.sh. A path one copy calls native and
another does not is either missing from a build box or unbound by the proofs.
The tokenizer's Unicode table generator (tokenizer/tools/) writes Mojo source
the tokenizer host binding compiles, so it is native source. File-only."""
from pathlib import Path
import re
import unittest

import check_linux_release_qualification as gate

ROOT = Path(__file__).resolve().parent.parent
GENERATORS = ('tokenizer/tools/gen_unicode_table.sh', 'tokenizer/tools/gen_unicode_categories.py')
COPIES = ('tools/check_linux_release_qualification.py', 'tools/linux_surface_qualification.sh', 'tools/gemm_remote_leg.sh')


def prefixes(rel):
    """The prefix tuple each copy tests with startswith before its .py/.sh suffix."""
    text = (ROOT / rel).read_text()
    found = re.findall(r"startswith\(\(\s*((?:['\"][^'\"]+/['\"],?\s*)+)\)\)", text)
    tuples = [tuple(re.findall(r"['\"]([^'\"]+/)['\"]", group)) for group in found]
    native = [t for t in tuples if 'bindings/' in t]
    if len(native) != 1:
        raise AssertionError(f'{rel}: expected one native-source prefix tuple, found {native}')
    return native[0]


class NativeSourceRule(unittest.TestCase):
    def test_three_copies_agree(self):
        seen = {rel: prefixes(rel) for rel in COPIES}
        self.assertEqual(len(set(seen.values())), 1, seen)

    def test_tokenizer_generator_is_native_source(self):
        self.assertIn('tokenizer/tools/', prefixes(COPIES[0]))
        listed = {rel for rel, _ in gate.native_inventory(ROOT)}
        for rel in GENERATORS:
            self.assertTrue((ROOT / rel).is_file(), rel)
            self.assertIn(rel, listed)

    def test_generated_evidence_is_not_native_source(self):
        # 0.9.0 linux-pack: a box-only evidence copy in the leg's inventory (lane mac-speed-1, 2026-11-27)
        rel = 'build/generated-sources/1795798918547688684-283/generated-inputs/core/native_archive_unicode_names.mojo'
        self.assertFalse(gate.is_native_source(rel))
        self.assertTrue(gate.is_native_source('core/native_archive_unicode_names.mojo'))
        # left out of the inventory only after its bytes are checked (tools/test_generated_source_check.py)
        import generated_source_check
        self.assertEqual(gate.GENERATED_EVIDENCE_PREFIX, generated_source_check.PREFIX)
        shell = (ROOT / 'tools/linux_surface_qualification.sh').read_text()
        self.assertIn("rel.startswith(generated_source_check.PREFIX)", shell)
        self.assertIn("generated_source_check.verify_copy(root, rel, path)", shell)


if __name__ == '__main__':
    unittest.main()
