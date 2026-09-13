"""EVERY READ-BACK REDUCTION MUST SKIP THE HOST ROW, AND A COMMENT IS NOT A CHECK.

The CPU training binding (DEVIATION 2680) is vendor-neutral: it answers `cpu`
in readback.txt and `NONE-BY-DESIGN` in arch_readback.txt. Both are correct and
neither is a GPU value, so any check that reduces one of those files to a single
value counts the host row as a second architecture or a second vendor and
refuses a build that is in fact fine.

That happened. On 2026-09-12 two rented boxes died with

    REFUSING: the binaries do not agree on ONE architecture: gfx942

while reporting exactly one architecture, because `$1!="host"` had been added to
ARCH_SET alone and not to N_ARCH, ARCH and VENDORS. build_sets.sh now carries a
comment saying every reducing site must skip the host row. A comment does not
fail, so the next reduction added without the exclusion costs another GPU hour
to discover. This test fails instead.

The three classes of site are deliberately different and all three are checked:

  reducing   `VAR=$(awk ... "$ARCHBACK")`  -- MUST exclude the host row
  display    `awk ... | sed`/`head`        -- must NOT exclude it, or the
                                              evidence stops showing the binary
                                              it is evidence for
  exact-match `$3=="NONE"`                 -- safe only while it stays an exact
                                              comparison; loosened to a regex it
                                              would match NONE-BY-DESIGN and
                                              refuse every host build
"""

import re
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "packaging" / "linux" / "build_sets.sh"
READ_BACK_FILES = ('"$ARCHBACK"', '"$READBACK"')
HOST_EXCLUSION = '$1!="host"'


def read_back_awk_lines():
    """Every line running awk over a read-back file, as (lineno, text)."""
    out = []
    for n, line in enumerate(SCRIPT.read_text().splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        if "awk" not in stripped:
            continue
        if not any(f in stripped for f in READ_BACK_FILES):
            continue
        out.append((n, stripped))
    return out


def classify(text):
    """reducing | display | exact-match, by what the line does with the output."""
    if re.match(r'^[A-Za-z_][A-Za-z0-9_]*=\$\(awk', text):
        return "reducing"
    if re.search(r'\$3\s*==\s*"[A-Z-]+"', text):
        return "exact-match"
    return "display"


class BuildSetsHostReductionTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.is_file(), f"{SCRIPT} is missing")
        self.lines = read_back_awk_lines()

    def test_the_script_still_has_read_back_reductions(self):
        """A refactor that removes them all must not silently pass this file."""
        kinds = [classify(t) for _, t in self.lines]
        self.assertGreaterEqual(
            kinds.count("reducing"), 4,
            "expected at least the four known reducing sites "
            "(ARCH_SET, N_ARCH, ARCH, VENDORS); found "
            f"{kinds.count('reducing')} in {len(self.lines)} read-back awk lines. "
            "If the reductions moved, move this check with them.",
        )

    def test_every_reducing_site_skips_the_host_row(self):
        bad = [
            (n, t) for n, t in self.lines
            if classify(t) == "reducing" and HOST_EXCLUSION not in t
        ]
        self.assertEqual(
            bad, [],
            "these sites reduce a read-back file to one value without skipping "
            f"the host row, so the CPU training binding reads as a second "
            f"architecture/vendor and the build is refused:\n"
            + "\n".join(f"  {SCRIPT.name}:{n}: {t}" for n, t in bad),
        )

    def test_display_sites_still_show_the_host_row(self):
        bad = [
            (n, t) for n, t in self.lines
            if classify(t) == "display" and HOST_EXCLUSION in t
        ]
        self.assertEqual(
            bad, [],
            "these sites only PRINT the read-back file as evidence, and they "
            "exclude the host row, so the evidence stops showing the binary it "
            "is evidence for:\n"
            + "\n".join(f"  {SCRIPT.name}:{n}: {t}" for n, t in bad),
        )

    def test_the_none_check_stays_an_exact_comparison(self):
        """`$3=="NONE"` is safe; `$3~/NONE/` would match NONE-BY-DESIGN."""
        for n, t in self.lines:
            if "NONE" not in t:
                continue
            self.assertNotRegex(
                t, r'\$3\s*~',
                f"{SCRIPT.name}:{n} matches the architecture field with a "
                "regex. The host row answers NONE-BY-DESIGN, which a /NONE/ "
                "regex matches, so every build carrying the CPU training "
                f"binding would be refused as carrying no device code:\n  {t}",
            )


if __name__ == "__main__":
    unittest.main()
