# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Generate the tokenizer's Unicode class ranges AT BUILD TIME into
`tokenizer/impl/unicode_table_generated.mojo` (not tracked; .gitignore).

NO DATA FILE IS TRACKED (2026-09-15). The ranges are computed here from the
running interpreter's standard `unicodedata` and compiled into the tokenizer
binding as a string constant, so neither the checkout nor a wheel carries a
table file. `bindings/build_host_family.sh tokenizer` and
`pixi run check-tokenizer` run `tokenizer/tools/gen_unicode_table.sh`, which
runs this script under a Python whose `unicodedata` is the pinned version.

THE PIN. `tokenizer/impl/unicode_class.mojo` holds `UNICODE_VERSION_PINNED`
and `UNICODE_TABLE_SHA256_PINNED`. This script refuses to write when the
interpreter's `unicodedata.unidata_version` is not the pinned version or the
generated text's sha256 is not the pinned hash, so every build compiles the
same bytes, and the loader refuses a generated module whose constants
disagree with the pin. The pin lives in tracked Mojo source so the binding
cache key (tools/bincache.py hashes the import closure) moves with it.

THE CLASSES the GPT-2 pre-tokenizer pattern names, and nothing else:

  L   general category Letter = Lu + Ll + Lt + Lm + Lo (`\\p{L}`; marks are
      not letters).
  N   general category Number = Nd + Nl + No (`\\p{N}`).
  WS  the White_Space property (`\\s`), written out below. It is NOT
      Python's `str.isspace()`, which also answers True for U+001C..U+001F.

Run from the repository root:

    sh tokenizer/tools/gen_unicode_table.sh            # the build's entry
    python3 tokenizer/tools/gen_unicode_categories.py --print-sha
"""

import hashlib
import os
import re
import sys
import unicodedata

# The White_Space property (unchanged since Unicode 4.1).
WHITE_SPACE = [
    (0x0009, 0x000D),
    (0x0020, 0x0020),
    (0x0085, 0x0085),
    (0x00A0, 0x00A0),
    (0x1680, 0x1680),
    (0x2000, 0x200A),
    (0x2028, 0x2029),
    (0x202F, 0x202F),
    (0x205F, 0x205F),
    (0x3000, 0x3000),
]

PIN_SOURCE = "tokenizer/impl/unicode_class.mojo"
OUT = "tokenizer/impl/unicode_table_generated.mojo"


def ranges(prefix):
    out = []
    start = None
    for cp in range(0x110000):
        hit = unicodedata.category(chr(cp)).startswith(prefix)
        if hit and start is None:
            start = cp
        elif not hit and start is not None:
            out.append((start, cp - 1))
            start = None
    if start is not None:
        out.append((start, 0x10FFFF))
    return out


def table_text():
    """The class table as text: one `# unicodedata <version>` header line,
    then `class<TAB>first_hex<TAB>last_hex` lines, ascending and disjoint
    within a class."""
    lines = ["# unicodedata %s" % unicodedata.unidata_version]
    for name, rs in (("L", ranges("L")), ("N", ranges("N")), ("WS", WHITE_SPACE)):
        for first, last in rs:
            lines.append("%s\t%04X\t%04X" % (name, first, last))
    return "\n".join(lines) + "\n"


def pins():
    with open(PIN_SOURCE, "r", encoding="utf-8") as fh:
        src = fh.read()
    version = re.search(r'^comptime UNICODE_VERSION_PINNED = "([^"]+)"', src, re.M)
    sha = re.search(r'^comptime UNICODE_TABLE_SHA256_PINNED = "([0-9a-f]{64})"', src, re.M)
    if not version or not sha:
        raise SystemExit(f"gen_unicode_categories: {PIN_SOURCE} carries no UNICODE_VERSION_PINNED / "
                         "UNICODE_TABLE_SHA256_PINNED")
    return version.group(1), sha.group(1)


def mojo_literal(text):
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\t", "\\t").replace("\n", "\\n") + '"'


def main(argv):
    text = table_text()
    sha = hashlib.sha256(text.encode("ascii")).hexdigest()
    if "--print-sha" in argv:
        print("unicodedata %s sha256 %s (%d lines)" % (unicodedata.unidata_version, sha, text.count("\n")))
        return 0
    want_version, want_sha = pins()
    if unicodedata.unidata_version != want_version:
        sys.stderr.write(
            "gen_unicode_categories: this Python (%s) has unicodedata %s; the tokenizer pins %s. "
            "Run through tokenizer/tools/gen_unicode_table.sh, which finds a matching Python.\n"
            % (sys.version.split()[0], unicodedata.unidata_version, want_version))
        return 3
    if sha != want_sha:
        sys.stderr.write("gen_unicode_categories: generated sha256 %s is not the pinned %s; refusing to write %s\n"
                         % (sha, want_sha, OUT))
        return 3
    body = (
        "# SPDX-License-Identifier: Apache-2.0\n"
        "# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632\n"
        "# GENERATED at build time by tokenizer/tools/gen_unicode_categories.py from\n"
        "# Python's unicodedata. Not tracked, do not edit.\n"
        "\n"
        'comptime UNICODE_TABLE_VERSION = "%s"\n'
        'comptime UNICODE_TABLE_SHA256 = "%s"\n'
        "comptime UNICODE_TABLE_TEXT = %s\n" % (unicodedata.unidata_version, sha, mojo_literal(text))
    )
    try:
        with open(OUT, "r", encoding="utf-8") as fh:
            if fh.read() == body:
                print("%s: up to date (unicodedata %s, sha256 %s)" % (OUT, want_version, sha[:16]))
                return 0
    except OSError:
        pass
    tmp = "%s.%d.tmp" % (OUT, os.getpid())
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(body)
    os.replace(tmp, OUT)
    print("%s: written (unicodedata %s, sha256 %s)" % (OUT, want_version, sha[:16]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
