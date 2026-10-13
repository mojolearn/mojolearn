# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The README as the wheels' long description, with repository-relative links made absolute.

PyPI renders the README outside the repository, so `[x](docs/VERIFY.md)` resolves to nothing there.
Both wheel builders call this: packaging/linux/pack_wheel.py imports `pypi_readme`, and
packaging/macos/build_release_wheel.sh runs `python3 tools/pypi_readme.py README.md python/README.md`.
Release tooling only; never imported by the mojolearn runtime.

    python3 tools/pypi_readme.py <README.md> <out.md>
"""
import re
import sys

REPO_URL = "https://github.com/mojolearn/mojolearn/blob/main/"

# `](target)` where the target is not absolute (scheme, //, mailto:) and not an in-page #anchor.
_LINK = re.compile(r"\]\((?![a-zA-Z][a-zA-Z0-9+.-]*:|//|#)([^)\s]+)\)")


def pypi_readme(text):
    """Return `text` with every relative Markdown link pointed at the repository on GitHub."""
    return _LINK.sub(lambda m: "](" + REPO_URL + re.sub(r"^(\./)+", "", m.group(1)) + ")", text)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: pypi_readme.py <README.md> <out.md>")
    with open(sys.argv[1], encoding="utf-8") as src:
        converted = pypi_readme(src.read())
    with open(sys.argv[2], "w", encoding="utf-8") as dst:
        dst.write(converted)
