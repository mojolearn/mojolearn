# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Build-path time limits for Python, read from tools/release_limits.sh, the
one file where they are written (see its header)."""
import re
from pathlib import Path

PATH = Path(__file__).resolve().parent / "release_limits.sh"
_LINE = re.compile(r"^([A-Z][A-Z0-9_]*)=([0-9]+)\s*$")


def load(path=PATH):
    """{NAME: int} for every NAME=integer line; anything else that is not a
    comment or blank is refused, so the file cannot grow logic Python skips."""
    out = {}
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        m = _LINE.match(s)
        if not m:
            raise ValueError(f"{path}:{n}: not NAME=integer: {line!r}")
        out[m.group(1)] = int(m.group(2))
    return out


LIMITS = load()


def get(name):
    return LIMITS[name]
