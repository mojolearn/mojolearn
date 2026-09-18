#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Rebuild MAIN's `_byte_lm_impl.py` from this branch's, in a copy of python/.

    python3 tools/lm_byte_path_main_copy.py SRC_PYTHON_DIR DST_PYTHON_DIR

lane/tokenized-corpus changed ONE file on the byte training path,
`python/mojolearn/_byte_lm_impl.py`, by ADDING three blocks: the
`_require_schedule_vocabulary` function, its call in the constructor, and the
`data_schedule` property. This copies SRC to DST and removes exactly those
three blocks from DST's copy, then REFUSES unless the result hashes to main's
file (`MAIN_SHA256`, `git show <main>:python/mojolearn/_byte_lm_impl.py`).
A byte run under DST is then main's Python over the same built binaries, so
comparing its witness hashes with the same run under SRC shows whether the
branch moved a byte-path bit. Nothing is patched by pattern guessing: if the
removal does not land on main's exact bytes, nothing runs.
"""
import hashlib
from pathlib import Path
import shutil
import sys

#: sha256 of python/mojolearn/_byte_lm_impl.py at origin/main c7442abed..5bde47f20
#: (unchanged across that range; lane/tokenized-corpus, 2026-09-18)
MAIN_SHA256 = "d6948abc9409e1a75bdfde3aa04a193940a486bb4b9a23adb48dabd512b3f94e"


def cut(text, start, stop):
    a = text.index(start)
    b = text.index(stop, a)
    return text[:a] + text[b:]


def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    if dst.exists():
        raise SystemExit(f"{dst} exists; refusing to overwrite")
    shutil.copytree(src, dst, symlinks=True)
    p = dst / "mojolearn" / "_byte_lm_impl.py"
    t = p.read_text()
    t = cut(t, "def _require_schedule_vocabulary(descriptor, shape):", "def _schedule(value):")
    t = t.replace("        _require_schedule_vocabulary(descriptor, shape)\n", "", 1)
    t = cut(t, "    @property\n    def data_schedule(self):", "    @staticmethod\n    def parameter_registry(shape=None):")
    got = hashlib.sha256(t.encode()).hexdigest()
    if got != MAIN_SHA256:
        raise SystemExit(f"REFUSING: rebuilt _byte_lm_impl.py hashes {got}, main's is {MAIN_SHA256}")
    p.write_text(t)
    print(f"main copy OK: {p} sha256 {got}")


if __name__ == "__main__":
    main()
