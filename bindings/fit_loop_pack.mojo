# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""PACK of the public `fit_loop_` attribute (lane alm-neural, 2026-11-24;
parent ab-loop-mode): the tuple (iterations, milliseconds) built from the
record words a native trainer wrote after its loop
(sequence/fit_loop_words.mojo). Mojo values in, one Python tuple out."""

from std.python import Python, PythonObject
from sequence.fit_loop_words import fit_loop_read


def fit_loop_pack(words: List[Int64]) raises -> PythonObject:
    """`fit_loop_`: (iterations as int, the loop's milliseconds as float)."""
    var loop = fit_loop_read(words)
    return Python.tuple(PythonObject(loop.iterations), PythonObject(loop.milliseconds()))
