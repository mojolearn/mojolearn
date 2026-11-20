# SPDX-License-Identifier: Apache-2.0
"""RUN half of the neural block admission (lane rs-neural, wave 8 run split,
from bindings/neural_block_admission.mojo): the refusal text for a
non-float32 tensor, from its captured dtype code. Typed Mojo only, no Python
import (tools/python_free_dirs.sh). Same body."""


def _neural_dtype_refusal(name: String, code: String) -> String:
    # code is captured native Array metadata or the admitted buffer format,
    # never an arbitrary object's dtype attribute or a coercion callback.
    var label = String(code)
    if code == "<f8" or code == ">f8" or code == "d" or code == "<d" or code == ">d" or code == "=d" or code == "@d":
        label = "float64"
    elif code == "<f2" or code == ">f2" or code == "e" or code == "<e" or code == ">e" or code == "=e" or code == "@e":
        label = "float16"
    elif code == "<u2" or code == ">u2" or code == "H" or code == "<H" or code == ">H" or code == "=H" or code == "@H":
        label = "uint16 (bfloat16 bits)"
    return ("TypeError: neural " + name + " has dtype " + label + "; float32 only: bfloat16, "
            + "float16 and float64 are refused rather than converted. Convert to float32.")
