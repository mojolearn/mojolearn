# SPDX-License-Identifier: Apache-2.0
# Compile-only retained complete public corpus C-entry closure; never execute.
from std.python._cpython import PyObjectPtr
from bindings.lm_corpus_public_native import lm_corpus_public_c


@export
def peer3_corpus_public_compile(owner: PyObjectPtr, arguments: PyObjectPtr) abi("C") -> PyObjectPtr:
    return lm_corpus_public_c(owner, arguments)
