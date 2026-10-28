# SPDX-License-Identifier: Apache-2.0
# Focused compile of the complete public corpus door, including type registration.
# Compile only; this probe must not be imported or executed as product evidence.
from std.python import PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import PythonModuleBuilder
from bindings.lm_corpus_public_native import lm_corpus_public_c, LmCorpusIteratorHandle


@export
def peer3_corpus_compile(py_self: PyObjectPtr, args: PyObjectPtr) abi("C") -> PyObjectPtr:
    return lm_corpus_public_c(py_self, args)


@export
def PyInit_peer3_corpus_compile() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("peer3_corpus_compile")
        module.def_py_c_function(lm_corpus_public_c, "lm_corpus_public_call")
        _ = module.add_type[LmCorpusIteratorHandle]("_LmCorpusIteratorHandle")
        return module.finalize()
    except:
        return PythonObject(None)
