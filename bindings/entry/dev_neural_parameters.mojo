# Incremental public-operation verification group. The production tokenizer
# module registers these same compiled functions; no alternate implementation.
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.neural_block_public import neural_block_c


@export
def PyInit__mojolearn_neural_parameters() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_neural_parameters")
        module.def_py_c_function(neural_block_c[0], "neural_block_init")
        module.def_py_c_function(neural_block_c[4], "neural_block_set_params")
        return module.finalize()
    except:
        return PythonObject()
