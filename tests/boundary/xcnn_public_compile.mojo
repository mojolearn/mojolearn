# Compile-only fixture: actual CNN public dispatcher, no module import/execution.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.xcnn_native import xcnn_c
from bindings.xseq_neural_native import xseq_labels, xseq_take
from bindings.xcnn_bn_native import xcnn_bn_backward_dev

def bn_device_compile(args: PythonObject) raises -> PythonObject:
    xcnn_bn_backward_dev(args[0], args[1], args[2], args[3], args[4], Bool(py=args[5]), args[6])
    return PythonObject(None)

def labels_compile(args: PythonObject) raises -> PythonObject:
    return xseq_labels(args[0], args[1], args[2])

def take_compile(args: PythonObject) raises -> PythonObject:
    return xseq_take(args[0], args[1], args[2])

@export
def PyInit__mojolearn_xcnn_public_compile() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_xcnn_public_compile")
        m.def_py_c_function(xcnn_c, "xcnn_native_call")
        m.def_function[bn_device_compile]("bn_backward_dev")
        m.def_function[labels_compile]("labels")
        m.def_function[take_compile]("take")
        return m.finalize()
    except error:
        abort(String(error))
