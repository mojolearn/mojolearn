# Compile-only fixture: whole convolution public owner; never loaded or run.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.xcnn_conv_native import xcnn_conv_layers

def convolution_public_call(layer: PythonObject, operation: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return xcnn_conv_layers(layer, String(operation), args, context)

@export
def PyInit__mojolearn_cnn_conv_compile() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_cnn_conv_compile")
        m.def_function[convolution_public_call]("convolution_public_call")
        return m.finalize()
    except error:
        abort(String(error))
