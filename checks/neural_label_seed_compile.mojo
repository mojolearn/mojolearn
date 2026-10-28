# Focused compile-only instantiation of shared neural labels and seed admission.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.neural_labels_native import neural_encode_labels, neural_decode_labels
from bindings.xseq_native_common import XseqStream

def encode_compile(args: PythonObject) raises -> PythonObject:
    return neural_encode_labels(args[0], args[1], args[2])

def decode_compile(args: PythonObject) raises -> PythonObject:
    return neural_decode_labels(args[0], args[1], args[2])

def seed_compile(args: PythonObject) raises -> PythonObject:
    var stream = XseqStream(args[0], args[1])
    return PythonObject(Int(stream.seed & UInt64(0x7FFFFFFFFFFFFFFF)))

@export
def PyInit__mojolearn_neural_label_seed_compile() abi("C") -> PythonObject:
    try:
        var module = PythonModuleBuilder("_mojolearn_neural_label_seed_compile")
        module.def_function[encode_compile]("encode")
        module.def_function[decode_compile]("decode")
        module.def_function[seed_compile]("seed")
        return module.finalize()
    except error:
        abort(String(error))
