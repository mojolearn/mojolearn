# Compile-only fixture: the full existing metric operation owner, including
# the migrated binding selection, argument admission and finished-result paths.
from std.os import abort
from std.python import PythonObject
from std.python.bindings import PythonModuleBuilder
from bindings.metrics_cls_native import metrics_cls_operation

def metrics_public_compile(model: PythonObject, operation: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    return metrics_cls_operation(model, String(operation), args, context)

@export
def PyInit__mojolearn_metrics_public_compile() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_metrics_public_compile")
        m.def_function[metrics_public_compile]("metrics_public_operation")
        return m.finalize()
    except error:
        abort(String(error))
