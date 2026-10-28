# Compile-only snapshot and gather owner fixture; never loaded or executed.
from std.os import abort
from std.python import Python, PythonObject
from std.python.bindings import PythonModuleBuilder
from providers.python.resample_rows_pyapi import ResampleRows, resample_rows_indices, resample_rows_gpu, resample_rows_narrow
from bindings.classical_attributes import classical_connect_numpy

def rows_compile(source: PythonObject, binding: PythonObject, context: PythonObject) raises -> PythonObject:
    classical_connect_numpy(context)
    var rows = List[ResampleRows]()
    rows.append(ResampleRows(source, context))
    var indices = resample_rows_indices(binding, rows[0].n, 2, True, PythonObject(0), context)
    var output = resample_rows_gpu(rows, binding, rows[0].n, 2, True, PythonObject(0), context)
    if output is None:
        output = resample_rows_narrow(rows, binding, rows[0].n, 2, PythonObject(0), context)
    return rows[0].take(indices, context) if output is None else output

@export
def PyInit__mojolearn_resample_rows_compile() abi("C") -> PythonObject:
    try:
        var m = PythonModuleBuilder("_mojolearn_resample_rows_compile")
        m.def_function[rows_compile]("rows")
        return m.finalize()
    except error:
        abort(String(error))
