# SPDX-License-Identifier: Apache-2.0
"""Operation-local parameter choices and direct Cartesian enumeration.

Public constructors retain their original value containers. Only an individual
iteration/search/length operation captures prepared choices; no cross-operation
cache hides later edits to lists, dictionaries, arrays or their metadata.
"""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.native_loaded_module_door import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_setitem
from bindings.msel_search_state_door import msel_parameter_values
from bindings.msel_parameter_grid_run import ParameterGridShape, parameter_grid_advance


def parameter_grid_dicts(param_grid_in: PythonObject, context: PythonObject,
                         capture_values: Bool = False) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var param_grid = param_grid_in
    if py_type(param_grid) is b.dict:
        param_grid = Python.list(param_grid)
    if py_type(param_grid) is not b.list and py_type(param_grid) is not b.tuple:
        raise Error("TypeError: Parameter grid requires a native dict or list of dicts; Python iterator hooks are not supported")
    var out = Python.list()
    for i in range(len(param_grid)):  # driver-loop: validate grids in original outer order
        var g = param_grid[i]
        if py_type(g) is not b.dict:
            raise Error("TypeError: Parameter grid requires native dictionaries")
        var keys = py_call(b.list, g.keys())
        var stored = Python.dict()
        for j in range(len(keys)):  # driver-loop: preserve insertion-order admission and first invalid parameter
            var key = keys[j]
            if py_type(key) is not b.str:
                raise Error("TypeError: parameter names must be exact strings")
            var values = msel_parameter_values(g[key], context)
            if len(values) == 0:
                raise Error("Parameter grid for parameter " + String(py_call(b.repr, key)) + " need to be a non-empty sequence.")
            # Constructors keep original mutable containers; native operations
            # keep this one prepared list rather than convert it again later.
            py_setitem(stored, key, values if capture_values else g[key])
        _ = py_call(out.append, stored)
    return out


struct ParameterGridRow(Movable):
    """The owners of one parameter dictionary: its names and its prepared
    choice lists (opaque to RUN)."""
    var keys: List[PythonObject]
    var choices: List[PythonObject]

    def __init__(out self, captured: PythonObject, b: PythonObject, sorted_keys: Bool,
                 mut shape: ParameterGridShape) raises:
        var names = py_call(b.sorted, captured) if sorted_keys else py_call(b.list, captured)
        self.keys = List[PythonObject]()
        self.choices = List[PythonObject]()
        var widths = List[Int]()
        for i in range(len(names)):  # driver-loop: retain exact key/choice owners once before enumeration
            var key = names[i]
            var values = captured[key]
            self.keys.append(key)
            self.choices.append(values)
            widths.append(len(values))
        shape.add(widths^)


struct PreparedParameterGrid(Movable):
    """UNPACK of a parameter grid: the owners (`rows`) and the typed shape
    RUN computes on (bindings/msel_parameter_grid_run.mojo); `item` and
    `expand` are the PACK of candidates from RUN's positions."""
    var rows: List[ParameterGridRow]
    var shape: ParameterGridShape

    def __init__(out self, source: PythonObject, context: PythonObject,
                 captured: Bool = False, sorted_keys: Bool = True) raises:
        # `captured` is exclusively for the sampler's preceding native
        # distribution admission, which already prepared these exact lists.
        var grids = source if captured else parameter_grid_dicts(source, context, True)
        var b = native_loaded_module("builtins")
        var rows = List[ParameterGridRow]()
        var shape = ParameterGridShape()
        # Finish every insertion-order validation before sorting/expansion,
        # matching the old grid_dicts -> grid_list refusal boundary.
        for i in range(len(grids)):  # driver-loop: one native schema per parameter dictionary
            rows.append(ParameterGridRow(grids[i], b, sorted_keys, shape))
        self.rows = rows^
        self.shape = shape^

    def count(self) -> Int:
        return self.shape.count()

    def item(self, index_in: Int) raises -> PythonObject:
        """`list(ParameterGrid)[index]` without expanding the grid: the
        candidate at a (negative-normalized) index by mixed-radix digits,
        rightmost parameter fastest, as `expand` emits them (lane
        perf-sweep-msel-nbr-svc: `ParameterGrid[i]` expanded every candidate
        to return one)."""
        var positions = List[Int]()
        var i = self.shape.locate(index_in, positions)
        ref row = self.rows[i]
        var candidate = Python.dict()
        for j in range(len(row.keys)):  # driver-loop: sorted parameter insertion order
            py_setitem(candidate, row.keys[j], row.choices[j][positions[j]])
        return candidate

    def expand(self) raises -> PythonObject:
        var output = Python.list()
        ref cpy = Python().cpython()
        var append = cpy.lib.get_function[Int32]("PyList_Append")
        var set_item = cpy.lib.get_function[Int32]("PyDict_SetItem")
        var get_item = cpy.lib.get_function[PyObjectPtr]("PyList_GetItem")
        for i in range(len(self.rows)):  # driver-loop: preserve outer-grid ordering
            ref row = self.rows[i]
            ref widths = self.shape.widths[i]
            var rank = len(row.keys)
            var positions = List[Int](length=rank, fill=0)
            while True:  # driver-loop: emit final dictionaries directly, without product tuples/zip/dict calls
                var candidate = Python.dict()
                for j in range(rank):  # driver-loop: sorted parameter insertion order
                    # Every choices owner is an exact, private prepared list.
                    # PyList_GetItem borrows; PyDict_SetItem retains the value.
                    var value = get_item(row.choices[j]._obj_ptr, positions[j])
                    if not value or set_item(candidate._obj_ptr, row.keys[j]._obj_ptr, value) != 0:
                        raise cpy.unsafe_get_error()
                if append(output._obj_ptr, candidate._obj_ptr) != 0:
                    raise cpy.unsafe_get_error()
                _ = candidate  # retain through the borrowed C call above
                # RUN: the odometer step, rightmost parameter fastest, as
                # itertools.product (inlined: no call per candidate).
                if not parameter_grid_advance(widths, positions):
                    break
        return output
