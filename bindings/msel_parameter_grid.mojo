# SPDX-License-Identifier: Apache-2.0
"""Operation-local parameter choices and direct Cartesian enumeration.

Public constructors retain their original value containers. Only an individual
iteration/search/length operation captures prepared choices; no cross-operation
cache hides later edits to lists, dictionaries, arrays or their metadata.
"""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.native_loaded_module import native_loaded_module
from bindings.py_balanced import py_call, py_type, py_setitem
from bindings.msel_search_state import msel_parameter_values


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
    var keys: List[PythonObject]
    var choices: List[PythonObject]
    var widths: List[Int]

    def __init__(out self, captured: PythonObject, b: PythonObject, sorted_keys: Bool) raises:
        var names = py_call(b.sorted, captured) if sorted_keys else py_call(b.list, captured)
        self.keys = List[PythonObject]()
        self.choices = List[PythonObject]()
        self.widths = List[Int]()
        for i in range(len(names)):  # driver-loop: retain exact key/choice owners once before enumeration
            var key = names[i]
            var values = captured[key]
            self.keys.append(key)
            self.choices.append(values)
            self.widths.append(len(values))


struct PreparedParameterGrid(Movable):
    var rows: List[ParameterGridRow]

    def __init__(out self, source: PythonObject, context: PythonObject,
                 captured: Bool = False, sorted_keys: Bool = True) raises:
        # `captured` is exclusively for the sampler's preceding native
        # distribution admission, which already prepared these exact lists.
        var grids = source if captured else parameter_grid_dicts(source, context, True)
        var b = native_loaded_module("builtins")
        self.rows = List[ParameterGridRow]()
        # Finish every insertion-order validation before sorting/expansion,
        # matching the old grid_dicts -> grid_list refusal boundary.
        for i in range(len(grids)):  # driver-loop: one native schema per parameter dictionary
            self.rows.append(ParameterGridRow(grids[i], b, sorted_keys))

    def count(self) -> Int:
        var total = 0
        for i in range(len(self.rows)):  # driver-loop: native cardinalities; no value rematerialization
            var count = 1
            for j in range(len(self.rows[i].widths)):  # driver-loop: dimensions of one grid
                count *= self.rows[i].widths[j]
            total += count
        return total

    def expand(self) raises -> PythonObject:
        var output = Python.list()
        ref cpy = Python().cpython()
        var append = cpy.lib.get_function[Int32]("PyList_Append")
        var set_item = cpy.lib.get_function[Int32]("PyDict_SetItem")
        var get_item = cpy.lib.get_function[PyObjectPtr]("PyList_GetItem")
        for i in range(len(self.rows)):  # driver-loop: preserve outer-grid ordering
            ref row = self.rows[i]
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
                # Rightmost parameter changes fastest, as itertools.product.
                # No multiplied product size can overflow an enumeration bound.
                var axis = rank - 1
                while axis >= 0:  # driver-loop: carry across native mixed-radix counters
                    positions[axis] += 1
                    if positions[axis] < row.widths[axis]:
                        break
                    positions[axis] = 0
                    axis -= 1
                if axis < 0:
                    break
        return output
