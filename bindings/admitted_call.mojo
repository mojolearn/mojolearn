# SPDX-License-Identifier: Apache-2.0
# One admission record per public call (plan NATIVE_ENTRY_POINTS_AFTER_0866,
# workstream E, lane ne1-e-admitted-call, 2026-11-06).
#
# A public door (classical_c, the family lifecycle doors, the public attribute
# entry) builds ONE AdmittedCall on its stack and passes it by reference
# (`mut call: AdmittedCall`, or `call: AdmittedCall` for read-only helpers) to
# every helper of that operation. Helpers read what the door already admitted
# instead of admitting it again:
#
#   modules         the operation's module connections (NativeModules: the
#                   full native_loaded_module() check once per module per
#                   operation, again after every GIL release window).
#   binding         the provider module resolved for this call, with its
#                   numeric mode and capability words (estimator_binding_*
#                   resolve it once per (name, mode, checked) per call).
#   fields          the model's instance dictionary admitted once, plus a
#                   lazily filled fixed array of fitted fields by index.
#   inputs          per admitted argument: address, shape, strides, item code,
#                   read-only flag, sparse code and the owner that keeps the
#                   buffer alive.
#   layout_checked  the class layout check of this call already ran;
#   layout_plan     the connected public plan that check admitted (None when
#                   the class has no installer metadata). Added for
#                   classical_attributes (row N259).
#
# Rules:
# - Never stored on a Python object and never kept past its operation: no
#   field survives the door's return. Nothing here is a cache across calls
#   (cross-call records live in bindings/native_model_records.mojo).
# - After every GIL release window the door (or the helper that released)
#   calls released(): the modules, the binding and the layout verdict are
#   re-admitted on next use; the inputs keep their owners (the buffers stay
#   pinned) and the fields keep the same live instance dictionary object.
# - A helper that needs something this struct does not carry adds a field
#   here (one shared record, plan lesson 8), never a private per-family copy.
# - No numerics: the same bytes reach the same kernels.
# Kept free of heavy imports on purpose: every group imports this module.
from std.python import Python, PythonObject
from bindings.native_loaded_module import NativeModules


# Item codes of an admitted input (the element type the admitting helper
# verified). Unknown means "admitted as a Python value, not a typed buffer".
comptime ADMITTED_ITEM_UNKNOWN = 0
comptime ADMITTED_ITEM_F32 = 1
comptime ADMITTED_ITEM_F64 = 2
comptime ADMITTED_ITEM_I32 = 3
comptime ADMITTED_ITEM_I64 = 4
comptime ADMITTED_ITEM_BOOL = 5
comptime ADMITTED_ITEM_U8 = 6
comptime ADMITTED_ITEM_F16 = 7
comptime ADMITTED_ITEM_OTHER = 15

# Sparse codes of an admitted input.
comptime ADMITTED_DENSE = 0
comptime ADMITTED_SPARSE_CSR = 1
comptime ADMITTED_SPARSE_CSC = 2
comptime ADMITTED_SPARSE_COO = 3
comptime ADMITTED_SPARSE_OTHER = 4
comptime ADMITTED_GRAPH_TUPLE = 5   # an (indptr, indices, n) adjacency tuple

# Number of fitted-field slots a door may fill by index. Families number
# their fitted fields 0..ADMITTED_FITTED_SLOTS-1 in their own door.
comptime ADMITTED_FITTED_SLOTS = 16


struct AdmittedInput(Copyable, Movable):
    """One admitted argument of a public call."""
    var position: Int        # index in the public argument tuple (or owner tuple)
    var source: PythonObject # the caller's object
    var owner: PythonObject  # the admitted object that keeps the buffer alive
    var address: Int         # first element (0 when not a typed buffer)
    var shape: List[Int]
    var strides: List[Int]   # in elements; empty when C-contiguous is all that was checked
    var item_code: Int
    var read_only: Bool
    var sparse_code: Int

    def __init__(out self, position: Int, source: PythonObject, owner: PythonObject):
        self.position = position
        self.source = source
        self.owner = owner
        self.address = 0
        self.shape = List[Int]()
        self.strides = List[Int]()
        self.item_code = ADMITTED_ITEM_UNKNOWN
        self.read_only = False
        self.sparse_code = ADMITTED_DENSE

    def rank(self) -> Int:
        return len(self.shape)

    def rows(self) -> Int:
        # -1 when no shape was admitted.
        return self.shape[0] if len(self.shape) > 0 else -1

    def columns(self) -> Int:
        # A rank-1 input has one column, as labels_shape_of reports it.
        if len(self.shape) >= 2:
            return self.shape[1]
        return 1 if len(self.shape) == 1 else -1


struct AdmittedCall(Movable):
    """The admission record of one public call. See the file header."""
    var modules: NativeModules
    # binding: the provider resolved last; the small lists hold every
    # (name, mode, checked) resolution of this call (a GP classifier resolves
    # its own binding and the base binding: two rows, never four lookups).
    var binding: PythonObject
    var binding_mode_word: Int     # 1 identical, 0 fast, -1 not resolved
    var binding_capability: Int    # provider capability bits; 0 when unread
    var _binding_names: List[String]
    var _binding_modes: List[String]
    var _binding_checked: List[Bool]
    var _bindings: List[PythonObject]
    var fields: PythonObject       # the model's instance dict, None until admitted
    var fields_admitted: Bool
    var _fitted: List[PythonObject]
    var _fitted_filled: List[Bool]
    var inputs: List[AdmittedInput]
    var layout_checked: Bool
    var layout_plan: PythonObject

    def __init__(out self):
        self.modules = NativeModules()
        self.binding = PythonObject(None)
        self.binding_mode_word = -1
        self.binding_capability = 0
        self._binding_names = List[String]()
        self._binding_modes = List[String]()
        self._binding_checked = List[Bool]()
        self._bindings = List[PythonObject]()
        self.fields = PythonObject(None)
        self.fields_admitted = False
        self._fitted = List[PythonObject]()
        self._fitted_filled = List[Bool]()
        for _ in range(ADMITTED_FITTED_SLOTS):  # small-loop(fixed fitted slot count)
            self._fitted.append(PythonObject(None))
            self._fitted_filled.append(False)
        self.inputs = List[AdmittedInput]()
        self.layout_checked = False
        self.layout_plan = PythonObject(None)

    # --- modules -----------------------------------------------------------
    def module(mut self, name: String) raises -> PythonObject:
        return self.modules.module(name)

    def builtins(mut self) raises -> PythonObject:
        return self.modules.module("builtins")

    def released(mut self):
        # A GIL release window ended: module connections, the resolved
        # bindings and the layout verdict are admitted again on next use.
        self.modules.released()
        self.binding = PythonObject(None)
        self.binding_mode_word = -1
        self.binding_capability = 0
        self._binding_names.clear()
        self._binding_modes.clear()
        self._binding_checked.clear()
        self._bindings.clear()
        self.layout_checked = False
        self.layout_plan = PythonObject(None)

    # --- binding -----------------------------------------------------------
    def find_binding(self, name: String, mode: String, checked: Bool) -> Int:
        # small-loop: one or two provider resolutions per public call.
        for i in range(len(self._binding_names)):
            if (self._binding_names[i] == name and self._binding_modes[i] == mode
                    and (self._binding_checked[i] or not checked)):
                return i
        return -1

    def binding_at(self, index: Int) -> PythonObject:
        return self._bindings[index]

    def remember_binding(mut self, name: String, mode: String, checked: Bool, module: PythonObject):
        self._binding_names.append(name)
        self._binding_modes.append(mode)
        self._binding_checked.append(checked)
        self._bindings.append(module)
        self.binding = module
        self.binding_mode_word = 1 if mode == "identical" else (0 if mode == "fast" else -1)

    # --- fields ------------------------------------------------------------
    def admit_fields(mut self, fields: PythonObject):
        # The door's one admitted instance dictionary (sequential_fields /
        # mode_metadata_native._fields / _instance_fields). The object is the
        # live dict, so a nested set_params is still observed through it.
        self.fields = fields
        self.fields_admitted = True

    def fitted_ready(self, slot: Int) -> Bool:
        return slot >= 0 and slot < ADMITTED_FITTED_SLOTS and self._fitted_filled[slot]

    def fitted(self, slot: Int) -> PythonObject:
        return self._fitted[slot]

    def set_fitted(mut self, slot: Int, value: PythonObject) raises:
        if slot < 0 or slot >= ADMITTED_FITTED_SLOTS:
            raise Error("RuntimeError: admitted fitted-field slot out of range")
        self._fitted[slot] = value
        self._fitted_filled[slot] = True

    # --- inputs ------------------------------------------------------------
    def add_input(mut self, var admitted: AdmittedInput) -> Int:
        # Replaces an earlier admission of the same position (a door that
        # re-admits after a GIL window keeps one row per argument).
        for i in range(len(self.inputs)):  # small-loop(admitted public arguments)
            if self.inputs[i].position == admitted.position:
                self.inputs[i] = admitted^
                return i
        self.inputs.append(admitted^)
        return len(self.inputs) - 1

    def find_input(self, position: Int) -> Int:
        for i in range(len(self.inputs)):  # small-loop(admitted public arguments)
            if self.inputs[i].position == position:
                return i
        return -1

    def find_source(self, source: PythonObject) -> Int:
        # The admitted row whose caller object or admitted owner IS `source`.
        for i in range(len(self.inputs)):  # small-loop(admitted public arguments)
            if self.inputs[i].source is source or self.inputs[i].owner is source:
                return i
        return -1

    def input_shape(self, index: Int) -> List[Int]:
        return self.inputs[index].shape.copy()


def admitted_input_dense(position: Int, source: PythonObject, owner: PythonObject, address: Int,
                         rows: Int, columns: Int, item_code: Int, read_only: Bool) -> AdmittedInput:
    """An admitted dense matrix row (columns < 0: a rank-1 input)."""
    var admitted = AdmittedInput(position, source, owner)
    admitted.address = address
    admitted.shape.append(rows)
    if columns >= 0:
        admitted.shape.append(columns)
    admitted.item_code = item_code
    admitted.read_only = read_only
    return admitted^


def admitted_input_shape(position: Int, source: PythonObject, owner: PythonObject,
                         shape: PythonObject, sparse_code: Int) raises -> AdmittedInput:
    """An admitted row from a native shape tuple (exact ints, already checked
    by the admitting helper)."""
    var admitted = AdmittedInput(position, source, owner)
    for i in range(len(shape)):  # small-loop(rank of one admitted shape)
        admitted.shape.append(Int(py=shape[i]))
    admitted.sparse_code = sparse_code
    return admitted^
