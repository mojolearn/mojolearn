# SPDX-License-Identifier: Apache-2.0
"""The Python-registered native owner of an arena state (lane arena,
2026-11-21).

A Python arena class (`model_selection._Resident`; `DeviceRows`,
`DeviceCache` and the resident scope follow) keeps ONE slot holding an
`ArenaOwner`: the kind of state, its handle word in the kind's registry
(`core/arena_state.mojo`) and, for the resident store owner, the
model-selection helper entry resolved once at construction. The state is
released when CPython frees the owner (its Mojo destructor), never by a
Python `__del__`.

The type is registered in ONE binding (`bindings/_mojolearn_tokenizer_host.mojo`,
`arena_owner_type`, the pattern of `bindings/native_identity_type_unpack.mojo`).
Every other image reads an owner through `arena_owner_view`: Mojo keeps one
Python type object per Mojo type per interpreter, so the downcast admits the
exact class from any image.

A door reads the slot ONCE per public operation (`arena_resident_view`): one
C slot read through the class's own member descriptor (no instance
dictionary, no attribute hook of a subclass can run), one downcast, two
words copied. Nothing here is reached per row, per batch or per token, so
there is no independent work to split across cores.
"""
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from std.python.bindings import PythonModuleBuilder, lookup_py_type_object
from bindings.native_owner_registry_pack import borrowed_native_owner
from bindings.optimizer_public_words_unpack import stash_python_error
from bindings.py_balanced import py_setattr
from bindings.py_cpython_table import cpython_table
from bindings.py_type_registry import py_type_absent
from core.arena_state import ARENA_RESIDENTS, arena_handle
from core.entry_table import EntryRef
from core.msel_helpers_calls import msel_store_free

#: The kinds of state an owner holds (one registry each).
comptime ARENA_OWNER_RESIDENT = 1


def arena_pin(value: PythonObject) -> Int:
    """Retain `value` for a native state: the pin token is its address."""
    ref cpy = Python().cpython()
    cpy.Py_IncRef(value._obj_ptr)
    return Int(value._obj_ptr)


def arena_pinned(token: Int) raises -> PythonObject:
    """The object a live pin token retains (a new reference)."""
    if not token:
        raise Error("RuntimeError: native arena state holds no owner for this entry")
    return borrowed_native_owner(token)


def arena_unpin(token: Int):
    """Drop the reference `arena_pin` took. The object's own finalizer may
    run here: call it after every native release and publication."""
    if token:
        var raw = Pointer[UInt8, MutUntrackedOrigin](unsafe_from_address=token)
        var owned = PythonObject(from_owned=PyObjectPtr(upcast_from=Optional(raw)))
        _ = owned^


def _resident_finalize(word: Int, store: EntryRef) raises:
    """Free every store id the resident state still holds, drop its pins and
    release the state. Best effort: this runs from the owner's destructor,
    where nothing can be reported; `close` is the reporting path."""
    var handle = arena_handle(word)
    var lease = ARENA_RESIDENTS.acquire(handle)
    var drained = lease[].drain()
    for i in range(len(drained)):  # driver-loop: over the inputs the owner still holds (each one small device free)
        if drained[i].store_id >= 0:
            try:
                _ = msel_store_free(store, drained[i].store_id)
            except error:
                pass
    ARENA_RESIDENTS.release(handle)
    # Owners last, after every free and the release.
    for i in range(len(drained)):  # driver-loop: over the drained entries' pins
        arena_unpin(drained[i].owner_pin)
        arena_unpin(drained[i].prepared_pin)


struct ArenaOwner(Movable, Writable):
    """The one object a Python arena class holds: `kind`, the handle `word`
    of its state and the device-store entry resolved at construction."""
    var kind: Int
    var word: Int
    var store: EntryRef

    def __init__(out self, kind: Int, word: Int, store: EntryRef):
        self.kind = kind
        self.word = word
        self.store = store

    def write_to(self, mut writer: Some[Writer]):
        writer.write("ArenaOwner(native)")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    def __del__(deinit self):
        # tp_dealloc holds the GIL; the frees are not GIL-released here.
        if self.word >= 0 and self.kind == ARENA_OWNER_RESIDENT:
            try:
                _resident_finalize(self.word, self.store)
            except error:
                pass


@fieldwise_init
struct ArenaOwnerView(ImplicitlyCopyable, Movable):
    """What a door copies out of an owner for one public operation."""
    var word: Int
    var store: EntryRef


def arena_owner_type(mut module: PythonModuleBuilder) raises:
    """Register `ArenaOwner` once per interpreter; a second initialization of
    the registering binding attaches the existing type."""
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[ArenaOwner]()
    except error:
        if not py_type_absent(error):
            raise error
        _ = module.add_type[ArenaOwner]("_ArenaOwner")
        return
    py_setattr(module.module, "_ArenaOwner", existing)


def arena_owner_new(kind: Int, word: Int, store: EntryRef) raises -> PythonObject:
    var owner = PythonObject(alloc=ArenaOwner(kind, word, store))
    return owner


def arena_owner_view(owner: PythonObject, kind: Int) raises -> ArenaOwnerView:
    """Admit `owner` as the exact native owner type of `kind` and copy its
    two words."""
    if owner is None:
        raise Error("RuntimeError: native arena owner was never initialized")
    ref native = owner.downcast_value_ptr[ArenaOwner]()[]
    if native.kind != kind or native.word < 0:
        raise Error("TypeError: native arena owner holds another kind of state")
    return ArenaOwnerView(native.word, native.store)


def arena_owner_slot(getter: PythonObject, model: PythonObject, cls: PythonObject,
                     absent: PythonObject) raises -> PythonObject:
    """The owner in `model`'s slot, read through the bound `__get__` of the
    class's own member descriptor (`getter`): a C slot read with the
    descriptor's exact type check. None when the slot was never set
    (`absent` is the AttributeError class the descriptor answers with); a
    foreign object raises the descriptor's own TypeError."""
    ref cpy = Python().cpython()
    var args = Python.tuple(model, cls)
    var result = cpy.PyObject_Call(getter._obj_ptr, args._obj_ptr, PyObjectPtr())
    _ = args
    if not result:
        if cpython_table()[].PyErr_ExceptionMatches(absent._obj_ptr) != 0:
            cpython_table()[].PyErr_Clear()
            return PythonObject(None)
        var failure = stash_python_error()
        raise failure^
    return PythonObject(from_owned=result)


def arena_resident_view(model: PythonObject, context: PythonObject) raises -> ArenaOwnerView:
    """The resident store owner state of a `model_selection._Resident`,
    resolved once for the public operation."""
    var owner = arena_owner_slot(context["arena_resident_owner_get"], model, context["arena_resident_class"],
                                 context["cache_attribute_error"])
    return arena_owner_view(owner, ARENA_OWNER_RESIDENT)
