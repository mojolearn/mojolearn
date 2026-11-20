# SPDX-License-Identifier: Apache-2.0
"""A built-in schedule's native rate block: `lr_at(t)` and `bits_at(t)` as
raw C methods (METH_VARARGS | METH_KEYWORDS) of a native owner, bound once
at construction.

lane fb2-lr (2026-11-17). Board rows algos/lr-* time 100,000 single
`schedule.lr_at(t)` calls. 0.8.55 served them from a 256-step block that
Python kept in a list (lr-constant 8.8 ms); the measured wheel built a
tuple, ran the router's string dispatch, re-admitted the schedule class and
every stored field, re-read the spec from the instance dictionary, allocated
a one-element list, released and re-took the GIL and filled ONE step per
call (lr-constant 7,538 ms). Here the schedule's construction (and
`from_config`, and unpickling) builds this owner once from the admitted
spec and stores its two bound methods in the schedule's instance
dictionary, where they shadow the class methods. A call is then CPython's
call of a builtin method: a one-item argument tuple (CPython's, from its
freelist), one exact-int check (a type pointer compare), PyLong_AsLongLong,
an index into the block, PyFloat_FromDouble. No lookup, admission walk,
allocation of ours, lock or GIL switch per call. `t=` keywords and wrong
arities take a separate path with the former method's messages.
Editing a setting (`sched.peak_lr = x`) rebinds the block through
`_LrTable.__setattr__`, a native descriptor (optimizer_native_router_door
`schedule_attribute_c`; one native call per edit, never per step).

Block size: `LR_BLOCK` (256), the run length `lr_spec_fill` already uses for
one table pass, so one refill is exactly one kernel run (one set of its
scratch allocations, one GIL release) and is spent over the next 255 calls
of a forward loop. Blocks are aligned to (k * 256 + 1) so forward, backward
and random steps all land in the block that holds them; a random step costs
one 256-step run (a few microseconds), a sequential one 1/256 of it. The
size comes from the kernel's run, not from any board call count.

Bits: every value is `lr_spec_fill`'s, which defines each step's float32
pointwise (the exact rational rounded once; the table answers only where it
provably rounds the same), so a block fill and a one-step fill give the same
bits. A failing step (overflow, rounding straddle, OneCycleLR past
total_steps + 1, a step below 1) raises exactly as the one-step route did,
only when that step is asked for.

The spec is fixed at construction, as the schedule's documented settings
are; it is not re-read from the instance dictionary per call.
"""
from std.memory import bitcast
from std.python import Python, PythonObject
from std.python._cpython import GILReleased, PyObjectPtr
from std.python.bindings import PythonModuleBuilder, lookup_py_type_object
from bindings.py_cpython_table import CPythonTable, cpython_table
from bindings.py_balanced import py_fields, py_setitem, py_type
from bindings.py_type_registry import py_type_absent, py_type_attach
from bindings.optimizer_public_words_unpack import optimizer_python_error, stash_python_error
from bindings.optimizer_scalar_boundary_unpack import optimizer_integer
from bindings.optimizer_public_door import schedule_spec_admitted
from sequence.lr_table import LrSpec, LR_BLOCK, LR_ONE_BASED, lr_spec_fill
from training.optimizer_schedule_plan import optimizer_schedule_refusal

#: the last step an aligned block may start below without `first + LR_BLOCK`
#: overflowing; larger steps take the one-step fill
comptime _LR_BLOCK_LAST_ALIGNED = (1 << 62)


struct LrScheduleBlock(Movable, Writable):
    """One schedule's admitted spec and its current block of float32 rates
    (steps `first` .. `first + count - 1`; `failed` is the refusal status of
    step `first + count` when the fill stopped there, else 0)."""

    var spec: LrSpec
    var style: Int
    var first: Int
    var count: Int
    var failed: Int
    var rates: List[Float32]
    # Resolved once at construction: the exact int class (the one-word
    # admission of a step), the process CPython table (no global lookup per
    # call) and the optimizer context (admission of any other step type).
    var int_class: PythonObject
    # Untracked origin: a struct field may not expose AnyOrigin (Mojo 1.2).
    var table: UnsafePointer[CPythonTable, MutUntrackedOrigin]
    var context: PythonObject

    def __init__(out self, spec: LrSpec, style: Int, int_class: PythonObject,
                 table: UnsafePointer[CPythonTable, MutAnyOrigin], context: PythonObject):
        self.spec = spec
        self.style = style
        self.first = 0
        self.count = 0
        self.failed = 0
        self.rates = List[Float32]()
        self.int_class = int_class
        self.table = table.unsafe_origin_cast[MutUntrackedOrigin]()
        self.context = context

    def write_to(self, mut writer: Some[Writer]):
        writer.write("LrScheduleBlock(", self.spec.kind, ", ", self.first, ", ", self.count, ")")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)

    @always_inline
    def rate(mut self, t: Int) raises -> Float32:
        """The float32 rate of ONE-BASED step t."""
        var i = t - self.first
        if self.first > 0 and i >= 0 and i < self.count:
            return self.rates[i]
        return self._miss(t)

    @no_inline
    def _miss(mut self, t: Int) raises -> Float32:
        if t < 1:
            raise optimizer_schedule_refusal(LR_ONE_BASED, t, self.style, self.spec.c)
        if t <= _LR_BLOCK_LAST_ALIGNED:
            var first = ((t - 1) // LR_BLOCK) * LR_BLOCK + 1
            if first != self.first:
                # One kernel run into fresh storage with the GIL released;
                # the block is replaced only after the GIL is held again, so
                # a concurrent caller never reads a half-filled block.
                var rates = List[Float32](length=LR_BLOCK, fill=0)
                var spec = self.spec
                var status: Tuple[Int, Int] = (0, 0)
                with GILReleased(Python()):
                    status = lr_spec_fill(spec, first, LR_BLOCK,
                        rates.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]())
                self.rates = rates^
                self.first = first
                self.count = status[1]
                self.failed = status[0]
            var i = t - self.first
            if i < self.count:
                return self.rates[i]
            if i == self.count and self.failed != 0:
                raise optimizer_schedule_refusal(self.failed, t, self.style, self.spec.c)
        # Past a refused step inside the block, or beyond the aligned range:
        # the one-step fill decides this step alone.
        var one = List[Float32](length=1, fill=0)
        var single = lr_spec_fill(self.spec, t, 1,
            one.unsafe_ptr().unsafe_mut_cast[True]().unsafe_origin_cast[MutUntrackedOrigin]())
        if single[0] != 0:
            raise optimizer_schedule_refusal(single[0], t, self.style, self.spec.c)
        return one[0]

    @no_inline
    def admit(self, arg: PyObjectPtr) raises -> Int:
        """Any step that is not an exact int in machine range: the public
        admission and conversion the router path applied (exact scalar
        types only, then `PyNumber_Long`)."""
        var value = PythonObject(from_borrowed=arg)
        var step = optimizer_integer(value, self.context)
        _ = value
        return step


@no_inline
def _block_argument(table: UnsafePointer[CPythonTable, MutUntrackedOrigin], args: PyObjectPtr,
    kwargs: PyObjectPtr, name: StaticString) raises -> PyObjectPtr:
    """The one step argument of a keyword call or a wrong arity, with the
    messages the former Python method `_LrTable.<name>(self, t)` raised.
    Runs only when the call is not one positional argument."""
    var where = "_LrTable." + String(name) + "()"
    var given = table[].PyTuple_Size(args)
    if given > 1:
        raise Error("TypeError: " + where + " takes 2 positional arguments but " + String(given + 1) + " were given")
    var found = table[].PyTuple_GetItem(args, 0) if given == 1 else PyObjectPtr()
    if kwargs:
        var position = List[Int](length=1, fill=0)
        var pair = List[PyObjectPtr](length=2, fill=PyObjectPtr())
        while table[].PyDict_Next(kwargs, Int(position.unsafe_ptr()),
                Int(pair.unsafe_ptr()), Int(pair.unsafe_ptr() + 1)):  # driver-loop: over the call's keyword arguments (at most one is valid)
            var key = String(py=PythonObject(from_borrowed=pair[0]))
            if key != "t":
                raise Error("TypeError: " + where + " got an unexpected keyword argument '" + key + "'")
            if found:
                raise Error("TypeError: " + where + " got multiple values for argument 't'")
            found = pair[1]
    if not found:
        raise Error("TypeError: " + where + " missing 1 required positional argument: 't'")
    return found


@always_inline
def _block_rate(py_self: PyObjectPtr, args: PyObjectPtr, kwargs: PyObjectPtr,
    name: StaticString) raises -> Float32:
    # CPython passes this type's own instance as self (a method of the
    # registered type, bound to its owner at construction).
    var owner = PythonObject(from_borrowed=py_self)
    var block = owner.unchecked_downcast_value_ptr[LrScheduleBlock]()
    var table = block[].table
    var arg: PyObjectPtr
    if not kwargs and table[].PyTuple_Size(args) == 1:
        arg = table[].PyTuple_GetItem(args, 0)
    else:
        arg = _block_argument(table, args, kwargs, name)
    ref cpy = Python().cpython()
    var t: Int
    if PyObjectPtr(upcast_from=cpy.Py_TYPE(arg)) == block[].int_class._obj_ptr:
        var word = table[].PyLong_AsLongLong(arg)
        if word == -1 and cpy.PyErr_Occurred():
            # Out of machine range: the general admission raises as before.
            table[].PyErr_Clear()
            t = block[].admit(arg)
        else:
            t = Int(word)
    else:
        t = block[].admit(arg)
    var rate = block[].rate(t)
    _ = owner
    return rate


# METH_VARARGS | METH_KEYWORDS (PyCFunctionWithKeywords): the stdlib's only
# keyword-accepting raw method form (std/python/bindings.mojo, Mojo 1.2:
# "METH_FASTCALL | METH_KEYWORDS ... is not implemented here yet"). CPython
# packs the positional arguments into a tuple per call; docs/MODULAR_ASKS.md
# asks for the fastcall-with-keywords form, which removes that tuple.
def lr_block_at_c(py_self: PyObjectPtr, args: PyObjectPtr, kwargs: PyObjectPtr) abi("C") -> PyObjectPtr:
    """`schedule.lr_at(t)`: the float32 rate of ONE-BASED step t, as a float."""
    try:
        var rate = _block_rate(py_self, args, kwargs, "lr_at")
        ref cpy = Python().cpython()
        return cpy.PyFloat_FromDouble(Float64(rate))
    except error:
        return optimizer_python_error(error)


def lr_block_bits_c(py_self: PyObjectPtr, args: PyObjectPtr, kwargs: PyObjectPtr) abi("C") -> PyObjectPtr:
    """`schedule.bits_at(t)`: `lr_at(t)` as its float32 bit pattern."""
    try:
        var rate = _block_rate(py_self, args, kwargs, "bits_at")
        ref cpy = Python().cpython()
        return cpy.PyLong_FromSsize_t(Int(bitcast[DType.uint32](rate)))
    except error:
        return optimizer_python_error(error)


def schedule_bind_block(owner: PythonObject, kind: Int, context: PythonObject) raises:
    """PACK: build the schedule's native block from its admitted spec and
    store the block's bound `lr_at` / `bits_at` in the schedule's instance
    dictionary (they shadow the class methods). Called once per
    construction, `from_config` and unpickling; never per step."""
    var spec = schedule_spec_admitted(owner)
    var style = 0 if kind < 3 else (1 if kind < 5 else 2)
    var table = cpython_table()
    var int_class = py_type(PythonObject(0))
    var block = PythonObject(alloc=LrScheduleBlock(spec, style, int_class, table, context))
    var fields = py_fields(owner)
    for name in ["lr_at", "bits_at"]:  # driver-loop: over the two bound native methods of one schedule
        var key = PythonObject(name)
        var pointer = table[].PyObject_GenericGetAttr(block._obj_ptr, key._obj_ptr)
        if not pointer:
            var failure = stash_python_error()
            raise failure^
        py_setitem(fields, key, PythonObject(from_owned=pointer))
    _ = block


def register_lr_schedule_block(mut module: PythonModuleBuilder) raises:
    """Register the block type once in this binding (its only owner); reuse
    the canonical type when another package alias initializes."""
    var existing = PythonObject(None)
    try:
        existing = lookup_py_type_object[LrScheduleBlock]()
    except error:
        if not py_type_absent(error):
            raise error
        _ = (
            module.add_type[LrScheduleBlock]("_LrScheduleBlock")
            .def_py_c_method(lr_block_at_c, "lr_at",
                "lr_at($self, t)\n--\n\nThe float32 learning rate of ONE-BASED step `t`, as a float.")
            .def_py_c_method(lr_block_bits_c, "bits_at",
                "bits_at($self, t)\n--\n\n`lr_at(t)` as its float32 bit pattern, for fixtures.")
        )
        return
    py_type_attach(module.module, "_LrScheduleBlock", existing)
