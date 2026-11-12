# SPDX-License-Identifier: Apache-2.0
"""The native arguments and outcome of `gbdt_fit`, and the gbdt binding ABI
(lane calltable-gbdt-forest-dart, 2026-11-11).

`GbdtFitArgs` is what the `gbdt_fit` Python entries' `params` / `strs` lists
held, as plain Mojo values: the six buffer addresses, the positional
parameters as an int view and a float view of equal length (with a flag
byte per position for a value that has no integer or no float reading, so a
read of a missing view refuses as the former `Int(py=)` / `Float64(py=)` did),
and the strings. `GbdtFitOutcome` is the former result list
`[model_text, best_iteration, stopped_early, learn_losses, test_losses]`.
The slot layout is documented on `bindings/_mojolearn_gbdt.mojo::gbdt_fit_native`.

THE ABI (`GBDT_ENTRY_ABI`, one `abi("C")` entry per gbdt image, published in
`core/gbdt_entry_table.mojo`; descriptor shape of
`core/native_descriptor_abi.mojo`, parameters as the int/float view pair):

  XG_CAPS           returns bit 0 when the image serves the row-major fit.
  XG_NUMERIC_MODE   returns the image's compiled tier (0 fast, 1 identical).
  XG_FIT            addrs [x, y, weights, cat_flags, eval_x, eval_y, flags,
  XG_FIT_ROWMAJOR    str_0 address, str_0 bytes, str_1 address, ...];
                    params = the fit's params list; flags = one byte per
                    parameter (`GbdtFitArgs.bad`) or 0. Returns an outcome
                    handle owned by the image until XG_OUTCOME_FREE.
  XG_OUTCOME_SIZES  addrs [outcome, out5]: writes [text bytes,
                    best_iteration, stopped_early, n_learn, n_test].
  XG_OUTCOME_COPY   addrs [outcome, text (UInt8, text bytes), learn
                    (Float64, n_learn), test (Float64, n_test)].
  XG_OUTCOME_FREE   addrs [outcome].
  XG_MODEL_DIM      addrs [text]; params [text bytes]; returns the model's
                    approx dimension.

The caller owns every buffer until the synchronous call returns.
"""
from std.memory import alloc, unsafe_memcpy
from std.sys.compile import is_defined

comptime GBDT_ENTRY_ABI = "mojolearn.trees.gbdt.v1"

comptime XG_CAPS = 0
comptime XG_NUMERIC_MODE = 1
comptime XG_FIT = 2
comptime XG_FIT_ROWMAJOR = 3
comptime XG_OUTCOME_SIZES = 4
comptime XG_OUTCOME_COPY = 5
comptime XG_OUTCOME_FREE = 6
comptime XG_MODEL_DIM = 7
comptime XG_OP_COUNT = 8

comptime XG_CAP_ROW_MAJOR = 1

#: flag bits of `GbdtFitArgs.bad`
comptime GBDT_ARG_NO_INT = 1
comptime GBDT_ARG_NO_FLOAT = 2


struct GbdtFitArgs(Movable, Sized):
    """The fit's request as plain values (see the module docstring)."""

    var x: Int
    var y: Int
    var weights: Int
    var cat_flags: Int
    var eval_x: Int
    var eval_y: Int
    var ints: List[Int]
    var floats: List[Float64]
    var bad: List[UInt8]
    var strs: List[String]

    def __init__(out self, x: Int, y: Int, weights: Int, cat_flags: Int, eval_x: Int, eval_y: Int,
                 var ints: List[Int], var floats: List[Float64], var bad: List[UInt8], var strs: List[String]) raises:
        if len(ints) != len(floats) or len(bad) != len(ints):
            raise Error("gbdt_fit: parameter views differ in length")
        self.x = x
        self.y = y
        self.weights = weights
        self.cat_flags = cat_flags
        self.eval_x = eval_x
        self.eval_y = eval_y
        self.ints = ints^
        self.floats = floats^
        self.bad = bad^
        self.strs = strs^

    def __len__(self) -> Int:
        return len(self.ints)

    def i(self, k: Int) raises -> Int:
        """Slot k as an integer (the former `Int(py=params[k])`)."""
        if k < 0 or k >= len(self.ints):
            raise Error("gbdt_fit: params index " + String(k) + " out of range")
        if self.bad[k] & GBDT_ARG_NO_INT:
            raise Error("gbdt_fit: params[" + String(k) + "] must be an integer")
        return self.ints[k]

    def f(self, k: Int) raises -> Float64:
        """Slot k as a float (the former `Float64(py=params[k])`)."""
        if k < 0 or k >= len(self.floats):
            raise Error("gbdt_fit: params index " + String(k) + " out of range")
        if self.bad[k] & GBDT_ARG_NO_FLOAT:
            raise Error("gbdt_fit: params[" + String(k) + "] must be a number")
        return self.floats[k]


struct GbdtFitOutcome(Movable):
    """The fit's result (the former `[text, best_iteration, stopped_early,
    learn_losses, test_losses]`)."""

    var text: String
    var best_iteration: Int
    var stopped_early: Bool
    var learn: List[Float64]
    var test: List[Float64]

    def __init__(out self, var text: String, best_iteration: Int, stopped_early: Bool, var learn: List[Float64],
                 var test: List[Float64]):
        self.text = text^
        self.best_iteration = best_iteration
        self.stopped_early = stopped_early
        self.learn = learn^
        self.test = test^


def gbdt_outcome_box(var outcome: GbdtFitOutcome) -> Int:
    """Move an outcome to the image's heap; the address is the handle the
    XG_OUTCOME_* operations take. Freed only by `gbdt_outcome_free` in the
    same image."""
    var box = alloc[GbdtFitOutcome](1)
    box.unsafe_write(outcome^)
    return Int(box)


def _outcome(handle: Int) raises -> UnsafePointer[GbdtFitOutcome, MutUntrackedOrigin]:
    if handle == 0:
        raise Error("gbdt fit outcome: null handle")
    return UnsafePointer[GbdtFitOutcome, MutUntrackedOrigin](unsafe_from_address=handle)


def gbdt_outcome_sizes(handle: Int, out5: Int) raises:
    var o = _outcome(handle)
    var dst = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=out5)
    dst[0] = o[].text.byte_length()
    dst[1] = o[].best_iteration
    dst[2] = 1 if o[].stopped_early else 0
    dst[3] = len(o[].learn)
    dst[4] = len(o[].test)


def gbdt_outcome_copy(handle: Int, text_dst: Int, learn_dst: Int, test_dst: Int) raises:
    var o = _outcome(handle)
    var nb = o[].text.byte_length()
    if nb > 0:
        unsafe_memcpy(dest=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=text_dst),
               src=o[].text.unsafe_ptr(), count=nb)
    if len(o[].learn) > 0:
        unsafe_memcpy(dest=MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=learn_dst),
               src=o[].learn.unsafe_ptr(), count=len(o[].learn))
    if len(o[].test) > 0:
        unsafe_memcpy(dest=MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=test_dst),
               src=o[].test.unsafe_ptr(), count=len(o[].test))


def gbdt_outcome_free(handle: Int) raises:
    var o = _outcome(handle)
    o.unsafe_deinit_pointee()
    o.free()


# A/B define (lane loops-trees-ab, 2026-11-11; perf audit 20261111 section 7):
# `-D MOJOLEARN_GBDT_FAST_FIT_TEXT_ONECOPY` builds the String in ONE copy from a
# byte view of the source span, instead of a byte List copy and then the
# String copy of that List. It halves the model-text copies of every GBDT
# fit on both sides of the table: the caller's outcome text
# (gbdt/fit_calls.mojo `xg_fit`, after the one crossing copy out of the
# image) and the image's `model_dim` read (gbdt/gbdt_entry.mojo
# XG_MODEL_DIM); request option strings take the same path. Same bytes, no
# bit change. It is a define, not a deletion, because the pointer-span
# constructor has not compiled on this toolchain yet (lanes do not compile);
# the integrator compiles it, the GBDT-fit-with-a-large-model A/B decides it,
# and a winner becomes the default (verdict pending:
# docs/apple-fast/EXPERIMENTS.md row MOJOLEARN_GBDT_FAST_FIT_TEXT_ONECOPY).
comptime _GBDT_FAST_FIT_TEXT_ONECOPY = is_defined["MOJOLEARN_GBDT_FAST_FIT_TEXT_ONECOPY"]()


def gbdt_string_at(address: Int, nbytes: Int) raises -> String:
    """A String of nbytes UTF-8 bytes at address (a request string or the
    model text)."""
    if nbytes < 0 or (nbytes > 0 and address == 0):
        raise Error("gbdt: null or invalid string span")
    comptime if _GBDT_FAST_FIT_TEXT_ONECOPY:
        if nbytes == 0:
            return String()
        var view = Span(unsafe_ptr=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address), length=nbytes)
        return String(StringSlice(unsafe_from_utf8=view))
    else:
        var bytes = List[UInt8](length=nbytes, fill=0)
        if nbytes > 0:
            unsafe_memcpy(dest=bytes.unsafe_ptr(), src=MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=address),
                          count=nbytes)
        var text = String(StringSlice(unsafe_from_utf8=Span(bytes)))
        _ = bytes
        return text^
