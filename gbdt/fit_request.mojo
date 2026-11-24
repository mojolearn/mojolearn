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
  XG_OUTCOME_SIZES  addrs [outcome, out9]: writes [text bytes,
                    best_iteration, stopped_early, n_learn, n_test, text
                    address, loop iterations, loop nanoseconds, loop
                    recorded]; the text address stays valid until
                    XG_OUTCOME_FREE, and the caller copies the text from it
                    once (same process). The last three are the fit's
                    `fit_loop_` (core/fit_loop_clock.mojo; v2, lane
                    ab-loop-mode: v1 wrote six words).
  XG_OUTCOME_COPY   addrs [outcome, text (UInt8, text bytes) or 0 to skip
                    the text, learn (Float64, n_learn), test (Float64,
                    n_test)].
  XG_OUTCOME_FREE   addrs [outcome].
  XG_MODEL_DIM      addrs [text]; params [text bytes]; returns the model's
                    approx dimension.
  XG_PREDICT        addrs [text, x, dst]; params [text bytes, n_rows]: raw
                    approxes of a one-dimensional model, x column-major
                    float32, dst float32 n_rows. Returns the rows written.
  XG_PREDICT_MULTI  addrs [text, x, dst]; params [text bytes, n_rows, mode]
                    (mode 0 RAW, 1 SOFTMAX, 2 SIGMOID), dst float32 row-major.
                    Returns the width written per row.
  XG_BINARY         addrs [raw, dst]; params [n, probabilities]: the
                    adapters' binary transforms of n float32 margins, the
                    float32 pair [1 - p, p] (probabilities 1, dst 2 n) or the
                    int32 class code (0, dst n). Returns the values written.
  XG_SIGMOID_PAIR   addrs [raw, dst]; params [n]: float64 raw scores to the
                    float64 pair [1 - p, p]. Returns n.
  XG_RESIDENT_PREPARE  addrs [text]; params [text bytes]: parse, pack and
                    upload the model once; returns the handle of the image's
                    registry entry (released by the image's
                    `gbdt_resident_release`). Images with XG_CAP_RESIDENT only.
  XG_RESIDENT_PREDICT  addrs [x, dst]; params [handle, n_rows, mode,
                    row_major] (modes 0..2 as XG_PREDICT_MULTI, 3 the float64
                    sigmoid pair, 4..6 int64 class codes). Returns the width
                    written per row. Images with XG_CAP_RESIDENT only.
  XG_ORDERED_FIT    addrs [x, y, weights, permutation]; ints [n_rows,
                    n_features, n_weights, n_permutation, n_estimators,
                    max_depth, border_count, 0, 0]; floats [7] learning_rate,
                    [8] l2_leaf_reg. The OrderedRMSE fit; returns an outcome
                    handle whose text is the model (no curves).
  XG_FEATURE_FREQ_FIT  addrs [x, y, weights, sources]; ints [n_rows,
                    n_features, n_weights, n_sources, 0, 0, random_seed];
                    floats [4] learning_rate, [5] l2_leaf_reg. The two-level
                    FeatureFreq fit; returns an outcome handle likewise.
  XG_IMPORTANCES    addrs [text, x, weights, dst]; params [text bytes,
                    n_rows, n_input, row_major, n_weights]: the
                    `feature_importances_` of the model fitted on those rows
                    (core/gbdt_importance.mojo: the host column in the host
                    image, gbdt/gpu_importance.mojo in a GPU image), n_input
                    float32 to dst. Returns n_input (lane mf-imp6).

An image method that computes releases the GIL around its work itself (the
caller holds it); the caller owns every buffer until the synchronous call
returns.
"""
from std.memory import alloc, unsafe_memcpy
from core.fit_loop_clock import FitLoop

# v2 (lane ab-loop-mode, 2026-11-24): XG_OUTCOME_SIZES writes nine words (the
# fit loop's three after v1's six); a v1 image is refused by the table.
comptime GBDT_ENTRY_ABI = "mojolearn.trees.gbdt.v2"
#: the words XG_OUTCOME_SIZES writes
comptime XG_OUTCOME_WORDS = 9

comptime XG_CAPS = 0
comptime XG_NUMERIC_MODE = 1
comptime XG_FIT = 2
comptime XG_FIT_ROWMAJOR = 3
comptime XG_OUTCOME_SIZES = 4
comptime XG_OUTCOME_COPY = 5
comptime XG_OUTCOME_FREE = 6
comptime XG_MODEL_DIM = 7
comptime XG_PREDICT = 8
comptime XG_PREDICT_MULTI = 9
comptime XG_BINARY = 10
comptime XG_SIGMOID_PAIR = 11
comptime XG_RESIDENT_PREPARE = 12
comptime XG_RESIDENT_PREDICT = 13
comptime XG_ORDERED_FIT = 14
comptime XG_FEATURE_FREQ_FIT = 15
comptime XG_IMPORTANCES = 16
comptime XG_OP_COUNT = 17

comptime XG_CAP_ROW_MAJOR = 1
#: the image holds device-resident parsed models (XG_RESIDENT_*)
comptime XG_CAP_RESIDENT = 2

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
    learn_losses, test_losses]`) and its boosting loop (`fit_loop_`)."""

    var text: String
    var best_iteration: Int
    var stopped_early: Bool
    var learn: List[Float64]
    var test: List[Float64]
    #: the boosting loop's executed rounds and its time (core/fit_loop_clock.mojo;
    #: lane ab-loop-mode). Empty (not recorded) until the fit that built the
    #: outcome sets it after its loop; a text-only fit leaves it empty.
    var fit_loop: FitLoop

    def __init__(out self, var text: String, best_iteration: Int, stopped_early: Bool, var learn: List[Float64],
                 var test: List[Float64]):
        self.text = text^
        self.best_iteration = best_iteration
        self.stopped_early = stopped_early
        self.learn = learn^
        self.test = test^
        self.fit_loop = FitLoop.empty()


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


def gbdt_outcome_sizes(handle: Int, out9: Int) raises:
    var o = _outcome(handle)
    var dst = MutPointer[Int, MutUntrackedOrigin](unsafe_from_address=out9)
    dst[0] = o[].text.byte_length()
    dst[1] = o[].best_iteration
    dst[2] = 1 if o[].stopped_early else 0
    dst[3] = len(o[].learn)
    dst[4] = len(o[].test)
    dst[5] = Int(o[].text.unsafe_ptr()) if o[].text.byte_length() > 0 else 0
    dst[6] = o[].fit_loop.iterations
    dst[7] = o[].fit_loop.loop_ns
    dst[8] = 1 if o[].fit_loop.recorded else 0


def gbdt_outcome_copy(handle: Int, text_dst: Int, learn_dst: Int, test_dst: Int) raises:
    var o = _outcome(handle)
    var nb = o[].text.byte_length()
    if nb > 0 and text_dst != 0:
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


def gbdt_string_at(address: Int, nbytes: Int) raises -> String:
    """A String of nbytes UTF-8 bytes at address (a request string or the
    model text), built in one copy from a byte view of the span."""
    # One copy (round2-fixes, 2026-11-11; perf audit 2 A pattern 7): the
    # former default copied the span into a byte List and then the List into
    # the String (two copies of every model text and option string). The
    # experiment `MOJOLEARN_GBDT_FAST_FIT_TEXT_ONECOPY` held this one-copy
    # form behind a define only because its span constructor had not been
    # compiled yet; it is the same bytes with one copy fewer, so no measured
    # question remains and the define and the List form are deleted
    # (docs/apple-fast/EXPERIMENTS.md row). The view form is the one
    # core/kde_native_api.mojo `_kde_name` already compiles.
    if nbytes < 0 or (nbytes > 0 and address == 0):
        raise Error("gbdt: null or invalid string span")
    if nbytes == 0:
        return String()
    var view = Span(unsafe_ptr=Pointer[UInt8, ImmUntrackedOrigin](unsafe_from_address=address), length=nbytes)
    return String(StringSlice(unsafe_from_utf8=view))
