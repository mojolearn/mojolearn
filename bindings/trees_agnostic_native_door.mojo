# SPDX-License-Identifier: Apache-2.0
# Whole KernelExplainer and PermutationExplainer operations. The xtrees
# agnostic kernels (providers/python/xtrees_api_pyapi.mojo, xtrees/agnostic*)
# own every numerical step; this owner admits the model, background, settings
# and rows, evaluates the explained mojolearn estimator through its compiled
# model-selection plan (bindings/prep_native_estimator.mojo; never its Python
# predict), drives the chunk loops and returns the shaped float64 values.
# A Python callable model is refused: native code never calls back into it.
from std.memory import unsafe_memcpy
from std.python import Python, PythonObject
from std.python._cpython import GILReleased
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.mode_metadata_native_door import _fields, _class_fields
from bindings.native_loaded_module_door import native_loaded_module
from bindings.classical_attributes_door import classical_check_layout
from bindings.prep_native_estimator_door import (
    prep_estimator_admit, prep_estimator_has, prep_estimator_call, prep_estimator_predict_internal,
    prep_estimator_plan,
)
from bindings.member_held_door import MemberHeld, member_held, member_held_release
from bindings.member_set_door import MEMBER_SET_NATIVE
from core.member_abi import (
    MemberCall, MemberMatrix, MemberPlan, member_call, member_call_of, member_outputs_of,
    MEMBER_OP_CAPABILITIES, MEMBER_OP_PREDICT, MEMBER_OP_PREDICT_PROBA, MEMBER_CAN_PREDICT, MEMBER_CAN_PREDICT_PROBA,
    MEMBER_CAN_TEST_ROWS, MEMBER_F32, MEMBER_DTYPE_UNSAID,
)
from bindings.trees_public_inputs_run import trees_finite_f32
from bindings.trees_public_inputs_door import trees_table, trees_call, trees_address, trees_reshape, trees_cast
from bindings.admitted_input_unpack import admitted_vector, admitted_unfilled, classical_view
# TOMBSTONE: trees_view / trees_vector uses (the trees admission aliases) deleted 2026-11-09 by lane a1b-doors-opt-in-trees-linear; code recoverable at f59d913b0b.
# Restore: experiments/removed/A1B_LABELS_ADMISSION_COPY.patch.
from bindings.classical_pandas_door import classical_admit_value
from bindings.trees_public_forest_door import forest_effective_mode, _engine, _ordered_export, _resident
from xtrees.trees_forest_run import _family
from bindings.trees_public_boost_door import _seed
from bindings.trees_explainer_native_door import _get, _int, _entry, _snapshot, _replace, _tree_mode
from xtrees.trees_agnostic_native_run import _name, _SWITCH_FACTS, _chunk
from bindings.trees_keys_unpack import trees_keys
from bindings.operation_names_trees import TREES_OPN_TREES_AGNOSTIC_KERNEL_INIT, TREES_OPN_TREES_AGNOSTIC_PERMUTATION_INIT, TREES_OPN_TREES_AGNOSTIC_KERNEL_VALUES, TREES_OPN_TREES_AGNOSTIC_PERMUTATION_VALUES
from bindings.py_balanced import py_item

comptime _I64 = MutPointer[Int64, MutUntrackedOrigin]
comptime _F32 = MutPointer[Float32, MutUntrackedOrigin]

#: MOJOLEARN_IDN_PSHAP_RESIDENT: 2^28 float32 words (1 GiB of synthetic rows
#: on the device), a memory bound that fits every supported GPU.
comptime _AGN_BUDGET_RESIDENT = 1 << 28
#: x_trees_fast_switches bits (providers/python/xtrees_api_pyapi.mojo
#: XTREES_FAST_SWITCHES; python/mojolearn/_expansion_trees.py documents the
#: first three): MOJOLEARN_KSHAP_FAST_BATCH, MOJOLEARN_AGN_IDN_SYN_POOL,
#: MOJOLEARN_PSHAP_DELTA, MOJOLEARN_IDN_SHAP_DEVICE_MODEL (an IDENTICAL GPU
#: build: Kernel/Permutation SHAP over a built-in flat forest evaluate the
#: model on the device, xtrees/agnostic_device.mojo `model_kernel`; moves no
#: bit).
comptime _KSHAP_FAST_BATCH = 8
comptime _AGN_IDN_SYN_POOL = 16
comptime _PSHAP_DELTA = 32
comptime _AGN_IDN_DEVICE_MODEL = 64
# TOMBSTONE: MOJOLEARN_KSHAP_FAST_DEVICE_MODEL (never measured, deleted) deleted 2026-11-09 by e2adeea367; code
# recoverable at f59d913b0b. Same commit: its twin MOJOLEARN_PSHAP_FAST_DEVICE_MODEL (bits 1024 / 2048).
# Restore: git apply experiments/removed/MOJOLEARN_KSHAP_FAST_DEVICE_MODEL.patch


def _b() raises -> PythonObject:
    return native_loaded_module("builtins")



def _switches(table: PythonObject) raises -> Int:
    var query = _entry(table, "x_trees_fast_switches", False)
    if query is None:
        return 0
    var key = Int(query._obj_ptr)
    var facts = _SWITCH_FACTS.get_or_create_ptr()
    if key in facts[].answers:
        return facts[].answers[key]
    var bits = py_call(query)
    if py_type(bits) is not _b().int:
        raise Error("TypeError: x_trees_fast_switches must return an exact integer")
    var answer = Int(py=bits)
    facts[].answers[key] = answer
    return answer


def _call(table: PythonObject, name: String, args: PythonObject) raises -> PythonObject:
    """`entry(*args)` of a connected compiled x_trees entry, keeping the
    class of a refusal it raises."""
    return trees_call(_entry(table, name), args)


def _ints(*values: Int) raises -> PythonObject:
    var out = Python.list()
    for i in range(len(values)):  # driver-loop: fixed binding parameter words
        _ = py_call(out.append, PythonObject(values[i]))
    return out

def _class_value(model: PythonObject, name: String) raises -> PythonObject:
    """A class attribute found on the model's MRO namespaces (no descriptor
    or metaclass protocol), or None."""
    var nk = trees_keys()
    var mro = py_call(py_item(_b().type.__dict__, nk[].k___mro__).__get__, py_type(model))
    for i in range(len(mro)):  # driver-loop: the model class ancestry
        var namespace = _class_fields(mro[i])
        if py_contains(namespace, name):
            return namespace[name]
    return PythonObject(None)


def _method(model: PythonObject, context: PythonObject) raises -> String:
    """The explained method: a classifier's predict_proba, else predict."""
    var kind = _class_value(model, "_estimator_type")
    if (kind is not None and py_type(kind) is _b().str and String(kind) == "classifier"
            and prep_estimator_has(model, "predict_proba", context)):
        return "predict_proba"
    if not prep_estimator_has(model, "predict", context):
        raise Error("TypeError: the explained mojolearn estimator has no native predict")
    return "predict"


def _evaluate(model: PythonObject, method: String, x: PythonObject, n: Int,
              context: PythonObject) raises -> PythonObject:
    """The model on rows x (an (n, d) Array) as a float32 (n, k) Array."""
    # Lane fx-rows-rest (2026-11-18): predict through the plan's internal
    # float32 operation when it names one (prep_estimator_predict_internal, as
    # the stacking and voting doors): this door reads float32 words, and the
    # public predict of a linear model returned a float64 ndarray that was then
    # narrowed back through a zero-filled snapshot, per chunk. Same words (the
    # widening was exact). A plan without one runs its predict as before.
    var out = (prep_estimator_predict_internal(model, Python.tuple(x), context) if method == "predict"
               else prep_estimator_call(model, method, Python.tuple(x), context))
    classical_admit_value(out, context)
    var view = classical_view(out, context, "output", True)
    var rank = Int(py=view.ndim)
    if rank != 1 and rank != 2:
        raise Error("ValueError: the explained model output must be 1-D or 2-D")
    if Int(py=view.shape[0]) != n:
        raise Error("ValueError: the explained model returned a different row count")
    var k = Int(py=view.shape[1]) if rank == 2 else 1
    _ = view
    # Lane fb3-named (2026-11-17; kernel-shap +247 ms taxi, +1143 ms istella vs 0.8.55): the model output of
    # every chunk went through _snapshot, a zero-filled Array of the output's size plus a serial memcpy of the
    # whole output (and a second zero-filled Array plus a cast when it was not float32): nsamples * background
    # rows per explained row. The output of a compiled estimator call is a fresh owned Array nobody else holds,
    # so a C-order float32 one is read in place (0.8.55's as_f32_c borrowed it the same way) and any other
    # numeric dtype is cast once into an unfilled float32 Array. The reshaped Array shares the store, which
    # keeps the words alive; every caller also holds its result through the kernels that read it.
    var nk = trees_keys()
    if (py_type(out) is py_item(context, nk[].k_array_class) and out._base is None and out._pin is None
            and (rank == 1 or String(out.order) == "C")):
        var source_type = String(out.dtype)
        if source_type == "<f4":
            return trees_reshape(out, Python.tuple(n, k), context)
        var narrow = admitted_unfilled([n * k], "<f4", context)
        _ = _entry(py_item(context, nk[].k_trees_helpers), "cast_elements")
        trees_cast(trees_address(out), source_type, narrow.addr, "<f4", n * k, context)
        _ = out  # the source words are read through the whole cast
        return trees_reshape(narrow.keep, Python.tuple(n, k), context)
    # Any other output object (a view over another owner): the snapshot copy, with a fresh snapshot table
    # per output (a freed output's identity may be reused).
    var flat = _snapshot(out, "<f4", rank, "output", context, Python.dict())
    _ = out
    return trees_reshape(flat, Python.tuple(n, k), context)


def _shape(phi: PythonObject, n: Int, d: Int, k: Int, context: PythonObject) raises -> PythonObject:
    if n == 0:
        var empty = admitted_vector(0, "<f8", context)
        return trees_reshape(empty, Python.tuple(0, d) if k == 1 else Python.tuple(0, d, k), context)
    return trees_reshape(phi, Python.tuple(n, d) if k == 1 else Python.tuple(n, d, k), context)


def _device_forest(model: PythonObject, mode: String, permutation: Bool, table: PythonObject,
                   context: PythonObject) raises -> PythonObject:
    """MOJOLEARN_IDN_SHAP_DEVICE_MODEL: (arrays, k,
    n_trees, rf_input) when the model is a built-in flat forest whose
    predict is the strict increasing-tree device kernel; else None (the
    model runs through its native predict per chunk). rf_input: the random
    forest's predict flushes the compared feature, ExtraTrees' does not."""
    var bits = _switches(table)
    # The FAST tier has no device-model route (MOJOLEARN_KSHAP_FAST_DEVICE_MODEL deleted 2026-11-09).
    if mode == "fast" or (bits & _AGN_IDN_DEVICE_MODEL) == 0:
        return PythonObject(None)
    var m = model
    var outer = _fields(model)
    var inner = _get(outer, "_random_inner")
    if inner is not None:
        m = inner
    var cls = py_type(m)
    var nk = trees_keys()
    var keys = py_call(_b().list, py_item(context, nk[].k_trees_classes))
    var kind = String("")
    for i in range(len(keys)):  # driver-loop: connected exact tree classes
        if cls is keys[i]:
            kind = String(py_item(context, nk[].k_trees_classes)[keys[i]])
    var rf_input = 0
    if kind == "rf_classifier" or kind == "rf_regressor":
        rf_input = 1
    elif kind != "et_classifier" and kind != "et_regressor":
        return PythonObject(None)
    var fields = _fields(m)
    if not py_contains(fields, nk[].k__offsets):
        return PythonObject(None)
    try:
        var forest_mode = forest_effective_mode(fields, context)
        var native = trees_table(context, forest_mode, "_mojolearn_trees" if _family(kind) == "et" else "_mojolearn_rf")
        # identical only: the FAST tier left before this point (K1 deleted its route)
        if forest_mode != "identical" or not (_engine(fields, forest_mode) == "sequential" and _ordered_export(native)):
            return PythonObject(None)
        _ = _resident(m, fields, native, forest_mode, context)
    except:
        # The forest has no resident route here: its native predict serves.
        return PythonObject(None)
    var names = Python.tuple("_offsets", "_colid", "_quesval", "_left_child", "_leaves")
    var arrays = Python.list()
    for i in range(5):  # driver-loop: the five flat model arrays
        _ = py_call(arrays.append, fields[names[i]])
    return Python.tuple(py_call(_b().tuple, arrays), _int(py_item(fields, nk[].k__num_outputs), "output count", True),
                        _int(py_item(fields, nk[].k__n_trees), "tree count", True), rf_input)


def _model_load(dev: PythonObject, bg: PythonObject, table: PythonObject) raises -> Bool:
    """Uploads the explained forest and the background for one shap_values
    call (`x_trees_agn_model_load`); the caller releases it."""
    if dev is None:
        return False
    var arrays = dev[0]
    var addresses = Python.list()
    for i in range(5):  # driver-loop: one address per model array
        _ = py_call(addresses.append, trees_address(arrays[i]))
    _ = _call(table, "x_trees_agn_model_load", Python.list(addresses, PythonObject(trees_address(bg)),
              _ints(Int(py=dev[2]), Int(py=arrays[1].size), Int(py=dev[1]), Int(py=bg.shape[1]),
                    Int(py=bg.shape[0]), Int(py=dev[3]))))
    _ = arrays
    return True


def _pool_release(table: PythonObject) raises:
    var release = _entry(table, "x_trees_agn_pool_release", False)
    if release is not None:
        _ = py_call(release)


def _construct(fields: PythonObject, args: PythonObject, permutation: Bool,
               context: PythonObject) raises:
    var model = args[0]
    var data = args[1]
    var random_state = args[2]
    var link = args[3]
    var requested = args[4]
    var name = _name(permutation)
    if not permutation:
        if py_type(link) is not _b().str or (String(link) != "identity" and String(link) != "logit"):
            raise Error("NotImplementedError: " + name + " supports the identity and logit links, by name")
    # Only connected mojolearn estimators run here; their predict is a
    # compiled operation. A Python callable or foreign model is refused.
    prep_estimator_admit(model, name, context)
    var seed = _seed(random_state, context)
    _ = seed
    var model_fields = _fields(model)
    if requested is None:
        requested = _get(model_fields, "numeric_mode")
    var nk = trees_keys()
    var mode = _tree_mode(requested if requested is not None else py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE))
    var table = trees_table(context, mode, "_mojolearn_x_trees")
    var method = _method(model, context)
    var bg = _snapshot(data, "<f4", 2, "data", context, Python.dict())
    var nb = Int(py=bg.shape[0])
    var d = Int(py=bg.shape[1])
    if nb < 1:
        raise Error("ValueError: " + name + " requires at least one background row")
    var out = _evaluate(model, method, bg, nb, context)
    var k = Int(py=out.shape[1])
    var ev = admitted_vector(k, "<f8", context)
    _ = _call(table, "x_trees_block_mean", Python.list(PythonObject(trees_address(out)), PythonObject(trees_address(ev)),
              _ints(1, nb, k)))
    _ = out
    if not permutation and String(link) == "logit":
        # shap KernelExplainer: expected_value = link(fnull); the link meets
        # each output after the background mean.
        _ = _call(table, "x_trees_logit", Python.list(PythonObject(trees_address(ev)), _ints(k)))
    var expected = PythonObject(Float64(MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=trees_address(ev))[0])) if k == 1 else ev
    var dev = _device_forest(model, mode, permutation, table, context)
    if dev is not None:
        var fitted = _get(model_fields, "n_features_in_")
        if Int(py=dev[1]) != k or fitted is None or py_type(fitted) is not _b().int or Int(py=fitted) != d:
            dev = PythonObject(None)
    var changes = Python.dict()
    var keys = Python.tuple("model", "data", "random_state", "numeric_mode", "_bg", "_fnull", "_method",
                            "_dev", "n_outputs_", "expected_value", "n_features_in_")
    var values = Python.tuple(model, data, random_state, requested, bg, ev,
                              PythonObject(method), dev, PythonObject(k), expected, PythonObject(d))
    for i in range(len(keys)):  # driver-loop: fixed explainer state fields
        py_setitem(changes, keys[i], values[i])
    if not permutation:
        py_setitem(changes, nk[].k_link, link)
    _replace(fields, changes, context)


struct _State(Copyable, Movable):
    var model: PythonObject
    var method: String
    var bg: PythonObject
    var table: PythonObject
    var mode: String
    var seed: Int
    var nb: Int
    var d: Int
    var k: Int
    var dev: PythonObject

    def __init__(out self, model: PythonObject, method: String, bg: PythonObject, table: PythonObject,
                 mode: String, seed: Int, nb: Int, d: Int, k: Int, dev: PythonObject):
        self.model = model
        self.method = method
        self.bg = bg
        self.table = table
        self.mode = mode
        self.seed = seed
        self.nb = nb
        self.d = d
        self.k = k
        self.dev = dev


def _state(fields: PythonObject, name: String, context: PythonObject) raises -> _State:
    var bg = _get(fields, "_bg")
    var method = _get(fields, "_method")
    if bg is None or method is None:
        raise Error("RuntimeError: " + name + " has not been initialized by the native owner")
    var requested = _get(fields, "numeric_mode")
    var nk = trees_keys()
    var mode = _tree_mode(requested if requested is not None else py_item(py_item(context, nk[].k_backend_fields), nk[].k__DEFAULT_MODE))
    var table = trees_table(context, mode, "_mojolearn_x_trees")
    var model = _get(fields, "model")
    prep_estimator_admit(model, name, context)
    return _State(model, String(method), bg, table, mode, _seed(_get(fields, "random_state"), context),
                  Int(py=bg.shape[0]), Int(py=bg.shape[1]), _int(_get(fields, "n_outputs_"), "output count", True),
                  _get(fields, "_dev"))


def _rows(x_source: PythonObject, s: _State, name: String, context: PythonObject) raises -> PythonObject:
    var x = _snapshot(x_source, "<f4", 2, "X", context, Python.dict())
    if Int(py=x.shape[1]) != s.d:
        raise Error("ValueError: X has " + String(Int(py=x.shape[1])) + " features, data has " + String(s.d))
    return x


# ---------------------------------------------------------------- the explained model as a native member
struct _Member(Movable):
    """The explained estimator as a NATIVE MEMBER (core/member_abi.mojo) for
    one values call. `native` False: the model's class has no member entry
    that serves this call, and every chunk runs the model's own compiled
    door (`_evaluate`), as before. True: `held` is the fitted member state
    the estimator keeps (bindings/member_held_door.mojo: imported at its
    first use, a hit on every later call until the estimator changes) and
    `operation` the member operation of the explained method; each chunk is
    one member call with the GIL released, written straight into the float32
    (rows, k) Array the kernels read. Released on every exit of the values
    call (`_member_release`)."""
    var native: Bool
    var held: MemberHeld
    var call: MemberCall
    var operation: Int

    def __init__(out self):
        var plan = MemberPlan()
        self.native = False
        self.held = MemberHeld()
        self.call = MemberCall(plan.route)
        self.operation = MEMBER_OP_PREDICT


def _member_release(mut em: _Member):
    """End of the values call (or of the member's use in it): a state the
    estimator keeps stays with it, a loose one is freed. Never raises."""
    em.native = False
    member_held_release(em.held)


def _member_open(mut em: _Member, s: _State, x: PythonObject, n: Int, context: PythonObject):
    """UNPACK, once per values call: the explained model's held member state
    when its family serves the explained method (predict of a regressor
    kind, predict_proba of a classifier) and its output width is the one
    construction recorded. Anything else (no mark, a classifier explained
    through predict, whose member predict writes codes, an admission or
    import the family refuses, another width, an element type other than
    float32, a classifier member that does not say its element type) leaves
    `em` closed: the fallback rule, the model's own door then answers or
    raises its own words. Never raises.

    WHO REFUSES A NON-FINITE ROW. This door snapshots X (`x`, `n` rows) and
    the background without a finite test: the model's own door tests every
    chunk it is given. A member whose kind tests its rows
    (MEMBER_CAN_TEST_ROWS) refuses a non-finite chunk in its own kernel pass
    (`_member_rows` then answers None and the model's door raises its words):
    no host scan here. Any other member reads admitted words, and every
    chunk is a mix of cells of X and of the background, so both are scanned
    once per values call (against the door's test of every synthesized
    chunk); not finite = `em` stays closed and the door refuses."""
    comptime if not MEMBER_SET_NATIVE:
        return
    try:
        var plan = prep_estimator_plan(s.model, context)
        if plan is None or py_get(plan, "member") is None:
            return
        var proba = s.method == "predict_proba"
        if not proba and Bool(py=plan["classifier"]):
            return
        var b = _b()
        var member_context = plan["context"]
        var fields = py_call(plan["fields_get"], s.model, py_type(s.model))
        var operation = MEMBER_OP_PREDICT_PROBA if proba else MEMBER_OP_PREDICT
        var held = member_held(s.model, fields, plan, member_context, b,
                               MEMBER_CAN_PREDICT_PROBA if proba else MEMBER_CAN_PREDICT)
        if not held.published:
            return
        var cols = -1
        var dtype = MEMBER_DTYPE_UNSAID
        var tests = False
        try:
            var probe = member_call_of(held.plan, held.state)
            cols = member_outputs_of(held.plan, probe, operation)
            dtype = probe.out_dtype
            if dtype == MEMBER_DTYPE_UNSAID and not proba:
                dtype = MEMBER_F32   # a regressor writes float32 (every published regressor documents it)
            # Kind-exact, asked of the held state; a family whose union mask
            # lacks the bit is not asked.
            if held.plan.member.can(MEMBER_CAN_TEST_ROWS):
                tests = (member_call(held.plan.member, MEMBER_OP_CAPABILITIES, probe) & MEMBER_CAN_TEST_ROWS) != 0
        except:
            cols = -1
        var served = cols == s.k and dtype == MEMBER_F32
        if served and not tests:
            served = (trees_finite_f32(trees_address(x), n * s.d)
                      and trees_finite_f32(trees_address(s.bg), s.nb * s.d))
            _ = x  # keep the Python objects alive through the scans above
            _ = s.bg
        if not served:
            member_held_release(held)
            return
        em.held = held^
        # Built from the hold where it now lives: the call carries the
        # addresses of its plan's parameter and device words.
        em.call = member_call_of(em.held.plan, em.held.state)
        em.operation = operation
        em.native = True
    except:
        return


def _member_rows(mut em: _Member, x: PythonObject, n: Int, d: Int, k: Int,
                 context: PythonObject) raises -> PythonObject:
    """The member on rows x (an (n, d) float32 C-order Array this door built)
    as a float32 (n, k) Array: one member call, the GIL released, the output
    written in place (no output admission, no cast, no snapshot). None when
    the member refuses the call: the member is released and the caller runs
    the model's own door for this and every later chunk."""
    # Unfilled: the member writes every cell of dst before anything reads it.
    var dst = admitted_unfilled([n * k], "<f4", context)
    em.call.x = MemberMatrix(trees_address(x), n, d)
    em.call.dst = MemberMatrix(dst.addr, n, k)
    var done = False
    try:
        var py = Python()
        with GILReleased(py):
            _ = member_call(em.held.plan.member, em.operation, em.call)
        done = True
    except:
        done = False
    _ = x  # the rows are read through the whole member call
    if not done:
        _member_release(em)
        return PythonObject(None)
    return trees_reshape(dst.keep, Python.tuple(n, k), context)


def _model_out(mut em: _Member, s: _State, x: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    """The explained model on rows x as a float32 (n, k) Array: through the
    held native member when one is open, else the model's compiled door."""
    if em.native and n > 0:
        var out = _member_rows(em, x, n, s.d, s.k, context)
        if out is not None:
            return out
    return _evaluate(s.model, s.method, x, n, context)


def _model_rows(mut em: _Member, s: _State, syn: PythonObject, rows: Int, context: PythonObject) raises -> PythonObject:
    """The model on a chunk's synthetic matrix: float32 (rows, k)."""
    var out = _model_out(em, s, trees_reshape(syn, Python.tuple(rows, s.d), context), rows, context)
    if Int(py=out.shape[1]) != s.k:
        raise Error("ValueError: the explained model output width changed after construction")
    return out


def _kernel_values(fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """KernelExplainer values; the member state opened inside is released on
    every exit."""
    var em = _Member()
    var result = PythonObject(None)
    try:
        result = _kernel_values_over(fields, args, context, em)
    finally:
        _member_release(em)
    return result


def _kernel_values_over(fields: PythonObject, args: PythonObject, context: PythonObject,
                        mut em: _Member) raises -> PythonObject:
    var name = String("KernelExplainer")
    var nsamples = args[1]
    var l1 = args[2]
    var b = _b()
    var l1_ok = ((py_type(l1) is b.str and String(l1) == "auto") or (py_type(l1) is b.bool and not Bool(py=l1))
                 or (py_type(l1) is b.int and Int(py=l1) == 0))
    if not l1_ok:
        raise Error("NotImplementedError: KernelExplainer carries no l1 feature selection; l1_reg must be 'auto', False or 0")
    var s = _state(fields, name, context)
    var link = _get(fields, "link")
    var logit = py_type(link) is b.str and String(link) == "logit"
    var x = _rows(args[0], s, name, context)
    var n = Int(py=x.shape[0])
    var d = s.d
    var k = s.k
    var nb = s.nb
    var M = d
    var ns = 0
    if py_type(nsamples) is b.str and String(nsamples) == "auto":
        ns = 2 * M + 2048
    elif py_type(nsamples) is b.int:
        ns = Int(py=nsamples)
    else:
        raise Error("TypeError: nsamples must be an int or 'auto'")
    if M <= 30:
        ns = min(ns, (1 << M) - 2)
    var h = max(1, M // 2)
    var size_off = admitted_vector(h + 1, "<i8", context)
    var size_w = admitted_vector(h, "<f8", context)
    var cdf = admitted_vector(h, "<f8", context)
    var sched = admitted_vector(6, "<i8", context)
    if M > 1:
        _ = _call(s.table, "x_trees_kshap_schedule", Python.list(Python.tuple(trees_address(size_off), trees_address(size_w), trees_address(cdf)),
                  PythonObject(trees_address(sched)), _ints(M, max(ns, 0))))
    var sp = _I64(unsafe_from_address=trees_address(sched))
    var m = Int(sp[0])
    var nfixed = Int(sp[1])
    var nfull = Int(sp[2])
    var npaired = Int(sp[3])
    var L = Int(sp[4])
    var wbits = Int(sp[5])
    _ = sched
    var taddr = Python.tuple(trees_address(size_off), trees_address(size_w), trees_address(cdf))
    var phi = admitted_vector(max(n * d * k, 1), "<f8", context)
    if n == 0:
        return _shape(phi, 0, d, k, context)
    # A forest evaluated on the device needs the model once (fx): no member
    # state is imported for it. Every other route calls the model per chunk.
    if s.dev is None:
        _member_open(em, s, x, n, context)
    var fx = _model_out(em, s, x, n, context)   # the explained rows, one model call
    if Int(py=fx.shape[1]) != k:
        raise Error("ValueError: the explained model output width changed after construction")
    var x0 = trees_address(x)
    var f0 = trees_address(fx)
    var p0 = trees_address(phi)
    var fnull = _get(fields, "_fnull")
    var nl = trees_address(fnull)
    var bg0 = trees_address(s.bg)
    var link_word = 1 if logit else 0
    var R = _chunk(m * nb * d, n)
    var bits = _switches(s.table)
    var fast = s.mode == "fast"
    if m > 0 and fast and (bits & _KSHAP_FAST_BATCH) != 0 and s.dev is None:
        # MOJOLEARN_KSHAP_FAST_BATCH: one host buffer for every chunk's
        # synthetic rows, each chunk's linked background means into one
        # float64 (n, m, k) block, then the solve over as many rows per call
        # as the budget holds (the same words as the per-chunk solve).
        var ey = admitted_vector(n * m * k, "<f8", context)
        var e0 = trees_address(ey)
        # Optional, not a PythonObject read and reassigned in a loop (modular/modular#7546 pattern; runtime-fix-save-forest sweep).
        var syn = Optional[PythonObject](admitted_vector(R * m * nb * d, "<f4", context))
        var r0 = 0
        while r0 < n:  # driver-loop: one binding call per row chunk
            var rows = min(R, n - r0)
            if rows < R:
                syn = Optional[PythonObject](admitted_vector(rows * m * nb * d, "<f4", context))
            _ = _call(s.table, "x_trees_kshap_synth", Python.list(PythonObject(x0 + 4 * r0 * d), PythonObject(bg0), taddr,
                      PythonObject(trees_address(syn.value())), _ints(rows, nb, d, nfixed, m, nfull, L, npaired, r0, s.seed, wbits)))
            var out = _model_rows(em, s, syn.value(), rows * m * nb, context)
            _ = _call(s.table, "x_trees_kshap_means", Python.list(PythonObject(trees_address(out)),
                      PythonObject(e0 + 8 * r0 * m * k), _ints(rows, nb, m, k, link_word)))
            _ = out
            r0 += rows
        _ = syn
        var S = _chunk(m * d, n)
        var s0 = 0
        while s0 < n:  # driver-loop: solve-batch loop (one binding call per batch)
            var rows = min(S, n - s0)
            _ = _call(s.table, "x_trees_kshap_solve_ey", Python.list(PythonObject(e0 + 8 * s0 * m * k), PythonObject(f0 + 4 * s0 * k),
                      PythonObject(nl), taddr, PythonObject(p0 + 8 * s0 * d * k),
                      _ints(rows, nb, d, nfixed, m, nfull, L, npaired, s0, s.seed, wbits, k, link_word)))
            s0 += rows
        _ = ey
    elif _model_load(s.dev, s.bg, s.table):
        # The forest on the device: masks -> model -> means -> solve per
        # chunk, nothing crossing but the chunk's rows and phi.
        try:
            var r0 = 0
            while r0 < n:  # driver-loop: one binding call per row chunk
                var rows = min(R, n - r0)
                _ = _call(s.table, "x_trees_kshap_solve_model", Python.list(PythonObject(x0 + 4 * r0 * d), PythonObject(f0 + 4 * r0 * k),
                          PythonObject(nl), taddr, PythonObject(p0 + 8 * r0 * d * k),
                          _ints(rows, nb, d, nfixed, m, nfull, L, npaired, r0, s.seed, wbits, k, link_word)))
                r0 += rows
        finally:
            _ = _call(s.table, "x_trees_agn_model_release", Python.list())
    else:
        var reuse = (bits & _AGN_IDN_SYN_POOL) != 0   # one synthetic buffer for every chunk
        # Optional, not a PythonObject read and reassigned in a loop (modular/modular#7546 pattern; runtime-fix-save-forest sweep).
        var syn = Optional[PythonObject]()
        try:
            var r0 = 0
            while r0 < n:  # driver-loop: one binding call per row chunk
                var rows = min(R, n - r0)
                var y_addr = 0
                var out = PythonObject(None)
                if m > 0:
                    if not reuse or not syn or Int(py=syn.value().size) != rows * m * nb * d:
                        syn = Optional[PythonObject]()
                        # Unfilled (fb-infer/C, 2026-11-17): x_trees_kshap_synth writes every
                        # cell (ksynth_kernel over all rows*m*nb*d, then the download), so the
                        # zero fill (a serial pass over up to the 128 MB chunk budget) is dropped.
                        syn = Optional[PythonObject](admitted_unfilled([rows * m * nb * d], "<f4", context).keep)
                    _ = _call(s.table, "x_trees_kshap_synth", Python.list(PythonObject(x0 + 4 * r0 * d), PythonObject(bg0), taddr,
                              PythonObject(trees_address(syn.value())),
                              _ints(rows, nb, d, nfixed, m, nfull, L, npaired, r0, s.seed, wbits)))
                    out = _model_rows(em, s, syn.value(), rows * m * nb, context)
                    y_addr = trees_address(out)
                    if not reuse:
                        syn = Optional[PythonObject]()
                _ = _call(s.table, "x_trees_kshap_solve", Python.list(PythonObject(y_addr), PythonObject(f0 + 4 * r0 * k),
                          PythonObject(nl), taddr, PythonObject(p0 + 8 * r0 * d * k),
                          _ints(rows, nb, d, nfixed, m, nfull, L, npaired, r0, s.seed, wbits, k, link_word)))
                _ = out
                r0 += rows
        finally:
            if reuse:
                _pool_release(s.table)   # the pooled device buffer does not outlive the explanation
        _ = syn
    _ = x
    _ = fx
    _ = fnull
    _ = size_off
    _ = size_w
    _ = cdf
    return _shape(phi, n, d, k, context)


def _linear_block(s: _State, context: PythonObject) raises -> PythonObject:
    """MOJOLEARN_IDN_PSHAP_RESIDENT: ([w | b] float32 block, link) when the
    binding carries the switch and the explained model's predict is this
    library's linear `link(X W^T + b)`; else None. Byte copies only."""
    var ok = _entry(s.table, "x_trees_pshap_linear_ok", False)
    if ok is None:
        return PythonObject(None)
    var flag = py_call(ok)
    if py_type(flag) is not _b().int or Int(py=flag) != 1:
        return PythonObject(None)
    var nk = trees_keys()
    var linear_predict = py_get(context, nk[].k_trees_linear_predict)
    if linear_predict is None or _class_value(s.model, "predict") is not linear_predict:
        return PythonObject(None)
    var fields = _fields(s.model)
    var fitted = _get(fields, "n_features_in_")
    if fitted is None or py_type(fitted) is not _b().int or Int(py=fitted) != s.d:
        return PythonObject(None)
    var link = _get(fields, "_link")
    if link is None:
        link = _class_value(s.model, "_link")
    if link is None or py_type(link) is not _b().int:
        return PythonObject(None)
    var coef = _get(fields, "coef_")
    var intercept = _get(fields, "intercept_")
    if coef is None or intercept is None:
        return PythonObject(None)
    var snaps = Python.dict()
    var c_view = classical_view(coef, context, "coef_")
    var c_rank = Int(py=c_view.ndim)
    _ = c_view
    var c = _snapshot(coef, "<f4", c_rank, "coef_", context, snaps)
    var bias = PythonObject(None)
    if py_type(intercept) is _b().float or py_type(intercept) is _b().int:
        bias = _snapshot(Python.list(intercept), "<f4", 1, "intercept_", context, snaps)
    else:
        var i_view = classical_view(intercept, context, "intercept_")
        var i_rank = Int(py=i_view.ndim)
        _ = i_view
        bias = _snapshot(intercept, "<f4", i_rank, "intercept_", context, snaps)
    var kk = Int(py=bias.size)
    var d = Int(py=c.size) // max(kk, 1)
    if kk != s.k or d != s.d or Int(py=c.size) != kk * d:
        return PythonObject(None)
    var wb = admitted_vector(kk * (d + 1), "<f4", context)
    var o = _F32(unsafe_from_address=trees_address(wb))
    var cp = _F32(unsafe_from_address=trees_address(c))
    var bp = _F32(unsafe_from_address=trees_address(bias))
    for j in range(kk):  # driver-loop: over the fitted output rows (model structure)
        for f in range(d):  # driver-loop: one coefficient row copy
            o[j * (d + 1) + f] = cp[j * d + f]
        o[j * (d + 1) + d] = bp[j]
    _ = c
    _ = bias
    return Python.tuple(wb, link)


def _permutation_values(fields: PythonObject, args: PythonObject, context: PythonObject) raises -> PythonObject:
    """PermutationExplainer values; the member state opened inside is
    released on every exit."""
    var em = _Member()
    var result = PythonObject(None)
    try:
        result = _permutation_values_over(fields, args, context, em)
    finally:
        _member_release(em)
    return result


def _permutation_values_over(fields: PythonObject, args: PythonObject, context: PythonObject,
                             mut em: _Member) raises -> PythonObject:
    var name = String("PermutationExplainer")
    var b = _b()
    var s = _state(fields, name, context)
    var x = _rows(args[0], s, name, context)
    var n = Int(py=x.shape[0])
    var d = s.d
    var k = s.k
    var nb = s.nb
    if py_type(args[1]) is not b.int:
        raise Error("TypeError: npermutations must be an int")
    var npm = Int(py=args[1])
    if npm < 1:
        raise Error("ValueError: npermutations must be >= 1")
    var phi = admitted_vector(max(n * d * k, 1), "<f8", context)
    if n == 0:
        return _shape(phi, 0, d, k, context)
    var mm = npm * (2 * d + 1)
    var x0 = trees_address(x)
    var p0 = trees_address(phi)
    var bg0 = trees_address(s.bg)
    var R = _chunk(mm * nb * d, n)
    if _model_load(s.dev, s.bg, s.table):
        # The forest on the device: permutations -> model -> means ->
        # marginals per chunk, nothing crossing but the chunk's rows and phi.
        try:
            var r0 = 0
            while r0 < n:  # driver-loop: one binding call per row chunk
                var rows = min(R, n - r0)
                _ = _call(s.table, "x_trees_pshap_values_model", Python.list(PythonObject(x0 + 4 * r0 * d),
                          PythonObject(p0 + 8 * r0 * d * k), _ints(rows, nb, d, npm, r0, s.seed, k)))
                r0 += rows
        finally:
            _ = _call(s.table, "x_trees_agn_model_release", Python.list())
        _ = x
        return _shape(phi, n, d, k, context)
    var lin = _linear_block(s, context)
    if lin is not None:
        # IDN_PSHAP_RESIDENT: synthetic rows, the linear predict, means and
        # marginals on the device per chunk; only x rows and phi cross.
        var wb = lin[0]
        var Rl = max(1, min(n, _AGN_BUDGET_RESIDENT // max(1, mm * nb * d)))
        var r0 = 0
        while r0 < n:  # driver-loop: one binding call per row chunk
            var rows = min(Rl, n - r0)
            _ = _call(s.table, "x_trees_pshap_values_linear", Python.list(PythonObject(x0 + 4 * r0 * d), PythonObject(bg0),
                      PythonObject(trees_address(wb)), PythonObject(p0 + 8 * r0 * d * k),
                      _ints(rows, nb, d, npm, r0, s.seed, k, Int(py=lin[1]))))
            r0 += rows
        _ = wb
        _ = x
        return _shape(phi, n, d, k, context)
    # Past the device forest and the resident linear routes (neither calls
    # the model per chunk): the explained model as a native member.
    _member_open(em, s, x, n, context)
    var bits = _switches(s.table)
    var fast = s.mode == "fast"
    var idn_pool = (bits & _AGN_IDN_SYN_POOL) != 0
    var reuse = (fast and (bits & _KSHAP_FAST_BATCH) != 0) or idn_pool
    if fast and (bits & _PSHAP_DELTA) != 0:
        # MOJOLEARN_PSHAP_DELTA: per chunk, the device builds only the
        # synthetic rows that differ from their previous coalition's into the
        # front of one reused buffer (their count into `tot`), the model runs
        # on that prefix, and the device expands the outputs back to every
        # (coalition, background row). Three Int64 words per (coalition,
        # background row): bound them as if d were at least 16.
        var Rd = min(R, _chunk(mm * nb * max(d, 16), n))
        var tot = admitted_vector(1, "<i8", context)
        var t0 = trees_address(tot)
        var syn = admitted_vector(Rd * mm * nb * d, "<f4", context)
        var r0 = 0
        while r0 < n:  # driver-loop: one model call per row chunk
            var rows = min(Rd, n - r0)
            _ = _call(s.table, "x_trees_pshap_dsynth", Python.list(PythonObject(x0 + 4 * r0 * d), PythonObject(bg0),
                      PythonObject(trees_address(syn)), PythonObject(t0), _ints(rows, nb, d, npm, r0, s.seed)))
            var nv = Int(_I64(unsafe_from_address=t0)[0])
            if nv < 0 or nv > rows * mm * nb:
                raise Error("RuntimeError: x_trees_pshap_dsynth returned an invalid row count")
            # The first nv rows of syn, copied once into the model's input.
            var front = admitted_vector(nv * d, "<f4", context)
            var src = _F32(unsafe_from_address=trees_address(syn))
            var dst = _F32(unsafe_from_address=trees_address(front))
            if nv * d > 0:
                unsafe_memcpy(dest=dst, src=src, count=nv * d)  # one block copy, not one per element
            _ = syn  # keep the source alive through the copy above
            var out = _model_rows(em, s, front, nv, context)
            _ = front
            _ = _call(s.table, "x_trees_pshap_dvalues", Python.list(PythonObject(x0 + 4 * r0 * d), PythonObject(bg0),
                      PythonObject(trees_address(out)), PythonObject(p0 + 8 * r0 * d * k), PythonObject(t0),
                      _ints(rows, nb, d, npm, r0, s.seed, k)))
            _ = out
            r0 += rows
        _ = syn
        _ = tot
        _ = x
        return _shape(phi, n, d, k, context)
    # Optional, not a PythonObject read and reassigned in a loop (modular/modular#7546 pattern; runtime-fix-save-forest sweep).
    var syn = Optional[PythonObject]()
    try:
        var r0 = 0
        while r0 < n:  # driver-loop: one model call per row chunk
            var rows = min(R, n - r0)
            if not reuse or not syn or Int(py=syn.value().size) != rows * mm * nb * d:
                syn = Optional[PythonObject]()
                syn = Optional[PythonObject](admitted_vector(rows * mm * nb * d, "<f4", context))
            _ = _call(s.table, "x_trees_pshap_synth", Python.list(PythonObject(x0 + 4 * r0 * d), PythonObject(bg0),
                      PythonObject(trees_address(syn.value())), _ints(rows, nb, d, npm, r0, s.seed)))
            var out = _model_rows(em, s, syn.value(), rows * mm * nb, context)
            if not reuse:
                syn = Optional[PythonObject]()
            _ = _call(s.table, "x_trees_pshap_values", Python.list(PythonObject(trees_address(out)),
                      PythonObject(p0 + 8 * r0 * d * k), _ints(rows, nb, d, npm, r0, s.seed, k)))
            _ = out
            r0 += rows
    finally:
        if idn_pool:
            _pool_release(s.table)   # the pooled device buffer does not outlive the explanation
    _ = syn
    _ = x
    return _shape(phi, n, d, k, context)


def trees_agnostic_operation(model: PythonObject, operation: Int, opname: String, args: PythonObject,
                             context: PythonObject) raises -> PythonObject:
    """One `trees_agnostic_*` method by its leaf code
    (bindings/operation_names_trees.mojo; lane bw3-trees); `opname` (the
    table's String, borrowed) is for the unknown-operation text only."""
    classical_check_layout(model, context)
    var fields = _fields(model)
    if operation == TREES_OPN_TREES_AGNOSTIC_KERNEL_INIT:
        _construct(fields, args, False, context)
        return PythonObject(None)
    if operation == TREES_OPN_TREES_AGNOSTIC_PERMUTATION_INIT:
        _construct(fields, args, True, context)
        return PythonObject(None)
    if operation == TREES_OPN_TREES_AGNOSTIC_KERNEL_VALUES:
        return _kernel_values(fields, args, context)
    if operation == TREES_OPN_TREES_AGNOSTIC_PERMUTATION_VALUES:
        return _permutation_values(fields, args, context)
    raise Error("unknown model-agnostic explainer operation " + opname)
