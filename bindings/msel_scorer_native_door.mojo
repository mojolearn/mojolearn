# SPDX-License-Identifier: Apache-2.0
# Fixed scorer records call verified compiled estimator/metric entries. The
# Python function objects identify import-time plans; they are never invoked.
from bindings.mode_answer_door import answer_mode_object
from core.mode_answer import MODE_FAMILY_MSEL
from bindings.native_loaded_module_door import NativeModules, native_loaded_module
from bindings.native_class_admission_door import class_record, class_admit
from bindings.py_model_records import RECORDS_PREPARED, ContentRecord, records_stamped_get, records_stamped_put
from bindings.py_cpython_table import cpython_table
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.family_common_pack import family_call, family_array
from bindings.classical_arrays_door import classical_address, classical_is_array, classical_dtype_str
from bindings.classical_keys_unpack import classical_keys
from bindings.msel_helpers_key_unpack import msel_helpers_key
from bindings.msel_helpers_released_unpack import msel_proba_column
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.metrics_native_support_door import metric_is_real, metric_float
from bindings.msel_inputs_unpack import msel_input, msel_admit
from bindings.native_buffer_owner_pack import native_storage_allocate
from bindings.msel_estimator_native_door import msel_estimator_call, msel_estimator_plan, msel_planned_call
from bindings.msel_member_device_unpack import msel_member_host_folds
from core.device_route import device_cpu_fast


def msel_metric_plan(function: PythonObject, context: PythonObject) raises -> PythonObject:
    if py_type(function) is not context["family_msel_metric_function_type"]:
        raise Error("TypeError: Python metric callbacks are not accepted; use a connected native mojolearn metric")
    var plan = py_get(context["family_msel_metric_plans"], function)
    if plan is None:
        raise Error("TypeError: Python metric callbacks are not accepted; use a connected native mojolearn metric")
    # Native function metadata avoids function.__code__'s audit callback while
    # preserving its connected-code identity check at every scoring boundary.
    # N067: through the one CPython table (resolved once per process), not
    # a by-name lookup on every score.
    ref cpy = Python().cpython()
    var code = cpython_table()[].PyFunction_GetCode(function._obj_ptr)
    if not code:
        raise cpy.unsafe_get_error()
    if code != plan["code"]._obj_ptr:
        raise Error("TypeError: the connected metric function was modified")
    return plan


#: The class-record slot of native_class_admission for the scorer class rows.
comptime _SCORER_CLASS_SLOT = "msel_scorer_class_v1"
#: The stamped-record name of a scorer's field-key admission (family RECORDS_PREPARED).
comptime _SCORER_FIELDS_RECORD = "msel_scorer_fields_v1"


def _scorer_class(context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = context["family_msel_scorer_class"]
    if py_type(cls) is not b.type:
        raise Error("TypeError: native scorer metadata refuses metaclass hooks")
    # N067: the ancestry and hook rows depend only on the class state, so
    # they are checked once per class state (native_class_admission type
    # watchers: any class or ancestor modification, `__bases__` included,
    # drops the record), not on every score. The payload is the address of
    # the import-time hook tuple (alive for the process), never the class.
    # Without watchers (Python < 3.12) the full check runs on every call.
    var hooks = context["family_msel_scorer_hooks"]
    var mods = NativeModules()
    var held = class_record(mods, _SCORER_CLASS_SLOT, cls)
    if held is not None and Int(py=held) == Int(hooks._obj_ptr):
        return cls
    _scorer_class_rows(cls, hooks, b)
    class_admit(mods, _SCORER_CLASS_SLOT, cls, PythonObject(Int(hooks._obj_ptr)))
    return cls


def _scorer_class_rows(cls: PythonObject, hooks: PythonObject, b: PythonObject) raises:
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    if len(mro) != 2 or mro[1] is not b.object:
        raise Error("TypeError: native scorer metadata refuses inherited lifecycle hooks")
    var source = py_call(b.type.__dict__["__dict__"].__get__, cls)
    for k in range(len(hooks)):  # driver-loop: connected scorer storage and method identities before allocation or access
        if py_get(source, hooks[k][0]) is not hooks[k][1]:
            raise Error("TypeError: native scorer methods or storage hooks changed")


def msel_scorer_create(function: PythonObject, response: PythonObject, sign: Int,
                       keywords: PythonObject, context: PythonObject) raises -> PythonObject:
    _ = msel_metric_plan(function, context)
    msel_admit(response, context)
    msel_admit(keywords, context)
    var cls = _scorer_class(context)
    var scorer = native_storage_allocate(cls)
    var fields = py_call(context["family_msel_scorer_fields"], scorer, cls)
    # Fresh admitted native storage; Python __setattr__ and property setters
    # are never involved and no pre-existing owner can be released here.
    py_setitem(fields, "_score_func", function)
    py_setitem(fields, "_sign", PythonObject(sign))
    py_setitem(fields, "_kwargs", keywords)
    py_setitem(fields, "_response_method", response)
    py_setitem(fields, "_name", PythonObject(None))
    return scorer


def msel_scorer_fields(scorer: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = _scorer_class(context)
    if py_type(scorer) is not cls:
        raise Error("TypeError: Python scorer callbacks are not accepted; use a scoring string (one of "
                    + "get_scorer_names()) or make_scorer over a native mojolearn metric")
    var fields = py_call(context["family_msel_scorer_fields"], scorer, cls)
    # N067: the exact-str key admission once per scorer state: a stamped
    # record of the scorer (bindings/py_model_records.mojo, keyed by the
    # scorer's identity, released when it dies). Its generation word follows
    # the scorer's instance dictionary (any insertion, replacement, deletion
    # or clear misses), and the keys are all that this scan reads, so the
    # record needs no content words. Without dictionary watchers (Python <
    # 3.12) the scan runs on every call.
    var hooks = context["family_msel_scorer_hooks"]
    var stamped = records_stamped_get(RECORDS_PREPARED, scorer, _SCORER_FIELDS_RECORD)
    if stamped is None or stamped is not hooks:
        var keys = py_call(b.list, fields.keys())
        for k in range(len(keys)):  # driver-loop: exact scorer keys before metadata lookup
            if py_type(keys[k]) is not b.str:
                raise Error("TypeError: native scorer fields require exact string keys")
        if py_type(fields) is b.dict:
            _ = records_stamped_put(RECORDS_PREPARED, scorer, _SCORER_FIELDS_RECORD, ContentRecord(), hooks)
    _ = msel_metric_plan(fields["_score_func"], context)
    return fields


struct _MetricItems(Movable):
    # Exact-dict identity snapshot. Native iteration neither hashes foreign
    # keys nor builds a Python key list; retained keys/values pin identities.
    var kind: PythonObject
    var keys: List[PythonObject]
    var values: List[PythonObject]

    def __init__(out self):
        self.kind = PythonObject(None)
        self.keys = List[PythonObject]()
        self.values = List[PythonObject]()

    def capture(mut self, value: PythonObject, dict_type: PythonObject, str_type: PythonObject) raises -> Bool:
        self.kind = py_type(value)
        self.keys = List[PythonObject]()
        self.values = List[PythonObject]()
        if self.kind is not dict_type:
            return False
        ref cpy = Python().cpython()
        var position = 0
        var key = PyObjectPtr()
        var item = PyObjectPtr()
        # N067: the one CPython table (addresses as Int, its stated form).
        var table = cpython_table()
        for i in range(len(value)):  # driver-loop: capture admitted option references once after preparation
            if table[].PyDict_Next(value._obj_ptr, Int(MutPointer(to=position)), Int(MutPointer(to=key)),
                                   Int(MutPointer(to=item))) == 0:
                return False
            var owner = PythonObject(from_owned=cpy.Py_NewRef(key))
            if py_type(owner) is not str_type:
                return False
            self.keys.append(owner)
            self.values.append(PythonObject(from_owned=cpy.Py_NewRef(item)))
        _ = value  # dictionary owns borrowed slots through the last NewRef
        return True

    def same(self, value: PythonObject) raises -> Bool:
        if py_type(value) is not self.kind or len(value) != len(self.keys):
            return False
        var position = 0
        var key = PyObjectPtr()
        var item = PyObjectPtr()
        var table = cpython_table()
        for i in range(len(self.keys)):  # driver-loop: check mutable options after every prediction/GIL boundary
            if table[].PyDict_Next(value._obj_ptr, Int(MutPointer(to=position)), Int(MutPointer(to=key)),
                                   Int(MutPointer(to=item))) == 0:
                return False
            if key != self.keys[i]._obj_ptr or item != self.values[i]._obj_ptr:
                return False
        return True


def _metric_recipe_stable(recipe: PythonObject, inputs: PythonObject, allowed: PythonObject,
                          operation: PythonObject, b: PythonObject) raises -> Bool:
    # Import-connected recipes are tuples of two-word tuples and immutable
    # scalar names/constants. Noncanonical mutable schemas still execute via
    # ordinary preparation on every call; they never enter the reuse path.
    if py_type(recipe) is not b.tuple or py_type(inputs) is not b.tuple or py_type(allowed) is not b.tuple or py_type(operation) is not b.str:
        return False
    for i in range(len(inputs)):  # driver-loop: immutable input-name schema
        if py_type(inputs[i]) is not b.str:
            return False
    for i in range(len(allowed)):  # driver-loop: immutable allowed-key schema
        if py_type(allowed[i]) is not b.str:
            return False
    for i in range(len(recipe)):  # driver-loop: immutable native ABI recipe schema
        var item = recipe[i]
        if py_type(item) is not b.tuple or len(item) != 2 or py_type(item[0]) is not b.str:
            return False
        var kind = py_type(item[1])
        if item[1] is not None and kind is not b.str and kind is not b.bool and kind is not b.int and kind is not b.float:
            return False
    return True


struct MselMetricPrepared(Movable):
    # One scorer's metadata-only template within a CV operation. Public scorer
    # admission still runs each score; options/schema/module identities are
    # checked after prediction, since native execution may have released GIL.
    # No fit state, y, prediction or result is retained here. The scored
    # estimator is pinned by identity only (`scored`): the metric module is
    # the one of ITS device (`host`: it says device="cpu" beside a GPU set,
    # so its metric runs on the CPU metrics module connected at import,
    # `family_msel_metric_bindings_cpu`). That one read of the member's
    # device is made when the template is prepared (once per scorer of one
    # cross-validation unit; a unit's clone can take its device from a
    # candidate), never on a template hit.
    var reusable: Bool
    var ready: Bool
    var function: PythonObject
    var plan: PythonObject
    var recipe: PythonObject
    var inputs: PythonObject
    var allowed: PythonObject
    var entry: PythonObject
    var operation: PythonObject
    var metric_context: PythonObject
    var defaults: _MetricItems
    var keywords: _MetricItems
    var arguments: List[PythonObject]
    var sources: List[Int]
    var binding_keys: List[PythonObject]
    var binding_values: List[PythonObject]
    var uses_default_mode: Bool
    var default_mode: PythonObject
    var scored: PythonObject
    var host: Bool

    def __init__(out self, reusable: Bool = False):
        self.reusable = reusable
        self.ready = False
        self.function = PythonObject(None)
        self.plan = PythonObject(None)
        self.recipe = PythonObject(None)
        self.inputs = PythonObject(None)
        self.allowed = PythonObject(None)
        self.entry = PythonObject(None)
        self.operation = PythonObject(None)
        self.metric_context = PythonObject(None)
        self.defaults = _MetricItems()
        self.keywords = _MetricItems()
        self.arguments = List[PythonObject]()
        self.sources = List[Int]()
        self.binding_keys = List[PythonObject]()
        self.binding_values = List[PythonObject]()
        self.uses_default_mode = False
        self.default_mode = PythonObject(None)
        self.scored = PythonObject(None)
        self.host = False

    def same(self, function: PythonObject, plan: PythonObject, keywords: PythonObject,
             context: PythonObject, estimator: PythonObject) raises -> Bool:
        if not self.ready or function is not self.function or plan is not self.plan:
            return False
        if estimator is not self.scored:
            return False
        if (plan["recipe"] is not self.recipe or plan["inputs"] is not self.inputs
                or plan["allowed"] is not self.allowed or plan["entry"] is not self.entry
                or plan["operation"] is not self.operation or plan["context"] is not self.metric_context):
            return False
        if not self.defaults.same(plan["defaults"]) or not self.keywords.same(keywords):
            return False
        if self.uses_default_mode and context["family_msel_backend_fields"]["_DEFAULT_MODE"] is not self.default_mode:
            return False
        if self.host:
            for i in range(len(self.binding_values)):  # driver-loop: the CPU metric module remains current
                if py_get(context, "family_msel_metric_bindings_cpu") is not self.binding_values[i]:
                    return False
            return True
        for i in range(len(self.binding_keys)):  # driver-loop: selected native binding owners remain current
            if context["family_msel_metric_bindings"][self.binding_keys[i]] is not self.binding_values[i]:
                return False
        return True

    def prepare(mut self, function: PythonObject, plan: PythonObject, y: PythonObject,
                predicted: PythonObject, keywords: PythonObject, context: PythonObject,
                b: PythonObject, estimator: PythonObject, estimator_plan: PythonObject) raises:
        self.ready = False
        self.scored = estimator
        self.host = False
        var asked = estimator is None
        self.function = function
        self.plan = plan
        self.recipe = plan["recipe"]
        self.inputs = plan["inputs"]
        self.allowed = plan["allowed"]
        self.entry = plan["entry"]
        self.operation = plan["operation"]
        self.metric_context = plan["context"]
        self.arguments = List[PythonObject]()
        self.sources = List[Int]()
        self.binding_keys = List[PythonObject]()
        self.binding_values = List[PythonObject]()
        self.uses_default_mode = False
        var values = py_call(b.dict, plan["defaults"])
        py_setitem(values, self.inputs[0], y)
        py_setitem(values, self.inputs[1], predicted)
        var keys = py_call(b.list, keywords.keys())
        for k in range(len(keys)):  # driver-loop: preserve ordered metric keyword refusals on preparation
            var name = keys[k]
            if py_type(name) is not b.str or not py_contains(self.allowed, name):
                raise Error("TypeError: native metric received an unexpected keyword argument")
            if name is self.inputs[0] or name is self.inputs[1] or Bool(py=name == self.inputs[0]) or Bool(py=name == self.inputs[1]):
                raise Error("TypeError: native metric received multiple values for a positional input")
            py_setitem(values, name, keywords[name])
        for k in range(len(self.recipe)):  # driver-loop: compile fixed ABI tags/options once per unchanged schema
            var item = self.recipe[k]
            var tag = String(item[0])
            var source = 0
            var value: PythonObject
            if tag == "parameter":
                value = values[item[1]]
                if Bool(py=item[1] == self.inputs[1]):
                    source = 2
                elif Bool(py=item[1] == self.inputs[0]):
                    source = 1
            elif tag == "binding":
                var mode = PythonObject(None) if item[1] is None else values[item[1]]
                if mode is None:
                    mode = context["family_msel_backend_fields"]["_DEFAULT_MODE"]
                    self.uses_default_mode = True
                    self.default_mode = mode
                if py_type(mode) is not b.str:
                    raise Error("ValueError: numeric_mode must be 'fast' or 'identical'")
                var clean = py_call(py_call(mode.strip).lower)
                var clean_word = String(clean)
                if clean_word != "identical" and clean_word != "fast":
                    raise Error("ValueError: numeric_mode must be 'fast' or 'identical'")
                answer_mode_object(clean, clean_word, MODE_FAMILY_MSEL)
                if not asked:
                    # The scored estimator's device, read once per template
                    # (the first binding item), after the mode refusals above.
                    asked = True
                    self.host = msel_member_host_folds(
                        py_call(estimator_plan["fields_get"], estimator, py_type(estimator)), b, context)
                if self.host:
                    # The CPU metrics module is the identical one (the shared
                    # refusal of device="cpu" with the fast mode).
                    if clean_word != "identical":
                        device_cpu_fast()
                    var cpu = py_get(context, "family_msel_metric_bindings_cpu")
                    if cpu is None:
                        raise Error("ImportError: device='cpu' is not connected in this installation")
                    value = cpu
                else:
                    value = context["family_msel_metric_bindings"][clean]
                if py_type(value) is b.str:
                    raise Error(String(value))
                self.binding_keys.append(clean)
                self.binding_values.append(value)
            else:
                value = item[1]
            self.sources.append(source)
            self.arguments.append(PythonObject(None) if source != 0 else value)
        # Single-call public wrappers do not capture a cache they cannot reuse.
        self.ready = (self.reusable and _metric_recipe_stable(self.recipe, self.inputs, self.allowed, self.operation, b)
            and self.defaults.capture(plan["defaults"], b.dict, b.str)
            and self.keywords.capture(keywords, b.dict, b.str))

    def call(mut self, function: PythonObject, y: PythonObject, predicted: PythonObject,
             keywords: PythonObject, context: PythonObject, estimator: PythonObject,
             estimator_plan: PythonObject) raises -> PythonObject:
        # `estimator` (with its checked plan) is the estimator whose output is
        # scored, or None for a metric called without one: the process's set.
        var b = native_loaded_module("builtins")
        var plan = msel_metric_plan(function, context)
        var entry = plan["entry"]
        if entry is None or py_type(entry) is not py_type(b.len):
            raise Error("ImportError: native scorer metric binding is unavailable")
        if py_type(keywords) is not b.dict:
            raise Error("TypeError: native metric keywords must be an exact dict")
        msel_admit(keywords, context)
        if not self.same(function, plan, keywords, context, estimator):
            self.prepare(function, plan, y, predicted, keywords, context, b, estimator, estimator_plan)
        # Fresh immutable call arguments: no tuple retained by an entry can be
        # mutated by a later phase, and dynamic y/predicted never enter template.
        # N067: the standard library's tuple builders (the cpython_table
        # rule keeps them there), not two by-name lookups per score.
        ref cpy = Python().cpython()
        var raw = cpy.PyTuple_New(len(self.arguments))
        if not raw:
            raise cpy.unsafe_get_error()
        var envelope = PythonObject(from_owned=raw)
        for i in range(len(self.arguments)):  # driver-loop: native ABI slots and owned references only
            var value = y if self.sources[i] == 1 else predicted if self.sources[i] == 2 else self.arguments[i]
            # SetItem steals the new reference, including on failure.
            if cpy.PyTuple_SetItem(envelope._obj_ptr, i, cpy.Py_NewRef(value._obj_ptr)) != 0:
                raise cpy.unsafe_get_error()
        # call-table: member metric (x_metrics door over Python arrays) has no native entry yet
        return family_call(entry, PythonObject(None), self.operation, envelope, self.metric_context)


def _scorer_column(predicted: PythonObject, context: PythonObject) raises -> PythonObject:
    var array = predicted
    var dtype = String(array.dtype)
    if dtype != "<f4" and dtype != "<f8":
        raise Error("TypeError: native probability output must be float32 or float64")
    var rows = Int(py=array.shape[0])
    var result = family_array(Python.tuple(rows), "<f4", context)
    _ = msel_proba_column(msel_helpers_key(context), Int(py=classical_address(array)),
                          Int(py=context["family_msel_codes"][array.dtype]), rows, 2, 1,
                          Int(py=classical_address(result)))
    _ = array  # keep the Python object alive through the native call above
    _ = result  # keep the Python object alive through the native call above
    return result


def msel_scorer_call(scorer: PythonObject, estimator: PythonObject, x: PythonObject, y: PythonObject,
                     sample_weight: PythonObject, memo: PythonObject, context: PythonObject) raises -> PythonObject:
    """A scorer called as `scorer(estimator, X, y)`: its signed value, or
    (lane mf-score6) the signed per-label array of an average=None metric,
    as scikit-learn's `_Scorer._score` returns `sign * score_func(...)`."""
    var prepared = MselMetricPrepared()
    return msel_scorer_object(scorer, estimator, x, y, sample_weight, memo, context, prepared)


def _scorer_response(fields: PythonObject, model: PythonObject, b: PythonObject) raises -> PythonObject:
    """The response method a scorer reads off an estimator of plan `model`:
    the first of its preference order the class has."""
    var methods = fields["_response_method"]
    if py_type(methods) is b.str:
        methods = Python.tuple(methods)
    elif py_type(methods) is not b.tuple:
        raise Error("TypeError: native scorer response_method requires a string or tuple of strings")
    # Optional, not a PythonObject read and reassigned in a loop (modular/modular#7546 pattern; runtime-fix-save-forest sweep).
    var found_method = Optional[PythonObject]()
    for k in range(len(methods)):  # driver-loop: fixed response preference order
        if py_type(methods[k]) is not b.str:
            raise Error("TypeError: native scorer response method names must be exact strings")
        if py_contains(model["methods"], methods[k]):
            found_method = methods[k]
            break
    if not found_method:
        raise Error("AttributeError: native estimator has none of the requested scorer response methods")
    return found_method.value()


def msel_scorer_response(scorer: PythonObject, model: PythonObject, context: PythonObject) raises -> PythonObject:
    """The response method name `scorer` reads off an estimator whose checked
    plan is `model` (model selection's native member units ask it once per
    scorer of an operation, to know which member output to place)."""
    return _scorer_response(msel_scorer_fields(scorer, context), model, native_loaded_module("builtins"))


def msel_scorer_value(scorer: PythonObject, estimator: PythonObject, x: PythonObject, y: PythonObject,
                     sample_weight: PythonObject, memo: PythonObject, context: PythonObject,
                      mut prepared: MselMetricPrepared) raises -> Float64:
    """The scorer's number, as model selection's `_score` reads a scorer's
    answer (scikit-learn model_selection/_validation.py `_score`): a real
    as is; an array (lane mf-score6: an average=None metric's per-label
    values) its one value when it holds one (`.item()`), else scikit-learn's
    refusal, raised whatever the error_score."""
    var result = msel_scorer_object(scorer, estimator, x, y, sample_weight, memo, context, prepared)
    if metric_is_real(result, context):
        return metric_float(result, context)
    var b = native_loaded_module("builtins")
    var typestr = _per_label_typestr(result, context)
    if typestr.byte_length() > 0 and Int(py=result.shape[0]) == 1:
        var value: Float64
        if typestr == "<f4":
            var one = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(result)))
            value = Float64(one[0])
        else:
            var one = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(result)))
            value = one[0]
        _ = result  # read through its address above
        return value
    raise Error("ValueError: scoring must return a number, got " + String(py=py_call(b.str, result)) + " ("
                + String(py=py_call(b.str, py_type(result))) + ") instead. (scorer="
                + String(py=py_call(b.repr, scorer)) + ")")


def msel_scorer_object(scorer: PythonObject, estimator: PythonObject, x: PythonObject, y: PythonObject,
                       sample_weight: PythonObject, memo: PythonObject, context: PythonObject,
                       mut prepared: MselMetricPrepared) raises -> PythonObject:
    """The scorer's signed answer as a Python value (a float, or a signed
    float64 per-label array; lane mf-score6)."""
    var b = native_loaded_module("builtins")
    var fields = msel_scorer_fields(scorer, context)
    var model = msel_estimator_plan(estimator, context)
    var method = _scorer_response(fields, model, b)
    if memo is not None:
        if py_type(memo) is not b.dict:
            raise Error("TypeError: native scorer prediction cache must be an exact dict")
        msel_admit(memo, context)
    var predicted = PythonObject(None) if memo is None else py_get(memo, method)
    if predicted is None:
        # `model` is the plan checked above; only scorer admission (no
        # Python code) ran since, so the dispatch reuses it (lane
        # perf-sweep-msel-nbr-svc: two full plan checks per score before).
        predicted = msel_planned_call(estimator, model, String(method), Python.tuple(x), context, True)
        if memo is not None:
            py_setitem(memo, method, predicted)
    return _scorer_metric(fields, estimator, model, method, predicted, y, sample_weight, memo, context, prepared, b)


def _scorer_metric(fields: PythonObject, estimator: PythonObject, model: PythonObject, method: PythonObject,
                   predicted_in: PythonObject, y: PythonObject, sample_weight: PythonObject, memo: PythonObject,
                   context: PythonObject, mut prepared: MselMetricPrepared, b: PythonObject) raises -> PythonObject:
    """The metric of one scorer over a computed response (the binary
    probability column, the keyword options, the sign): a Python float, or
    (lane mf-score6) an average=None metric's 1-D float64 per-label array,
    signed (`_signed_array`)."""
    var predicted = predicted_in
    var name = fields["_name"]
    if name is not None and py_type(name) is not b.str:
        raise Error("TypeError: native scorer name must be an exact string")
    if String(method) == "predict_proba" and py_contains(context["family_msel_binary_proba"], name):
        var array = msel_input(predicted, "pred", context)
        if Int(py=array.ndim) == 2 and Int(py=array.shape[1]) == 2:
            var column = PythonObject(None) if memo is None else py_get(memo, "predict_proba[:, 1]")
            if column is None:
                column = _scorer_column(array, context)
                if memo is not None:
                    py_setitem(memo, "predict_proba[:, 1]", column)
            predicted = column
    var keywords = fields["_kwargs"]
    if py_type(keywords) is not b.dict:
        raise Error("TypeError: native scorer keyword options must be an exact dict")
    msel_admit(keywords, context)
    if sample_weight is not None:
        keywords = py_call(b.dict, keywords)
        py_setitem(keywords, "sample_weight", sample_weight)
    var result = prepared.call(fields["_score_func"], y, predicted, keywords, context, estimator, model)
    classical_connect_numpy(context)
    var real = metric_is_real(result, context)
    var typestr = String("") if real else _per_label_typestr(result, context)
    if not real and typestr.byte_length() == 0:
        raise Error("TypeError: scoring must return a real scalar")
    var sign = fields["_sign"]
    if py_type(sign) is not b.int or (Int(py=sign) != 1 and Int(py=sign) != -1):
        raise Error("TypeError: native scorer sign must be 1 or -1")
    if not real:
        return _signed_array(result, typestr, Int(py=sign), context)
    return PythonObject(Float64(Int(py=sign)) * metric_float(result, context))


def _per_label_typestr(result: PythonObject, context: PythonObject) raises -> String:
    """Lane mf-score6: an average=None metric's answer, a 1-D float32 or
    float64 array (a mojolearn Array, or an ndarray when NumPy is
    connected): its typestr, else ""."""
    if not classical_is_array(result, context):
        var nk = classical_keys()
        var ndarray = py_get(context, nk[].k_ndarray)
        if ndarray is None or py_type(result) is not ndarray:
            return String("")
    if Int(py=result.ndim) != 1:
        return String("")
    var typestr = classical_dtype_str(result, context)
    if typestr == "<f4" or typestr == "<f8":
        return typestr
    return String("")


def _signed_array(result: PythonObject, typestr: String, sign: Int, context: PythonObject) raises -> PythonObject:
    """`sign * array` (scikit-learn's `_Scorer._score`): the metric's own
    array for sign 1, else a new array of its dtype with its values negated
    natively."""
    if sign == 1:
        return result
    var n = Int(py=result.shape[0])
    var negated = family_array(Python.tuple(n), typestr, context)
    if n > 0:
        if typestr == "<f4":
            var src = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(result)))
            var dst = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(negated)))
            for i in range(n):  # driver-loop: over the labels' values (the sign)
                dst[i] = -src[i]
        else:
            var src = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(result)))
            var dst = MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=Int(py=classical_address(negated)))
            for i in range(n):  # driver-loop: over the labels' values (the sign)
                dst[i] = -src[i]
    _ = result  # read through its address above
    _ = negated  # written through its address above
    return negated
