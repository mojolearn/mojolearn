# SPDX-License-Identifier: Apache-2.0
# Fixed scorer records call verified compiled estimator/metric entries. The
# Python function objects identify import-time plans; they are never invoked.
from bindings.native_loaded_module import native_loaded_module
from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr
from bindings.py_balanced import py_call, py_type, py_get, py_setitem, py_contains
from bindings.family_common import family_call, family_array
from bindings.classical_arrays import classical_address
from bindings.classical_attributes import classical_connect_numpy
from bindings.metrics_native_support import metric_is_real, metric_float
from bindings.msel_inputs import msel_input, msel_admit
from bindings.native_buffer_owner import native_storage_allocate
from bindings.msel_estimator_native import msel_estimator_call, msel_estimator_plan


def msel_metric_plan(function: PythonObject, context: PythonObject) raises -> PythonObject:
    if py_type(function) is not context["family_msel_metric_function_type"]:
        raise Error("TypeError: Python metric callbacks are not accepted; use a connected native mojolearn metric")
    var plan = py_get(context["family_msel_metric_plans"], function)
    if plan is None:
        raise Error("TypeError: Python metric callbacks are not accepted; use a connected native mojolearn metric")
    # Native function metadata avoids function.__code__'s audit callback while
    # preserving its connected-code identity check at every scoring boundary.
    ref cpy = Python().cpython()
    var code = cpy.lib.get_function[PyObjectPtr]("PyFunction_GetCode")(function._obj_ptr)
    if not code:
        raise cpy.unsafe_get_error()
    if code != plan["code"]._obj_ptr:
        raise Error("TypeError: the connected metric function was modified")
    return plan


def _scorer_class(context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = context["family_msel_scorer_class"]
    if py_type(cls) is not b.type:
        raise Error("TypeError: native scorer metadata refuses metaclass hooks")
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    if len(mro) != 2 or mro[1] is not b.object:
        raise Error("TypeError: native scorer metadata refuses inherited lifecycle hooks")
    var source = py_call(b.type.__dict__["__dict__"].__get__, cls)
    var hooks = context["family_msel_scorer_hooks"]
    for k in range(len(hooks)):  # driver-loop: connected scorer storage and method identities before allocation or access
        if py_get(source, hooks[k][0]) is not hooks[k][1]:
            raise Error("TypeError: native scorer methods or storage hooks changed")
    return cls


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
        raise Error("TypeError: Python scorer callbacks are not accepted; use get_scorer or make_scorer with a native metric")
    var fields = py_call(context["family_msel_scorer_fields"], scorer, cls)
    var keys = py_call(b.list, fields.keys())
    for k in range(len(keys)):  # driver-loop: exact scorer keys before metadata lookup
        if py_type(keys[k]) is not b.str:
            raise Error("TypeError: native scorer fields require exact string keys")
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
        var next_item = cpy.lib.get_function[Int32]("PyDict_Next")
        for i in range(len(value)):  # driver-loop: capture admitted option references once after preparation
            if next_item(value._obj_ptr, Pointer(to=position), Pointer(to=key), Pointer(to=item)) == 0:
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
        ref cpy = Python().cpython()
        var position = 0
        var key = PyObjectPtr()
        var item = PyObjectPtr()
        var next_item = cpy.lib.get_function[Int32]("PyDict_Next")
        for i in range(len(self.keys)):  # driver-loop: check mutable options after every prediction/GIL boundary
            if next_item(value._obj_ptr, Pointer(to=position), Pointer(to=key), Pointer(to=item)) == 0:
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
    # No estimator, fit state, y, prediction or result is retained here.
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

    def same(self, function: PythonObject, plan: PythonObject, keywords: PythonObject,
             context: PythonObject) raises -> Bool:
        if not self.ready or function is not self.function or plan is not self.plan:
            return False
        if (plan["recipe"] is not self.recipe or plan["inputs"] is not self.inputs
                or plan["allowed"] is not self.allowed or plan["entry"] is not self.entry
                or plan["operation"] is not self.operation or plan["context"] is not self.metric_context):
            return False
        if not self.defaults.same(plan["defaults"]) or not self.keywords.same(keywords):
            return False
        if self.uses_default_mode and context["family_msel_backend_fields"]["_DEFAULT_MODE"] is not self.default_mode:
            return False
        for i in range(len(self.binding_keys)):  # driver-loop: selected native binding owners remain current
            if context["family_msel_metric_bindings"][self.binding_keys[i]] is not self.binding_values[i]:
                return False
        return True

    def prepare(mut self, function: PythonObject, plan: PythonObject, y: PythonObject,
                predicted: PythonObject, keywords: PythonObject, context: PythonObject,
                b: PythonObject) raises:
        self.ready = False
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
                if String(clean) != "identical" and String(clean) != "fast":
                    raise Error("ValueError: numeric_mode must be 'fast' or 'identical'")
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
             keywords: PythonObject, context: PythonObject) raises -> PythonObject:
        var b = native_loaded_module("builtins")
        var plan = msel_metric_plan(function, context)
        var entry = plan["entry"]
        if entry is None or py_type(entry) is not py_type(b.len):
            raise Error("ImportError: native scorer metric binding is unavailable")
        if py_type(keywords) is not b.dict:
            raise Error("TypeError: native metric keywords must be an exact dict")
        msel_admit(keywords, context)
        if not self.same(function, plan, keywords, context):
            self.prepare(function, plan, y, predicted, keywords, context, b)
        # Fresh immutable call arguments: no tuple retained by an entry can be
        # mutated by a later phase, and dynamic y/predicted never enter template.
        ref cpy = Python().cpython()
        var raw = cpy.lib.get_function[PyObjectPtr]("PyTuple_New")(len(self.arguments))
        if not raw:
            raise cpy.unsafe_get_error()
        var envelope = PythonObject(from_owned=raw)
        var put = cpy.lib.get_function[Int32]("PyTuple_SetItem")
        for i in range(len(self.arguments)):  # driver-loop: native ABI slots and owned references only
            var value = y if self.sources[i] == 1 else predicted if self.sources[i] == 2 else self.arguments[i]
            # SetItem steals the new reference, including on failure.
            if put(envelope._obj_ptr, i, cpy.Py_NewRef(value._obj_ptr)) != 0:
                raise cpy.unsafe_get_error()
        return family_call(entry, PythonObject(None), self.operation, envelope, self.metric_context)


def msel_metric_call(function: PythonObject, y: PythonObject, predicted: PythonObject,
                     keywords: PythonObject, context: PythonObject) raises -> PythonObject:
    var prepared = MselMetricPrepared()
    return prepared.call(function, y, predicted, keywords, context)


def _scorer_column(predicted: PythonObject, context: PythonObject) raises -> PythonObject:
    var array = predicted
    var dtype = String(array.dtype)
    if dtype != "<f4" and dtype != "<f8":
        raise Error("TypeError: native probability output must be float32 or float64")
    var rows = Int(py=array.shape[0])
    var result = family_array(Python.tuple(rows), "<f4", context)
    var entry = context["family_msel_entries"]["msel_proba_column"]
    if py_type(entry) is not py_type(native_loaded_module("builtins").len):
        raise Error("ImportError: native probability column entry is unavailable")
    _ = family_call(entry, classical_address(array), context["family_msel_codes"][array.dtype],
                    PythonObject(rows), PythonObject(2), PythonObject(1), classical_address(result))
    _ = array  # keep the Python object alive through the native call above
    _ = result  # keep the Python object alive through the native call above
    return result


def msel_scorer_call(scorer: PythonObject, estimator: PythonObject, x: PythonObject, y: PythonObject,
                     sample_weight: PythonObject, memo: PythonObject, context: PythonObject) raises -> PythonObject:
    var prepared = MselMetricPrepared()
    return PythonObject(msel_scorer_value(scorer, estimator, x, y, sample_weight, memo, context, prepared))


def msel_scorer_value(scorer: PythonObject, estimator: PythonObject, x: PythonObject, y: PythonObject,
                     sample_weight: PythonObject, memo: PythonObject, context: PythonObject,
                      mut prepared: MselMetricPrepared) raises -> Float64:
    var b = native_loaded_module("builtins")
    var fields = msel_scorer_fields(scorer, context)
    var model = msel_estimator_plan(estimator, context)
    var methods = fields["_response_method"]
    if py_type(methods) is b.str:
        methods = Python.tuple(methods)
    elif py_type(methods) is not b.tuple:
        raise Error("TypeError: native scorer response_method requires a string or tuple of strings")
    var method = PythonObject(None)
    for k in range(len(methods)):  # driver-loop: fixed response preference order
        if py_type(methods[k]) is not b.str:
            raise Error("TypeError: native scorer response method names must be exact strings")
        if py_contains(model["methods"], methods[k]):
            method = methods[k]
            break
    if method is None:
        raise Error("AttributeError: native estimator has none of the requested scorer response methods")
    if memo is not None:
        if py_type(memo) is not b.dict:
            raise Error("TypeError: native scorer prediction cache must be an exact dict")
        msel_admit(memo, context)
    var predicted = PythonObject(None) if memo is None else py_get(memo, method)
    if predicted is None:
        predicted = msel_estimator_call(estimator, String(method), Python.tuple(x), context, True)
        if memo is not None:
            py_setitem(memo, method, predicted)
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
    var result = prepared.call(fields["_score_func"], y, predicted, keywords, context)
    classical_connect_numpy(context)
    if not metric_is_real(result, context):
        raise Error("TypeError: scoring must return a real scalar")
    var sign = fields["_sign"]
    if py_type(sign) is not b.int or (Int(py=sign) != 1 and Int(py=sign) != -1):
        raise Error("TypeError: native scorer sign must be 1 or -1")
    return Float64(Int(py=sign)) * metric_float(result, context)
