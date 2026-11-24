# SPDX-License-Identifier: Apache-2.0
# Direct native estimator ABI calls. Python only connects these fixed plans
# at package import; CV never calls a constructor, method or clone callback.
from bindings.native_array_storage_door import native_array_empty
from bindings.native_loaded_module_door import NativeModules, native_loaded_module
from bindings.native_class_admission_door import class_record, class_admit
from bindings.msel_plan_records_unpack import MSEL_PLAN_CLASS_SLOT, msel_fields_admitted, msel_fields_admit
from std.python import Python, PythonObject
from bindings.py_balanced import py_call, py_type, py_get, py_contains, py_setitem
from bindings.native_buffer_owner_pack import native_storage_allocate, native_buffer_view
from bindings.native_numpy_metadata_pack import native_numpy_type
from bindings.classical_arrays_door import classical_wrap_array, _store_code
from bindings.msel_inputs_unpack import msel_input
from bindings.classical_attributes_door import classical_connect_numpy
from bindings.classical_pandas_door import classical_admit_value
from bindings.family_common_pack import family_call
from bindings.sequential_admission_door import sequential_admit_value
from bindings.py_model_records import bump_model_generation
from bindings.mode_metadata_native_door import instance_protocol_override
from bindings.mode_answer_door import answer_mode_python
from core.mode_answer import MODE_FAMILY_MSEL
from core.device_route import device_cpu_fast
from bindings.msel_member_device_unpack import msel_member_host_folds


def msel_estimator_plan(model: PythonObject, context: PythonObject) raises -> PythonObject:
    var b = native_loaded_module("builtins")
    var cls = py_type(model)
    if py_type(cls) is not b.type:
        raise Error("TypeError: native cross-validation refuses Python estimator metaclass hooks")
    var plan = py_get(context["family_msel_estimators"], cls)
    if plan is None:
        # Only an exact import-connected built-in can be pending migration.
        # An external class may use any __module__ string (including our
        # namespace); that is neither a native identity nor an ABI promise.
        if py_contains(context["family_msel_pending_estimators"], cls):
            raise Error("RuntimeError: this built-in estimator's native cross-validation schema is pending migration")
        raise Error("TypeError: Python estimator callbacks are not accepted; use a connected native estimator")
    # perf2-msel: the class rows (ancestry and every connected hook identity)
    # are checked once per class state (native_class_admission type watchers;
    # bindings/msel_plan_records.mojo), the instance keys once per dictionary
    # state (dictionary watchers). Without watchers (Python < 3.12) both run on
    # every call as before.
    var mods = NativeModules()
    var class_held = class_record(mods, MSEL_PLAN_CLASS_SLOT, cls)
    if class_held is None or Int(py=class_held) != Int(plan._obj_ptr):
        _msel_check_class_rows(cls, plan, b)
        # The payload is the plan's address (plans live in the import-time
        # registry for the process), never the plan itself: it holds the MRO,
        # and a record must not strongly retain its class.
        class_admit(mods, MSEL_PLAN_CLASS_SLOT, cls, PythonObject(Int(plan._obj_ptr)))
    var fields = py_call(plan["fields_get"], model, cls)
    if msel_fields_admitted(model, fields, plan):
        return plan
    _msel_check_fields(fields, plan, b)
    if py_type(fields) is b.dict:
        msel_fields_admit(model, fields, plan)
    return plan


def _msel_check_class_rows(cls: PythonObject, plan: PythonObject, b: PythonObject) raises:
    var mro = py_call(b.type.__dict__["__mro__"].__get__, cls)
    if len(mro) != len(plan["mro"]):
        raise Error("TypeError: native estimator class ancestry changed")
    for k in range(len(mro)):  # driver-loop: connected estimator class ancestry
        if mro[k] is not plan["mro"][k]:
            raise Error("TypeError: native estimator class ancestry changed")
    var hooks = plan["hooks"]
    # The hook rows are grouped by class (python/mojolearn/_native_msel_dispatch.py
    # `register`: every protected name of one ancestor, then the next), so
    # each ancestor's live class mapping is read once per check, not once per
    # name (lane perf-meta-estimators, 2026-11-05: every ensemble member call
    # and every native clone runs this check). The same identities are tested.
    var class_dict = b.type.__dict__["__dict__"].__get__
    # Optional, not a PythonObject(None) sentinel: Mojo 1.2.0 fails to lower a loop-carried PythonObject read and reassigned in the loop (modular/modular#7546).
    var base = Optional[PythonObject]()
    var base_fields = Optional[PythonObject]()
    for k in range(len(hooks)):  # driver-loop: connected estimator method and storage identities
        var hook = hooks[k]
        var owner = hook[0]
        if not base or owner is not base.value():
            base_fields = py_call(class_dict, owner)
            base = owner
        if py_get(base_fields.value(), hook[1]) is not hook[2]:
            raise Error("TypeError: native estimator methods or storage hooks changed")


def _msel_check_fields(fields: PythonObject, plan: PythonObject, b: PythonObject) raises:
    var names = py_call(b.list, fields.keys())
    for k in range(len(names)):  # driver-loop: instance keys before dictionary lookup
        if py_type(names[k]) is not b.str:
            raise Error("TypeError: native estimator state requires exact string keys")
    var protected = plan["protected"]
    for k in range(len(protected)):  # driver-loop: refuse instance method/protocol overrides before dispatch
        # Shared predicate (mode_metadata_native): the trees' native
        # `_has_constructor_parameters` bool is admitted; the class identity
        # is still protected above.
        if instance_protocol_override(fields, protected[k], b):
            raise Error("TypeError: native estimator instance overrides a connected method or protocol")


def msel_estimator_call(model: PythonObject, method: String, args: PythonObject,
                        context: PythonObject, scorer_response: Bool = False) raises -> PythonObject:
    # PythonObject initialization is runtime binding work; default arguments
    # are evaluated at compile time by Mojo.
    return msel_estimator_call(model, method, args, context, scorer_response, PythonObject(None))


def msel_estimator_call(model: PythonObject, method: String, args: PythonObject,
                        context: PythonObject, scorer_response: Bool,
                        keywords: PythonObject) raises -> PythonObject:
    var plan = msel_estimator_plan(model, context)
    if method == "__admit__":
        # Validate constructor configuration without executing any of its
        # Python protocols. Native clone owns the recursive value schemas.
        _ = msel_native_clone(model, context)
        return PythonObject(None)
    return _msel_admitted_dispatch(model, method, args, context, plan, scorer_response, keywords)


def msel_planned_call(model: PythonObject, plan: PythonObject, method: String, args: PythonObject,
                      context: PythonObject, scorer_response: Bool = False) raises -> PythonObject:
    """`msel_estimator_call` for a model whose plan the caller has just
    checked (`msel_estimator_plan`), with
    no Python code run since: the class, hook and instance checks are not
    repeated (lane perf-sweep-msel-nbr-svc: a fold's clone was checked by the
    clone, again by its fit and twice per score)."""
    return _msel_admitted_dispatch(model, method, args, context, plan, scorer_response, PythonObject(None))


def _msel_admitted_dispatch(model: PythonObject, method: String, args: PythonObject,
                            context: PythonObject, plan: PythonObject,
                            scorer_response: Bool, keywords: PythonObject) raises -> PythonObject:
    # The caller has checked the exact connected class, its methods and native
    # storage layout through msel_estimator_plan. Admission is a separate
    # operation: reading a clone's constructor parameters must not re-enter
    # the general admission/clone dispatcher. Keep the ordinary method recipes,
    # native-entry checks and keyword/output handling shared with public calls.
    var b = native_loaded_module("builtins")
    var entry = plan["entry"]
    var method_plan = py_get(plan["methods"], PythonObject(method))
    if method_plan is None:
        # A member-only plan (python/mojolearn/_native_msel_more.py
        # connect_members: a marked class registered from its mark, without
        # method recipes) reached its own doors: its member could not serve
        # this request, so it is refused in the words the class had before
        # its mark. Read only on this refusal path, never per served call.
        if py_get(plan, PythonObject("member_only")) is not None:
            raise Error("RuntimeError: this built-in estimator's native cross-validation schema is pending migration")
        if entry is None or py_type(entry) is not py_type(b.len):
            raise Error("ImportError: native cross-validation estimator binding is unavailable")
        raise Error("AttributeError: native estimator has no " + method)
    if entry is None or py_type(entry) is not py_type(b.len):
        raise Error("ImportError: native cross-validation estimator binding is unavailable")
    if keywords is not None:
        # The method's keyword names, read only when keywords were passed
        # (lane ne2-b-members, N068: an empty dict and tuple were built as
        # lookup defaults on every call).
        var table = py_get(plan, "keywords")
        var allowed = PythonObject(None) if table is None else py_get(table, PythonObject(method))
        if allowed is None:
            allowed = Python.tuple()
        # This context is independent of the public prep owner's context.
        # Connect NumPy here before admitting a first-call ndarray keyword;
        # the positional inputs may not have reached a native owner yet.
        classical_connect_numpy(context)
        if py_type(keywords) is not b.dict:
            raise Error("TypeError: native estimator keywords require an exact dict")
        var keys = py_call(b.list, keywords.keys())
        for k in range(len(keys)):  # driver-loop: named fit arguments before native dispatch
            if py_type(keys[k]) is not b.str:
                raise Error("TypeError: native estimator keyword names must be exact strings")
            if not py_contains(allowed, keys[k]):
                raise Error("TypeError: " + String(py_type(model).__name__) + "." + method
                            + " got an unexpected keyword argument '" + String(keys[k]) + "'")
            classical_admit_value(keywords[keys[k]], context)
    if method == "fit" and py_contains(plan, "fit_entry"):
        var fit = plan["fit_entry"]
        if py_type(fit) is not py_type(b.len):
            raise Error("TypeError: native scaler fitting entry is unavailable")
        # call-table: member scaler (plan "fit_entry", a Python-object door) has no native entry yet
        return family_call(fit, model, args[0], args[1], py_get(keywords, "sample_weight")
                           if keywords is not None else PythonObject(None), plan["context"])
    var values = Python.list()
    var recipe = method_plan[1]
    # The model's mode is read once per call, at the first binding item
    # (lane ne2-b-members, N068: the fields and mode were read again for
    # every binding item of the recipe). Same refusals in the same order.
    # Optional, not a PythonObject read and reassigned in a loop (modular/modular#7546 pattern; runtime-fix-save-forest sweep).
    # The mode's presence is the read-once flag.
    var read_mode = Optional[PythonObject]()
    # The member's device, read with its mode (same once-per-call read, same
    # dictionary): a member that says device="cpu" beside a GPU set is handed
    # the CPU identical module of the binding name, connected at import
    # (`family_msel_bindings_cpu`), never the process's GPU module. Only the
    # recipes that name a binding read it (the GP / kernel-ridge and KNN
    # members); every other member's own door routes by its estimator.
    var host = False
    for k in range(len(recipe)):  # driver-loop: fixed native ABI arguments
        var item = recipe[k]
        var kind = String(item[0])
        var value = item[1]
        if kind == "arg":
            value = args[Int(py=item[1])]
        elif kind == "keyword":
            value = py_get(keywords, item[1], item[2]) if keywords is not None else item[2]
        elif kind == "binding":
            if not read_mode:
                var fields = py_call(plan["fields_get"], model, py_type(model))
                var stored = py_get(fields, "numeric_mode")
                if stored is None:
                    stored = context["family_msel_backend_fields"]["_DEFAULT_MODE"]
                if py_type(stored) is not b.str:
                    raise Error("TypeError: native estimator numeric_mode must be an exact string or None")
                # Read once per call (the flag above): the word `fast`, where
                # the identical set answers it (core/mode_answer.mojo, family
                # MSEL), picks the identical row of the binding table.
                answer_mode_python(stored, MODE_FAMILY_MSEL)
                read_mode = stored
                host = msel_member_host_folds(fields, b, context)
                if host and String(stored) != "identical":
                    device_cpu_fast()
            var mode = read_mode.value()
            if host:
                var cpu = py_get(context, "family_msel_bindings_cpu")
                value = PythonObject(None) if cpu is None else py_get(cpu, item[1])
                if value is None:
                    raise Error("ImportError: device='cpu' is not connected for " + String(item[1])
                                + " in this installation")
            else:
                var modes = context["family_msel_bindings"][item[1]]
                value = py_get(modes, mode)
                if value is None:
                    raise Error("ImportError: native estimator binding is unavailable for " + String(mode))
            if py_type(value) is b.str:
                raise Error(String(value))
        _ = py_call(values.append, value)
    var envelope = py_call(b.tuple, values)
    if method == "fit" and plan["forest"] is not None:
        var fields = py_call(plan["fields_get"], model, py_type(model))
        var before = py_get(fields, "_mojolearn_fitting", PythonObject(0))
        if py_type(before) is not b.int:
            raise Error("TypeError: native estimator fit depth must be an exact integer")
        var depth = Int(py=before)
        py_setitem(fields, "_mojolearn_fitting", PythonObject(depth + 1))
        var result = PythonObject(None)
        try:
            # call-table: member forest estimator (family door over the Python model) has no native entry yet
            result = family_call(entry, model, method_plan[0], envelope, plan["context"])
        finally:
            fields = py_call(plan["fields_get"], model, py_type(model))
            if depth > 0:
                py_setitem(fields, "_mojolearn_fitting", PythonObject(depth))
            else:
                _ = py_call(fields.pop, "_mojolearn_fitting", PythonObject(None))
        py_setitem(fields, "_mojolearn_fitted", PythonObject(True))
        return result
    var operation = method_plan[0]
    var override = py_get(py_get(plan, "method_entries", Python.dict()), PythonObject(method))
    if override is not None:
        if py_type(override[0]) is not py_type(b.len):
            raise Error("ImportError: native estimator method binding is unavailable")
        # call-table: member estimator method override (family door over the Python model) has no native entry yet
        return family_call(override[0], model, operation, envelope, override[1])
    if scorer_response and method == "predict":
        var internal_operation = py_get(plan, "scorer_predict_operation")
        if internal_operation is not None:
            operation = internal_operation
    # call-table: member estimator (family door over the Python model) has no native entry yet
    return family_call(entry, model, operation, envelope, plan["context"])


def msel_clone_parameter(value: PythonObject, context: PythonObject, depth: Int = 0) raises -> PythonObject:
    if depth > 64:
        raise Error("TypeError: native estimator configuration exceeds the supported nesting depth")
    var b = native_loaded_module("builtins")
    var kind = py_type(value)
    if value is None or kind is b.bool or kind is b.int or kind is b.float or kind is b.complex or kind is b.str or kind is b.bytes or kind is b.range:
        return value
    if kind is b.type and py_contains(context["family_msel_splitter_layouts"], value):
        # Repeated splitters store their exact native splitter class as data.
        return value
    if kind is b.dict:
        var result = Python.dict()
        var pairs = py_call(b.list, value.items())
        for k in range(len(pairs)):  # driver-loop: native constructor parameter mapping
            var key = msel_clone_parameter(pairs[k][0], context, depth + 1)
            py_setitem(result, key, msel_clone_parameter(pairs[k][1], context, depth + 1))
        return result
    if kind is b.list or kind is b.tuple or kind is b.set or kind is b.frozenset:
        var items = py_call(b.list, value)
        var result = Python.list()
        for k in range(len(items)):  # driver-loop: native constructor parameter containers
            _ = py_call(result.append, msel_clone_parameter(items[k], context, depth + 1))
        return result if kind is b.list else py_call(kind, result)
    classical_connect_numpy(context)
    var scalars = context["scalar_types"]
    for k in range(len(scalars)):  # driver-loop: connected immutable NumPy scalar classes
        if kind is scalars[k]:
            return value
    if kind is context["array_class"]:
        sequential_admit_value(value, context)
        var packed = msel_input(value, "constructor parameter", context)
        var store = native_array_empty(String(_store_code(String(packed.dtype))))
        _ = py_call(store.frombytes, py_call(packed._mv.tobytes))
        _ = packed  # keep the native buffer owner alive through the copy
        return classical_wrap_array(store, packed.shape, String(packed.dtype), "C", context)
    if kind is py_get(context, "ndarray"):
        sequential_admit_value(value, context)
        # Numeric primitive dtype bytes own no Python object references.
        # Copy the admitted native buffer, then wrap independent storage;
        # neither ndarray.copy nor deepcopy/NumPy Python hooks are called.
        if Bool(py=value.dtype.hasobject):
            raise Error("TypeError: native constructor buffers cannot contain Python object references")
        var view = native_buffer_view(value)
        var order = PythonObject("F" if Bool(py=view.f_contiguous) and not Bool(py=view.c_contiguous) else "C")
        var store = py_call(b.bytearray, py_call(view.tobytes, order))
        var cls = native_numpy_type("ndarray", context)
        return py_call(cls, view.shape, dtype=value.dtype, buffer=store, order=order)
    if kind is context["array_type"]:
        var result = native_array_empty(String(value.typecode))
        var view = native_buffer_view(value)
        _ = py_call(result.frombytes, py_call(view.tobytes))
        _ = view  # keep the acquired native buffer alive through the copy
        return result
    if kind is b.bytearray:
        return py_call(b.bytearray, value)
    if py_type(kind) is b.type and py_contains(context["family_msel_estimators"], kind):
        return msel_native_clone(value, context, depth + 1)
    if kind is py_get(context, "family_msel_scorer_class"):
        # A native scorer (get_scorer / make_scorer) held as a parameter: a
        # search's `scoring`, cloned when the search itself is a member
        # (bindings/msel_member_units_door.mojo `msel_member_refit` clones
        # the search; lane mf-gate-fix-d: before this branch it reached the
        # Sequential admission's TypeError). scikit-learn's clone deep-copies
        # a non-estimator parameter: a fresh scorer of the exact class whose
        # fields are clones of the given's; its metric function is an
        # import-time native plan, kept by identity and never invoked here.
        # The ancestry check is the scorer release's (msel_search_state_door).
        var ancestry = py_call(b.type.__dict__["__mro__"].__get__, kind)
        if len(ancestry) != 2 or ancestry[1] is not b.object:
            raise Error("TypeError: native scorer cloning refuses inherited lifecycle hooks")
        # A scorer of the scorer table (get_scorer(name)) is a shared, never
        # written import-time record: kept by identity, which is how the
        # search kind reads its name back (providers/python/
        # member_ensemble_boundary_door.mojo `_scorer_name`).
        var table = py_call(b.list, py_call(context["family_msel_scorers"].values))
        for k in range(len(table)):  # driver-loop: over the scorer table (a fixed table, once per scorer parameter)
            if table[k] is value:
                return value
        var source = py_call(context["family_msel_scorer_fields"], value, kind)
        if py_type(source) is not b.dict:
            raise Error("TypeError: native scorer fields must be an exact dict")
        var scorer = native_storage_allocate(kind)
        var fields = py_call(context["family_msel_scorer_fields"], scorer, kind)
        var pairs = py_call(b.list, source.items())
        for k in range(len(pairs)):  # driver-loop: over the scorer's fields (five, fixed)
            var key = pairs[k][0]
            if py_type(key) is not b.str:
                raise Error("TypeError: native scorer fields require exact string keys")
            if String(py=key) == "_score_func":
                py_setitem(fields, key, pairs[k][1])
            else:
                py_setitem(fields, key, msel_clone_parameter(pairs[k][1], context, depth + 1))
        return scorer
    # A FUNCTION parameter (a callback, a refit policy, a schedule; lane
    # w3-fix-b row 14): scikit-learn's clone keeps it by identity
    # (`copy.deepcopy` treats functions and builtins as atomic), so the clone
    # carries the reference and never calls it; the estimator's or the
    # search's own door then refuses it in its words. Read only for a value
    # every branch above declined (no cost on the native parameters).
    var types = native_loaded_module("types")
    if kind is types.FunctionType or kind is types.BuiltinFunctionType:
        return value
    # A BUILT-IN LEARNING-RATE SCHEDULE (StepLR, ExponentialLR, OneCycleLR, ConstantLR, WarmupLinearLR,
    # WarmupCosineLR; lane w3f2-rfe, failures row A: a recurrent estimator's `lr_schedule` in a search raised
    # the Sequential admission's words below). scikit-learn's clone deep-copies it: a fresh instance of the
    # exact class with the same settings. Here: the optimizer owner's own compiled getstate and setstate
    # (the pickle and copy pair, bindings/optimizer_schedule_native_door.mojo), which copy the stored
    # settings and bind the fresh instance's native rate block; no Python method runs. The entry and its
    # context are connected at import (python/mojolearn/_families.py). Read only for a value every branch
    # above declined.
    var schedules = py_get(context, "family_msel_schedule_context")
    if schedules is not None:
        var classes = schedules["schedules"]
        for k in range(len(classes)):  # driver-loop: over the connected built-in schedule classes (six)
            if kind is classes[k]:
                var entry = context["family_msel_schedule_entry"]
                if py_type(entry) is not py_type(b.len):
                    raise Error("ImportError: native schedule cloning requires the compiled optimizer entry")
                var state = py_call(entry, PythonObject("schedule_getstate"), value, Python.tuple(),
                                    PythonObject(None), schedules)
                var fresh = native_storage_allocate(kind)
                _ = py_call(entry, PythonObject("schedule_setstate"), fresh, Python.tuple(state),
                            PythonObject(None), schedules)
                return fresh
    # Never invoke deepcopy/__reduce__/buffer protocols of an arbitrary
    # object. Additional native buffer/schema migration remains explicit
    # debt rather than sharing mutable constructor or fitted state.
    sequential_admit_value(value, context)
    raise Error("RuntimeError: native cloning of this constructor parameter remains pending migration")


def msel_native_clone(model: PythonObject, context: PythonObject, depth: Int = 0) raises -> PythonObject:
    if depth > 64:
        raise Error("TypeError: native estimator configuration exceeds the supported nesting depth")
    var b = native_loaded_module("builtins")
    var plan = msel_estimator_plan(model, context)
    var names = plan["parameters"]
    var given: PythonObject
    if plan["forest"] is not None:
        # call-table: member forest get_params (trees door over the Python model) has no native entry yet
        given = family_call(plan["entry"], model, "trees_forest_get_params", Python.tuple(), plan["context"])
    elif plan["init"] is None or not py_contains(plan["methods"], "get_params"):
        # A native initializer without a get_params recipe (KernelDensity,
        # ARIMA, Cholesky): its constructor stores each parameter under its
        # own name, so the stored fields are the parameters.
        given = py_call(plan["fields_get"], model, py_type(model))
    else:
        given = _msel_admitted_dispatch(model, "get_params", Python.tuple(), context,
                                        plan, False, PythonObject(None))
    var parameters = Python.dict()
    var args = Python.list()
    for k in range(len(names)):  # driver-loop: fixed constructor parameters, no fitted state
        if not py_contains(given, names[k]):
            raise Error("TypeError: native constructor parameter is missing from estimator state")
        var value = msel_clone_parameter(given[names[k]], context, depth + 1)
        py_setitem(parameters, names[k], value)
        _ = py_call(args.append, value)
    var result = native_storage_allocate(py_type(model))
    if plan["forest"] is not None:
        var fields = py_call(plan["fields_get"], result, py_type(result))
        _ = py_call(fields.update, parameters)
        py_setitem(fields, "_has_constructor_parameters", PythonObject(True))
        # Public forest constructors store parameters verbatim; native fit
        # derives/validates configuration in its original owner later.
    elif plan["init"] is None:
        # Exact fixed constructor schemas containing only field assignments.
        var fields = py_call(plan["fields_get"], result, py_type(result))
        _ = py_call(fields.update, parameters)
        var fixed = py_get(plan, "initial_fields")
        if fixed is not None:
            _ = py_call(fields.update, msel_clone_parameter(fixed, context))
    else:
        var entry = plan["entry"]
        if entry is None or py_type(entry) is not py_type(b.len):
            raise Error("ImportError: native estimator constructor binding is unavailable")
        # call-table: member estimator constructor (family door over the Python model) has no native entry yet
        _ = family_call(entry, result, plan["init"], py_call(b.tuple, args), plan["context"])
    return result


def msel_native_set_params(model: PythonObject, params: PythonObject,
                           context: PythonObject) raises -> PythonObject:
    # Generation word (bindings/py_model_records.mojo, lane ne1-e-records):
    # every stamped record of this object misses from here on.
    _ = bump_model_generation(model)
    var b = native_loaded_module("builtins")
    var plan = msel_estimator_plan(model, context)
    if py_type(params) is not b.dict:
        raise Error("TypeError: native estimator parameters require a native dict")
    var keys = py_call(b.list, params.keys())
    for k in range(len(keys)):  # driver-loop: parameter admission before hashing or native setter dispatch
        if py_type(keys[k]) is not b.str:
            raise Error("TypeError: native estimator parameter names must be exact strings")
        sequential_admit_value(params[keys[k]], context)
    if len(keys) == 0:
        return model
    if plan["init"] is None and plan["forest"] is None and not py_contains(plan["methods"], PythonObject("set_params")):
        # These exact constructors only assign their stored parameter schema.
        # Native search can reset them without a Python ParamsMixin call.
        var fields = py_call(plan["fields_get"], model, py_type(model))
        var names = plan["parameters"]
        var given = Python.dict()
        for k in range(len(names)):  # driver-loop: retain constructor configuration only
            py_setitem(given, names[k], fields[names[k]])
        for k in range(len(keys)):  # driver-loop: validate all names before mutating state
            if not py_contains(names, keys[k]):
                raise Error("ValueError: invalid native estimator parameter " + String(keys[k]))
            py_setitem(given, keys[k], msel_clone_parameter(params[keys[k]], context))
        _ = py_call(fields.clear)
        _ = py_call(fields.update, given)
        var fixed = py_get(plan, "initial_fields")
        if fixed is not None:
            _ = py_call(fields.update, msel_clone_parameter(fixed, context))
        return model
    if plan["forest"] is None:
        return msel_estimator_call(model, "set_params", Python.tuple(params), context)
    # call-table: member forest get_params (trees door over the Python model) has no native entry yet
    var values = family_call(plan["entry"], model, "trees_forest_get_params", Python.tuple(), plan["context"])
    var unknown = Python.list()
    for k in range(len(keys)):  # driver-loop: complete forest parameter name validation before mutation
        if not py_contains(values, keys[k]):
            _ = py_call(unknown.append, keys[k])
    if len(unknown):
        raise Error("Invalid parameter(s) " + String(py_call(b.repr, py_call(b.sorted, unknown)))
                    + " for " + String(py_type(model).__name__))
    _ = py_call(values.update, params)
    var fields = py_call(plan["fields_get"], model, py_type(model))
    _ = py_call(fields.clear)
    _ = py_call(fields.update, values)
    py_setitem(fields, "_has_constructor_parameters", PythonObject(True))
    return model
