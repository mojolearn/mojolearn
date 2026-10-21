# SPDX-License-Identifier: Apache-2.0
"""Import-time installation plan for the native linear-model public ABI."""
# No installation occurs at module import yet. Qualification calls _install
# on the real classes in a temporary test scope; public activation is separate.


def _methods(entry, context, *, ridge, host):
    # glue: create fixed-signature one-call shells once during ABI connection
    def ridge_init(self, *, alpha=1.0, solver="auto", fit_intercept=True,
                   normalize=False, numeric_mode=None):
        return entry(self, "initialize", (alpha, solver, fit_intercept, normalize, numeric_mode), context)

    def linear_init(self, *, fit_intercept=True, numeric_mode=None):
        return entry(self, "initialize", (fit_intercept, numeric_mode), context)

    def ridge_fit(self, X, y, sample_weight=None):
        return entry(self, "ridge_fit", (X, y, sample_weight), context)

    def linear_fit(self, X, y, sample_weight=None):
        return entry(self, "ols_fit", (X, y, sample_weight), context)

    def predict(self, X):
        return entry(self, "predict_public", (X,), context)

    def score(self, X, y):
        return entry(self, "score_public", (X, y), context)

    def save(self, path):
        return entry(self, "save", (path,), context)

    def load(cls, path):
        return entry(cls, "load", (path,), context)

    def get_params(self, deep=True):
        return entry(self, "get_params", (deep,), context)

    def set_params(self, **params):
        return entry(self, "set_params", (params,), context)

    def param_names(cls):
        return entry(cls, "param_names", (), context)

    def is_fitted(self):
        return entry(self, "is_fitted", (), context)

    def sklearn_tags(self):
        return entry(self, "sklearn_tags", (), context)

    def vendor_used(self):
        return entry(self, "vendor_used", (), context)

    def numeric_mode_used(self):
        return entry(self, "numeric_mode_used", (), context)

    def model_sha256(self):
        return entry(self, "model_sha256", (), context)

    methods = {"__init__": ridge_init if ridge else linear_init,
               "fit": ridge_fit if ridge else linear_fit, "predict": predict,
               "score": score, "save": save, "load": load,
               "get_params": get_params, "set_params": set_params,
               "_get_param_names": param_names, "__sklearn_is_fitted__": is_fitted,
               "__sklearn_tags__": sklearn_tags, "vendor_used": vendor_used,
               "numeric_mode_used": numeric_mode_used}
    if host:
        methods["model_sha256"] = model_sha256
    return methods


def _copy_metadata(function, original, cls, name):
    # glue: preserve public inspection/docs without following old runtime shells
    import inspect
    function.__name__ = name
    function.__qualname__ = cls.__qualname__ + "." + name
    function.__module__ = cls.__module__
    function.__doc__ = original.__doc__
    function.__annotations__ = dict(getattr(original, "__annotations__", {}))
    # A bound classmethod signature omits cls; retain its raw function signature.
    function.__signature__ = inspect.signature(getattr(original, "__func__", original))


def _layout(cls, connected_functions, attribute_descriptor):
    # glue: cache only verified native descriptor binders and our fixed wrappers
    from types import (FunctionType, GetSetDescriptorType, MemberDescriptorType,
                       MethodDescriptorType, WrapperDescriptorType)
    mro = type.__dict__["__mro__"].__get__(cls)
    dictionaries = tuple(type.__dict__["__dict__"].__get__(base) for base in mro)  # glue: read native class metadata once
    if type(cls) is not type or any(type(base) is not type for base in mro):  # glue: exact native class layout admission
        raise TypeError("native classical installation requires ordinary Python classes")
    fields_descriptor = next((fields["__dict__"] for fields in dictionaries if "__dict__" in fields), None)  # glue: locate the native instance dictionary member
    if type(fields_descriptor) is not GetSetDescriptorType:
        raise TypeError("native classical installation requires a native instance dictionary")
    names = {}
    for fields in dictionaries:  # glue: retain class lookup precedence at setup
        for name, value in fields.items():  # glue: fixed class attribute metadata
            names.setdefault(name, value)
    bindings = {}
    native_descriptors = (GetSetDescriptorType, MemberDescriptorType,
                          MethodDescriptorType, WrapperDescriptorType)
    for name, value in names.items():  # glue: verify callable descriptor origin at setup
        if type(value) in native_descriptors or value is attribute_descriptor:
            bindings[name] = (value, value.__get__)
        elif type(value) is FunctionType and value in connected_functions:
            bindings[name] = (value, value.__get__)
        elif type(value) in (classmethod, staticmethod) and value.__func__ in connected_functions:
            bindings[name] = (value, value.__get__)
    return {"mro": mro, "fields_descriptor": fields_descriptor,
            "fields_get": fields_descriptor.__get__, "bindings": bindings}


def _install(ridge_class, linear_class, host_ridge_class, host_linear_class):
    """Connect all metadata now; the package does not call this installer yet."""
    # glue: optional qualification/setup entry, never called by public methods
    from types import BuiltinFunctionType, MethodType
    from ._native_classical_bridge import connect, connect_host
    from .tokenizer import _binding

    entry, context = connect(ridge_class, linear_class)
    host_entry, host_context = connect_host(host_ridge_class, host_linear_class)
    if entry is not host_entry or not isinstance(entry, BuiltinFunctionType):
        raise TypeError("public classical methods require one shared compiled dispatcher")
    context["classical_host_context"] = host_context
    module = _binding()._module
    attribute_entry = module.classical_attribute_call
    descriptor_entry = module.classical_attribute_descriptor
    if not isinstance(attribute_entry, BuiltinFunctionType) or not isinstance(descriptor_entry, BuiltinFunctionType):
        raise TypeError("classical attributes require compiled ABI entries")
    attribute_descriptor = descriptor_entry(MethodType(attribute_entry, context))
    classes = ((ridge_class, True, False), (linear_class, False, False),
               (host_ridge_class, True, True), (host_linear_class, False, True))
    original = {}
    connected_functions = []
    try:
        for cls, ridge, host in classes:  # glue: connect the four exact model classes
            methods = _methods(entry, context, ridge=ridge, host=host)
            stored = dict(vars(cls))
            original[cls] = {name: (name in stored, stored.get(name)) for name in (*methods, "__getattribute__")}  # glue: reversible setup snapshot
            for name, function in methods.items():  # glue: fixed class method metadata installation
                previous = getattr(cls, name)
                _copy_metadata(function, previous, cls, name)
                connected_functions.append(function)
                setattr(cls, name, classmethod(function) if name in ("load", "_get_param_names") else function)
            setattr(cls, "__getattribute__", attribute_descriptor)
        layouts = {cls: _layout(cls, connected_functions, attribute_descriptor) for cls, _, _ in classes}  # glue: native descriptor metadata connection
        context.update(classical_public_layouts=layouts, classical_public_installed=True)
        host_context.update(classical_public_layouts=layouts, classical_public_installed=True)
    except BaseException:
        _restore(original)
        raise
    return {"original": original, "context": context, "host_context": host_context,
            "attribute_descriptor": attribute_descriptor}


def _restore(original):
    # glue: setup rollback; qualification teardown also uses this exact snapshot
    for cls, methods in reversed(tuple(original.items())):  # glue: reverse class metadata installation
        for name, (existed, value) in methods.items():  # glue: restore exact declared class attributes
            if existed:
                setattr(cls, name, value)
            else:
                delattr(cls, name)
