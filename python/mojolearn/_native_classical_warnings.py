# SPDX-License-Identifier: Apache-2.0
"""Connection metadata for native classical warning delivery."""


def connect_warnings():
    # glue: cache standard native warning/IO metadata once during ABI setup
    import _codecs
    import io
    import linecache
    import os
    import re
    import sys
    import warnings
    from types import BuiltinFunctionType, GetSetDescriptorType, FunctionType, WrapperDescriptorType, MethodDescriptorType, CodeType
    from . import _sk_validate, _mode

    fields = vars(warnings)
    functions = {name: fields[name] for name in ("showwarning", "formatwarning", "_showwarnmsg", "_showwarnmsg_impl", "_formatwarnmsg", "_formatwarnmsg_impl")}  # glue: capture standard warning hooks without ever calling them at runtime
    module_fields = functions["_formatwarnmsg_impl"].__globals__
    standard = (functions["showwarning"] is fields.get("_showwarning_orig")
                and functions["formatwarning"] is fields.get("_formatwarning_orig"))
    for name in ("_showwarnmsg", "_formatwarnmsg", "_formatwarnmsg_impl"):  # glue: record standard display-hook admission without calling the hooks
        function = functions[name]
        standard = standard and type(function) is FunctionType and function.__name__ == name and function.__code__.co_filename == functions["formatwarning"].__code__.co_filename
    record = warnings.WarningMessage
    descriptor = vars(record).get("__dict__")
    if type(record) is not type or type(descriptor) is not GetSetDescriptorType:
        raise TypeError("warnings require the standard native record dictionary")
    category = _sk_validate._conversion_warning()
    native_slots = {}
    for name in ("__new__", "__init__", "__str__", "__getattribute__"):  # glue: validate standard exception slots at connection
        native_slots[name] = getattr(category, name)
        if type(native_slots[name]) not in (BuiltinFunctionType, WrapperDescriptorType, MethodDescriptorType):
            raise TypeError("warning exception slots must be native")
    # Builtin warning bases own distinct native __new__ entries and, on
    # CPython 3.11, distinct native __init__ descriptors too. ComplexWarning
    # inherits RuntimeWarning rather than DataConversionWarning's UserWarning.
    # Retain only these exact builtin slots, never arbitrary native methods.
    warning_bases = (Warning, UserWarning, DeprecationWarning, PendingDeprecationWarning,
                     SyntaxWarning, RuntimeWarning, FutureWarning, ImportWarning, UnicodeWarning,
                     BytesWarning, ResourceWarning, EncodingWarning)
    warning_new = tuple(base.__new__ for base in warning_bases)  # glue: fixed builtin warning constructor identities
    warning_init = tuple(base.__init__ for base in warning_bases)  # glue: fixed builtin warning initializer identities
    if any(type(entry) is not BuiltinFunctionType for entry in warning_new):  # glue: native constructor identity admission at import
        raise TypeError("warning constructors must be native")
    if any(type(entry) is not WrapperDescriptorType for entry in warning_init):  # glue: native initializer identity admission at import
        raise TypeError("warning initializers must be native")
    guarded = next(code for code in _mode._guard_cpu_training.__code__.co_consts if type(code) is CodeType and code.co_name == "guarded")  # glue: retain the established warning location from the legacy wrapper code
    location_line = max(line for _, _, line in guarded.co_lines() if line is not None)  # glue: native warning location is the wrapper's final call line
    context = module_fields.get("_Context")
    global_context = module_fields.get("_global_context")
    context_descriptor = None if context is None else vars(context).get("__dict__")
    getframe = sys._getframe
    if type(getframe) is not BuiltinFunctionType:
        raise TypeError("warnings require the native frame metadata entry")
    return dict(warning_fields=fields, warning_functions=functions, warning_standard_hooks=standard,
                warning_category=category, warning_category_slots=native_slots,
                warning_native_new=warning_new, warning_native_init=warning_init, warning_base=Warning,
                warning_category_default=_sk_validate.DataConversionWarning,
                warning_record=record, warning_record_fields=descriptor,
                warning_context_class=context, warning_context_fields=context_descriptor,
                warning_global_context=global_context,
                warning_context_var=fields.get("_warnings_context"),
                warning_context_enabled=bool(getattr(sys.flags, "context_aware_warnings", 0)),
                warning_getframe=getframe, warning_sys_fields=vars(sys),
                warning_filename=guarded.co_filename, warning_lineno=location_line,
                warning_module=_mode.__name__, warning_source_line=linecache.getline(guarded.co_filename, location_line).strip(),
                warning_package_prefix=os.path.dirname(__file__) + os.sep,
                warning_pattern_type=re.Pattern, warning_pattern_match=re.Pattern.match,
                warning_seen={}, warning_filter_owner=None, warning_filter_rows=(),
                warning_stringio=io.StringIO, warning_stringio_write=io.StringIO.write,
                warning_textio=io.TextIOWrapper, warning_textio_write=io.TextIOWrapper.write,
                warning_textio_buffer=io.TextIOWrapper.buffer,
                warning_textio_encoding=io.TextIOWrapper.encoding,
                warning_textio_errors=io.TextIOWrapper.errors,
                warning_buffered=io.BufferedWriter, warning_buffered_raw=io.BufferedWriter.raw,
                warning_fileio=io.FileIO,
                warning_lookup_error=_codecs.lookup_error,
                warning_native_function=BuiltinFunctionType, warning_python_function=FunctionType,
                warning_stdlib_file=functions["_formatwarnmsg_impl"].__code__.co_filename)
