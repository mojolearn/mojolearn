# SPDX-License-Identifier: Apache-2.0
"""Shared import-time metadata for native ndarray protocol handling."""


def connect_array_protocol(protocol, module_name, *, warning_lineno=None):
    # glue: fixed native warning, regex and ContextVar identities; no NumPy or
    # sequence device import is needed to connect an array-protocol owner.
    from _contextvars import ContextVar
    from ._native_classical_warnings import connect_warnings
    import re
    context = {}
    context["sequence_attribute_error"] = AttributeError  # glue: fixed native missing-member category
    context["sequence_print_context_type"] = ContextVar
    context["sequence_print_context_get"] = ContextVar.get
    warning = connect_warnings()
    code = protocol.__code__
    warning.update(warning_filename=code.co_filename,
                   warning_lineno=(max(line for _, _, line in code.co_lines() if line is not None)
                                   if warning_lineno is None else warning_lineno),  # glue: retain the public array-protocol warning location
                   warning_module=module_name, warning_source_line="")
    context["sequence_array_warnings"] = warning
    context["sequence_dtype_pattern"] = re.compile(r"[<>=|]?(?:[?bBhHiIuUlLqQpPnNefdgFDGOSTVcmM][0-9]*(?:\[[0-9a-zA-Z]+\])?|(?:datetime64|timedelta64)(?:\[[0-9a-zA-Z]+\])?)")
    context["sequence_dtype_fullmatch"] = re.Pattern.fullmatch
    context["sequence_dtype_match"] = re.Pattern.match
    context["sequence_dtype_compact"] = re.compile(r"([<>|=]?)( *[(]?[ ,0-9]*[)]? *)([<>|=]?)([A-Za-z0-9.?]*(?:\[[a-zA-Z0-9,.]+\])?)")
    context["sequence_dtype_separator"] = re.compile(r"\s*,\s*")
    context["sequence_dtype_space"] = re.compile(r"\s+$")
    return context
