# SPDX-License-Identifier: Apache-2.0
"""Import-time connection for native fit-depth scalar state."""


def connect(context):
    # glue: retain the established wrapper's native warning locations and
    # builtin identities without importing optional NumPy or running a fit
    import builtins
    import dis
    import linecache
    from types import CodeType, ModuleType
    from . import _mode

    tracked = next(code for code in _mode._track_fit.__code__.co_consts  # glue: inspect fixed wrapper code metadata at package connection
                   if type(code) is CodeType and code.co_name == "tracked")
    lines = {}
    for instruction in dis.get_instructions(tracked):  # glue: exact old arithmetic/comparison source locations
        if instruction.opname == "BINARY_OP" and instruction.argrepr in ("+", "-"):
            lines["subtract" if instruction.argrepr == "-" else "add"] = instruction.positions.lineno
        elif instruction.opname == "COMPARE_OP" and instruction.argval == ">":
            lines["greater"] = instruction.positions.lineno
    settings = {}
    for operation in ("add", "subtract", "greater"):  # glue: separate native warning filters/repetition metadata
        warning = dict(context)
        warning.update(warning_filename=tracked.co_filename,
                       warning_lineno=lines[operation], warning_module=_mode.__name__,
                       warning_source_line=linecache.getline(tracked.co_filename, lines[operation]).strip(),
                       warning_seen={})
        settings[operation] = dict(modules=context["modules"], module_type=ModuleType,
                                   sequence_array_warnings=warning)
    context.update(host_depth_warning_contexts=settings,
                   host_depth_builtins=builtins,
                   host_depth_primitive_types=(type(None), bool, int, float, complex, str, bytes, bytearray),
                   host_depth_builtin_types={name: getattr(builtins, name) for name in  # glue: retain fixed builtin identities at package connection
                                             ("dict", "int", "list", "object", "str", "tuple", "type")})
