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
    line = tracked.co_firstlineno
    # Python 3.10 has no Instruction.positions and no BINARY_OP: an
    # instruction's line is the last starts_line seen, and the two operators
    # are BINARY_ADD / BINARY_SUBTRACT. 3.11+ uses positions and BINARY_OP.
    binary = {"BINARY_ADD": "add", "BINARY_SUBTRACT": "subtract"}
    for instruction in dis.get_instructions(tracked):  # glue: exact old arithmetic/comparison source locations
        number = getattr(instruction, "line_number", None)  # 3.13+: starts_line is a bool
        if number is not None:
            line = number
        elif type(instruction.starts_line) is int:
            line = instruction.starts_line
        positions = getattr(instruction, "positions", None)
        here = positions.lineno if positions is not None and positions.lineno is not None else line
        if instruction.opname == "BINARY_OP" and instruction.argrepr in ("+", "-"):
            lines["subtract" if instruction.argrepr == "-" else "add"] = here
        elif instruction.opname in binary:
            lines[binary[instruction.opname]] = here
        elif instruction.opname == "COMPARE_OP" and ">" in str(instruction.argval) and "=" not in str(instruction.argval):
            lines["greater"] = here
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
