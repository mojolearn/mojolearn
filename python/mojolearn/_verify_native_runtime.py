"""Verification-only profiling across a whole public operation.

This is a correctness probe, never a timer or a product execution helper. It
observes profile-visible Python callbacks made by native code and ordinary
Python calls. One entry into the selected public wrapper is permitted. Every
further profiled Python frame, including descriptors, comprehensions and worker
bodies, is reported. Native calls into NumPy/SciPy/scikit-learn are separately
rejected. CPython suppresses profiling of audit hooks unless the hook enables
tracing. A clean trace alone therefore does not prove that even the exercised
operation was callback-free. Use independent callback counters and controls
for those paths; do not change user hooks to obtain a passing result.
Native calls with unresolved origins also prevent qualification, including in
the existing forbidden-call signal used by refusal tests.
"""
import dis
import gc
import sys
import threading
from collections import Counter
from contextlib import contextmanager
from types import ModuleType

#: The audited call stands in for a user's: the public methods answer this
#: module as they answer code outside the package (`_mode._caller_outside`).
__mojolearn_user_boundary__ = True


_FORBIDDEN_NATIVE_MODULES = ("numpy", "scipy", "sklearn")


def loaded_binding(name, mode="identical", report=None):
    """The `mojolearn.runtime_report()` row of binding `name` (for example
    `_mojolearn_x_sequence`) loaded in tier `mode`, or None when no loaded
    binding of that tier is in the report. A row counts when the binding's
    own compiled read-back says `mode`; a binding without that read-back
    counts as the tier it was loaded for (its tier set, or the process tier
    for the selected binding). Call it AFTER the work that loads the binding:
    the report never loads one."""
    if report is None:
        from . import runtime_report
        report = runtime_report()
    keys = [(f"mojolearn._sets.{mode}.{name}", mode), (f"mojolearn.{name}", report["numeric_mode"])]
    if report.get("cpu_only"):
        # A CPU-only install serves `mojolearn.<name>` through a facade the
        # report skips (not an exact module); the binding that ran is its
        # host extension, `mojolearn._host.<basename>`, which carries its own
        # vendor and mode read-backs (host bindings build identical only).
        from . import host_surface
        basename = host_surface.routed_modules().get(name) or host_surface.inference_routes().get(name)
        if basename:
            keys.append((f"mojolearn._host.{basename}", "identical"))
    for key, tier in keys:
        row = report["bindings"].get(key)
        if row is None or not row["loaded"] or row["path"] is None:
            continue
        compiled = row.get("compiled_mode")
        if compiled == mode or (compiled is None and tier == mode):
            return row
    return None


def _ordinary_field(fields, wanted):
    # The profiler must not execute metadata equality or descriptor hooks.
    value = None
    for name, item in fields.items():
        if type(name) is not str:
            return None
        if name == wanted:
            value = item
    return value


def _module_name(value):
    if type(value) is str:
        return value or None
    if type(value) is ModuleType:
        # Some extension functions (including psutil on macOS) retain their
        # actual module as __module__, rather than its name string.
        fields = ModuleType.__dict__["__dict__"].__get__(value)
        name = _ordinary_field(fields, "__name__")
        return name if type(name) is str and name else None
    return None


def _native_call_origin(function):
    module = getattr(function, "__module__", None)
    if module is None or (type(module) is str and not module):
        owner = type(getattr(function, "__self__", None))
        fields = type.__dict__["__dict__"].__get__(owner)
        if any(type(name) is not str for name in fields):
            return None
        # Built-in owner types compute __module__ rather than storing it in
        # their dictionaries. Call the retained native type descriptor, not
        # a metaclass property or custom __getattribute__ implementation.
        try:
            module = type.__dict__["__module__"].__get__(owner)
        except AttributeError:
            # A heap type registered by a compiled binding under an undotted
            # name (Mojo `add_type`, e.g. the schedule's `_LrScheduleBlock`,
            # whose bound `lr_at` / `bits_at` are the public methods) has no
            # __module__. Its origin is the module that holds the type object
            # itself; with none, it stays unresolved (fail closed).
            module = _registering_extension(owner)
    return _module_name(module)


#: id(type) -> (type, name or None): the extension module holding each heap
#: type without __module__ (the type is retained so its id cannot be reused).
_REGISTERED_TYPE_MODULES = {}


def _registering_extension(owner):
    entry = _REGISTERED_TYPE_MODULES.get(id(owner))
    if entry is not None and entry[0] is owner:
        return entry[1]
    # Every loaded module that holds the type object itself. A forbidden
    # module among them wins, so a forbidden type re-exported elsewhere is
    # still reported; otherwise the first holder (the registering binding).
    holders = []
    read_fields = ModuleType.__dict__["__dict__"].__get__
    for module in list(sys.modules.values()):
        if type(module) is not ModuleType:
            continue
        fields = read_fields(module)
        if any(item is owner for item in list(fields.values())):
            name = _ordinary_field(fields, "__name__")
            if type(name) is str and name:
                holders.append(name)
    found = next((n for n in holders if any(n == m or n.startswith(m + ".") for m in _FORBIDDEN_NATIVE_MODULES)),
                 holders[0] if holders else None)
    if found is not None:
        _REGISTERED_TYPE_MODULES[id(owner)] = (owner, found)
    return found


class NativeRuntimeAudit:
    def __init__(self, approved_callbacks=None):
        self.result = None
        self.exception = None
        self.frames = Counter()
        self.approved_callbacks = approved_callbacks or {}
        self.approved_frames = Counter()
        self.fallback_calls = Counter()
        self.forbidden_native = Counter()
        self.unresolved_native = Counter()
        self.wrapper_entries = 0
        self.unobserved_threads = []
        self.wrapper_work = []

    @property
    def passed(self):
        return (self.exception is None and not self.frames and
                not self.forbidden_native and not self.unresolved_native and not self.unobserved_threads and
                not self.wrapper_work)

    @property
    def compatibility_qualified(self):
        return (self.exception is None and self.frames == self.approved_frames
                and not self.forbidden_native and not self.unresolved_native
                and not self.unobserved_threads and not self.wrapper_work)

    def report(self):
        return {
            "schema": "mojolearn.native-runtime-audit.v1",
            "passed": self.passed,
            "compatibility_qualified": self.compatibility_qualified,
            "approved_python_fallbacks": [
                dict(record, calls=self.fallback_calls[record['id']])
                for record in self.approved_callbacks.values()
                if self.fallback_calls[record['id']]
            ],
            "scope": "Profile-visible work in one whole operation; tracing is not timing.",
            "claim": "The passed field covers only profiler-visible work and wrapper-bytecode checks.",
            "observer_limitations": [
                "CPython audit hooks can execute Python without profile events; "
                "callback-sensitive paths require independent counters and controls."
            ],
            "wrapper_entries": self.wrapper_entries,
            "wrapper_work": self.wrapper_work,
            "python_frames": [
                dict(file=filename, line=line, function=name, calls=count)
                for (filename, line, name), count in sorted(self.frames.items())
            ],
            "forbidden_native_calls": [
                dict(function=name, calls=count)
                for name, count in sorted(self.forbidden_native.items())
            ],
            "unresolved_native_calls": [
                dict(function=name, calls=count)
                for name, count in sorted(self.unresolved_native.items())
            ],
            "unobserved_existing_threads": self.unobserved_threads,
            "exception": None if self.exception is None else {
                "type": type(self.exception).__name__, "message": str(self.exception)},
        }

    def assert_native(self):
        if not self.passed:
            report = self.report()
            details = report["python_frames"][:8]
            raise AssertionError(
                f"native runtime boundary failed: Python frames={details}; "
                f"forbidden native={report['forbidden_native_calls']}; "
                f"unresolved native={report['unresolved_native_calls']}; "
                f"wrapper work={self.wrapper_work}; "
                f"unobserved threads={self.unobserved_threads}; "
                f"exception={report['exception']}"
            ) from self.exception
        return self.result

    def assert_compatible(self):
        """Require only the explicit compatibility exceptions, retaining traces.

        ``passed`` and ``assert_native`` remain strict: an approved Python
        callback still prevents a Python-free claim.
        """
        if not self.compatibility_qualified:
            return self.assert_native()
        return self.result


def audit_native_call(operation, *args, **kwargs):
    """Exercise a call and return its value, failure and profile-visible trace.

    Call ``audit.assert_native()`` to require success and retrieve the result.
    Existing profiler hooks are restored even when the operation raises. Run
    probes in an isolated process: pre-existing other threads make the result
    incomplete, since their Python profiles cannot be installed portably.
    Audit hooks can also evade profiling; a passing trace is an observation,
    not standalone proof that the operation executed no Python callbacks.
    """
    return _audit_call(operation, args, kwargs, {})


def audited(audits, name, operation, *args, **kwargs):
    """Audit one complete public call, append its trace (with ``operation``
    set to ``name``) to the list ``audits`` and return its value; a call that
    entered Python raises (``assert_native``)."""
    audit = audit_native_call(operation, *args, **kwargs)
    entry = audit.report()
    entry["operation"] = name
    audits.append(entry)
    return audit.assert_native()


@contextmanager
def audited_context(audits, name, operation, *args, **kwargs):
    """Open a context-managed public object with ``operation`` and audit its
    construction, ``__enter__`` and ``__exit__`` as three complete public
    boundaries (``name``, ``name.enter``, ``name.exit``), each appended to
    ``audits``. The one shared controller of the distributed and partitioned
    program verifiers."""
    instance = audited(audits, name, operation, *args, **kwargs)
    try:
        assert audited(audits, name + ".enter", instance.__enter__) is instance
        yield instance
    finally:
        audited(audits, name + ".exit", instance.__exit__, None, None, None)


def audit_compatible_call(operation, *args, **kwargs):
    """Audit the public operation with its named, owner-authorized exceptions."""
    from ._runtime_fallbacks import approved_metadata_callbacks
    return _audit_call(operation, args, kwargs, approved_metadata_callbacks(operation))


def _audit_call(operation, args, kwargs, approved_callbacks):
    # Drain unreachable testing-controller objects before tracing. Otherwise
    # an allocation inside a native call can close an unrelated pytest parser
    # generator and report that old finalizer as product orchestration. Keep
    # collection enabled inside the operation: its own callbacks still count.
    gc.collect()
    audit = NativeRuntimeAudit(approved_callbacks)
    # A class call enters its public __init__ wrapper. Custom Python __new__
    # implementations and metaclass calls remain additional traced frames.
    function = operation.__init__ if type(operation) is type else getattr(operation, "__func__", operation)
    allowed = getattr(function, "__code__", None)
    if allowed is not None:
        instructions = list(dis.get_instructions(allowed))
        calls = [i for i in instructions if i.opname in
                 ("CALL", "CALL_FUNCTION", "CALL_METHOD", "CALL_FUNCTION_EX",
                  "CALL_KW")]
        if len(calls) != 1:
            audit.wrapper_work.append(f"wrapper contains {len(calls)} calls, expected one")
        # Inlined comprehensions (Python 3.12+) do not open a new frame. Inspect
        # wrapper bytecode too: argument preparation and routing must be native.
        forbidden = ("FOR_ITER", "GET_ITER", "GET_AITER", "SEND", "YIELD_VALUE",
                     "BINARY_OP", "BINARY_SUBSCR", "COMPARE_OP", "IS_OP",
                     "CONTAINS_OP", "BUILD_LIST", "BUILD_SET", "BUILD_MAP",
                     "BUILD_CONST_KEY_MAP", "LIST_APPEND", "MAP_ADD", "SET_ADD")
        audit.wrapper_work.extend(
            f"{i.opname} at bytecode offset {i.offset}" for i in instructions
            if i.opname in forbidden or "JUMP" in i.opname or
            i.opname.startswith(("BINARY_", "INPLACE_")))
    caller_thread = threading.get_ident()
    audit.unobserved_threads = [t.name for t in threading.enumerate()
                               if t.ident != caller_thread and t.is_alive()]
    old_profile = sys.getprofile()
    old_thread_profile = threading.getprofile()
    active = True

    def profile(frame, event, arg):
        if not active:
            return
        if event == "call":
            code = frame.f_code
            if (code is allowed and audit.wrapper_entries == 0 and
                    threading.get_ident() == caller_thread):
                audit.wrapper_entries += 1
            else:
                key = (code.co_filename, code.co_firstlineno, getattr(code, "co_qualname", code.co_name))  # co_qualname is 3.11+
                audit.frames[key] += 1
                if code in audit.approved_callbacks:
                    audit.approved_frames[key] += 1
                    audit.fallback_calls[audit.approved_callbacks[code]['id']] += 1
        elif event == "c_call":
            module = _native_call_origin(arg)
            name = getattr(arg, "__qualname__", None)
            if type(name) is not str:
                name = "<native callable>"
            if module is None:
                audit.unresolved_native[name] += 1
                # Existing refusal guards inspect forbidden_native directly
                # or its report list. Keep their admission fail-closed too;
                # unresolved_native retains the distinct provenance reason.
                audit.forbidden_native["<unresolved-origin>." + name] += 1
            elif any(module == m or module.startswith(m + ".")
                   for m in _FORBIDDEN_NATIVE_MODULES):
                audit.forbidden_native[module + "." + name] += 1

    # Install the future-thread hook before profiling the selected operation;
    # the audit controller's own setup/teardown must not enter the result.
    threading.setprofile(profile)
    sys.setprofile(profile)
    try:
        audit.result = operation(*args, **kwargs)
    except BaseException as exc:
        audit.exception = exc
    finally:
        active = False
        sys.setprofile(old_profile)
        threading.setprofile(old_thread_profile)
    return audit
