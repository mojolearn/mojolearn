"""Verification-only tracing of a complete public operation's Python boundary.

This is a correctness probe, never a timer or a product execution helper. It
observes Python callbacks made by native code as well as ordinary Python calls.
One entry into the selected public wrapper is permitted. Every further Python
frame, including descriptors, comprehensions and worker bodies, is reported.
Native calls into NumPy/SciPy/scikit-learn are separately rejected. A clean
trace proves only the exercised call, not every possible route through an API.
"""
import dis
import gc
import sys
import threading
from collections import Counter


_FORBIDDEN_NATIVE_MODULES = ("numpy", "scipy", "sklearn")


class NativeRuntimeAudit:
    def __init__(self):
        self.result = None
        self.exception = None
        self.frames = Counter()
        self.forbidden_native = Counter()
        self.wrapper_entries = 0
        self.unobserved_threads = []
        self.wrapper_work = []

    @property
    def passed(self):
        return (self.exception is None and not self.frames and
                not self.forbidden_native and not self.unobserved_threads and
                not self.wrapper_work)

    def report(self):
        return {
            "schema": "mojolearn.native-runtime-audit.v1",
            "passed": self.passed,
            "scope": "One exercised whole operation; tracing is not timing.",
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
                f"wrapper work={self.wrapper_work}; "
                f"unobserved threads={self.unobserved_threads}; "
                f"exception={report['exception']}"
            ) from self.exception
        return self.result


def audit_native_call(operation, *args, **kwargs):
    """Exercise a call and return its value, failure and complete boundary trace.

    Call ``audit.assert_native()`` to require success and retrieve the result.
    Existing profiler hooks are restored even when the operation raises. Run
    probes in an isolated process: pre-existing other threads make the result
    incomplete, since their Python profiles cannot be installed portably.
    """
    # Drain unreachable testing-controller objects before tracing. Otherwise
    # an allocation inside a native call can close an unrelated pytest parser
    # generator and report that old finalizer as product orchestration. Keep
    # collection enabled inside the operation: its own callbacks still count.
    gc.collect()
    audit = NativeRuntimeAudit()
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
                audit.frames[(code.co_filename, code.co_firstlineno,
                              code.co_qualname)] += 1
        elif event == "c_call":
            module = getattr(arg, "__module__", None)
            if not module:
                module = type(getattr(arg, "__self__", None)).__module__
            if any(module == m or module.startswith(m + ".")
                   for m in _FORBIDDEN_NATIVE_MODULES):
                audit.forbidden_native[module + "." +
                                       getattr(arg, "__qualname__", str(arg))] += 1

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
