"""Negative controls for whole-operation tracing, independent of native builds."""
import importlib.util
from pathlib import Path
import sys
import threading
import unittest


PATH = Path(__file__).resolve().parents[1] / "python/mojolearn/_verify_native_runtime.py"
SPEC = importlib.util.spec_from_file_location("native_runtime_audit_under_test", PATH)
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)


class NativeRuntimeAuditTests(unittest.TestCase):
    def test_one_wrapper_into_builtin_is_native(self):
        def public(values):
            return sum(values)
        result = AUDIT.audit_native_call(public, [2, 3])
        self.assertEqual(result.assert_native(), 5)
        self.assertEqual(result.report()["wrapper_entries"], 1)

    def test_transitive_helper_and_comprehension_are_detected(self):
        def helper(values):
            return [x * 2 for x in values]
        def public(values):
            return helper(values)
        result = AUDIT.audit_native_call(public, [2, 3])
        self.assertFalse(result.passed)
        self.assertTrue(any("helper" in row["function"]
                            for row in result.report()["python_frames"]))

    def test_python_work_inside_only_wrapper_is_detected(self):
        def arithmetic(value):
            return abs(value + 1)
        def loop(values):
            total = 0
            for value in values:
                total += value
            return abs(total)
        def comprehension(values):
            return sum([value * 2 for value in values])
        for operation, arg in ((arithmetic, 1), (loop, [1, 2]),
                               (comprehension, [1, 2])):
            result = AUDIT.audit_native_call(operation, arg)
            self.assertFalse(result.passed)
            self.assertTrue(result.wrapper_work)

    def test_native_builtin_invoking_python_callback_is_detected(self):
        def public(values):
            return sorted(values, key=lambda x: -x)
        result = AUDIT.audit_native_call(public, [2, 3])
        self.assertFalse(result.passed)
        self.assertTrue(any("lambda" in row["function"]
                            for row in result.report()["python_frames"]))

    def test_descriptor_and_recursive_wrapper_are_detected(self):
        class Value:
            @property
            def value(self):
                return 3
        def public(obj):
            return obj.value
        self.assertFalse(AUDIT.audit_native_call(public, Value()).passed)
        def recursive(n):
            return recursive(n - 1) if n else 0
        self.assertFalse(AUDIT.audit_native_call(recursive, 1).passed)

    def test_failure_is_not_hidden_and_profilers_restored(self):
        def failed():
            raise ValueError("deliberate failure")
        old = sys.getprofile(), threading.getprofile()
        result = AUDIT.audit_native_call(failed)
        self.assertEqual((sys.getprofile(), threading.getprofile()), old)
        self.assertEqual(result.report()["exception"]["type"], "ValueError")
        with self.assertRaises(AssertionError):
            result.assert_native()

    def test_worker_runtime_is_not_an_escape(self):
        def public():
            worker = threading.Thread(target=lambda: sum([1, 2]))
            worker.start()
            worker.join()
        result = AUDIT.audit_native_call(public)
        self.assertFalse(result.passed)
        self.assertTrue(any("lambda" in row["function"]
                            for row in result.report()["python_frames"]))

    def test_old_controller_cycles_are_collected_before_operation(self):
        finalized = []
        class Controller:
            def __del__(self):
                finalized.append(True)
        value = Controller()
        value.cycle = value
        del value
        self.assertEqual(AUDIT.audit_native_call(sum, [1, 2]).assert_native(), 3)
        self.assertEqual(finalized, [True])

    def test_finalizer_invoked_by_operation_remains_visible(self):
        class Value:
            def __del__(self):
                pass
        values = [Value()]
        result = AUDIT.audit_native_call(values.clear)
        self.assertFalse(result.passed)
        self.assertTrue(any("__del__" in row["function"]
                            for row in result.report()["python_frames"]))


if __name__ == "__main__":
    unittest.main()
