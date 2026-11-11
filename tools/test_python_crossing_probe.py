"""Observer controls; no product import, GPU, or compiler is needed."""
import sys
import threading

import pytest

from python_crossing_probe import PythonCalls, visibility_control


def test_nested_python_callback_is_visible_and_report_never_qualifies_zero():
    def callback():
        return 7

    observer = PythonCalls()
    assert observer.run(callback) == 7
    report = observer.report()
    assert any(row["name"].endswith(".<locals>.callback") and row["count"] == 1
               for row in report["calls"])
    assert report["python_free_qualification"] == "not established by profiling"
    assert PythonCalls().report()["python_free_qualification"] == "not established by profiling"


def test_profile_hooks_are_restored_after_exception():
    old, old_thread = sys.getprofile(), threading.getprofile()

    def failing_callback():
        raise ValueError("control")

    with pytest.raises(ValueError, match="control"):
        PythonCalls().run(failing_callback)
    assert sys.getprofile() is old
    assert threading.getprofile() is old_thread


def test_known_c_api_call_is_reported_separately_from_observed_calls():
    control = visibility_control()
    assert control["known_invocations_per_route"] == 1
    assert control["python_to_builtin_observed"] == 1
    # Visibility differs with Python implementations. The report must retain
    # the blind spot rather than infer zero invocations from zero events.
    assert control["c_api_blind_spot_demonstrated"] == (control["c_api_to_builtin_observed"] != 1)
    assert control["qualification"] == "visibility control only; never an estimator pass"
