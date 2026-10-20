#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Refcount oracle for a standalone Mojo stdlib extension; no mojolearn import."""
import gc
import json
import platform
import sys
import weakref

import _mojo_python_ownership as native


class Owner:
    pass


def probe(name):
    value = [] if name == "builtin_list" else Owner()
    owner = [] if name == "append_value" else Owner()
    before = sys.getrefcount(value)
    receiver_before = sys.getrefcount(owner)
    expected_call_delta = 0
    if name == "no_op":
        native.no_op(value)
    elif name == "assign_attribute":
        native.assign_attribute(owner, value)
        expected_call_delta = 1
    elif name == "read_attribute":
        owner.value = value
        result = native.read_attribute(owner)
        assert result is value
        expected_call_delta = 2
    elif name == "builtin_type":
        result = native.builtin_type(value)
        assert result is Owner
    elif name == "builtin_list":
        result = native.builtin_list(value)
        assert result == [] and result is not value
    elif name == "append_value":
        native.append_value(owner, value)
        assert owner[0] is value
        expected_call_delta = 1
    else:
        raise AssertionError(name)
    after_call = sys.getrefcount(value)
    receiver_after = sys.getrefcount(owner)
    if name in ("assign_attribute", "read_attribute"):
        del owner.value
    elif name == "append_value":
        owner.clear()
    if name in ("read_attribute", "builtin_type", "builtin_list"):
        del result
    gc.collect()
    after_release = sys.getrefcount(value)
    value_ref = None if name == "builtin_list" else weakref.ref(value)
    del value
    gc.collect()
    collected = None if value_ref is None else value_ref() is None
    passed = (after_call == before + expected_call_delta
              and after_release == before and receiver_after == receiver_before
              and collected is not False)
    return dict(name=name, before=before, after_call=after_call,
                expected_call=before + expected_call_delta, after_release=after_release,
                expected_release=before, receiver_before=receiver_before,
                receiver_after=receiver_after, value_collected=collected, passed=passed)


cases = [probe(name) for name in ("no_op", "assign_attribute", "read_attribute",
                                  "builtin_type", "builtin_list", "append_value")]
report = dict(python=sys.version, platform=platform.platform(), extension=native.__file__,
              cases=cases, passed=all(case["passed"] for case in cases))
print(json.dumps(report, indent=2))
raise SystemExit(0 if report["passed"] else 1)
