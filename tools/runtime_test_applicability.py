"""Exact runtime applicability additions; never admit an arbitrary pytest skip.

CPU HostOptKernels.io returns zero for the resident-parameter query. Python
defined buffer exporters use the Python 3.12 protocol and cannot be tested by
CPython 3.10/3.11. GPU resident tests and newer-interpreter exporter tests remain
required. Original worker receipts/JUnit are never changed by this policy.
"""
import copy
import re

SCHEMA = "mojolearn.runtime-applicability.v1"
CPU = {
    "test_samba_step_native.py": {
        "test_native_step_preserves_device_parameter_owner[None]": "native optimizer device-parameter entry is unavailable",
        "test_native_step_preserves_device_parameter_owner[0.3]": "native optimizer device-parameter entry is unavailable",
    },
    "test_training_scalar_buffer_native.py": {
        "test_device_loss_scalar_conversion_and_failure_remain_native[label_smoothing-g-0.125-swapped]": "training device pool requires GPU",
        "test_device_loss_scalar_conversion_and_failure_remain_native[ignore_index-c8-(-100+1j)-scalar]": "training device pool requires GPU",
        "test_device_loss_scalar_conversion_and_failure_remain_native[num_items-g-3.75-swapped]": "training device pool requires GPU",
        "test_device_loss_scalar_conversion_and_failure_remain_native[num_items-g-nan-scalar]": "training device pool requires GPU",
    },
    "test_cache_lifecycle_native.py": {
        'test_cache_live_matches_real_store_count_after_consumed_allocations[0-x_prep]': 'requires an actual native device store',
        'test_cache_live_matches_real_store_count_after_consumed_allocations[0-x_metrics]': 'requires an actual native device store',
        'test_cache_live_matches_real_store_count_after_consumed_allocations[1-x_prep]': 'requires an actual native device store',
        'test_cache_live_matches_real_store_count_after_consumed_allocations[1-x_metrics]': 'requires an actual native device store',
        'test_cache_live_matches_real_store_count_after_consumed_allocations[2-x_prep]': 'requires an actual native device store',
        'test_cache_live_matches_real_store_count_after_consumed_allocations[2-x_metrics]': 'requires an actual native device store',
        'test_cache_live_uses_captured_native_entry_and_ignores_mutable_int[x_prep]': 'requires an actual native device store',
        'test_cache_live_uses_captured_native_entry_and_ignores_mutable_int[x_metrics]': 'requires an actual native device store',
    },
}
PRE_PY312 = {
    "test_samba_native_admission.py": {
        "test_python_buffer_exporter_is_refused_before_buffer_callback": "Python-defined buffer exporters require Python 3.12",
    },
}


def pre_py312(interpreter):
    if not isinstance(interpreter, dict) or interpreter.get("implementation") != "cpython":
        return False
    version = re.match(r"^(\d+)\.(\d+)\.\d+(?:\s|$)", interpreter.get("version", ""))
    return bool(version and tuple(map(int, version.groups())) in ((3, 10), (3, 11)))


def supplement(scope, vendor, interpreter):
    """Derive a new judgment; only the fifteen exact named limitations may move."""
    result = copy.deepcopy(scope)
    accepted = []
    for filename, row in result["scopes"].items():
        allowed = dict(CPU.get(filename, {})) if vendor == "cpu" else {}
        if pre_py312(interpreter):
            allowed.update(PRE_PY312.get(filename, {}))
        remaining = []
        for case in row["unexpected_skips"]:
            if case["name"] in allowed and case["skip_reason"] == allowed[case["name"]]:
                row["not_applicable"].append(case)
                accepted.append(dict(file=filename, node=case["name"], reason=case["skip_reason"],
                                     basis="CPU has no native device store" if filename == "test_cache_lifecycle_native.py"
                                     else "CPU has no training device pool" if filename == "test_training_scalar_buffer_native.py"
                                     else "CPU has no resident parameter entry" if case["name"] in CPU.get(filename, {})
                                     else "recorded CPython version predates Python-defined buffer exporters"))
            else:
                remaining.append(case)
        row["unexpected_skips"] = remaining
        row["status"] = ("failed-or-incomplete" if not row["total"] or row["failures"] or remaining
                         else "passed" if row["passed"] else "not-applicable")
    if accepted:
        result["not_applicable_count"] = sum(len(row["not_applicable"]) for row in result["scopes"].values())
        result["complete"] = (bool(result["scopes"]) and not result["unmatched_cases"]
                              and all(row["status"] != "failed-or-incomplete" for row in result["scopes"].values()))
        if "exact_nodes" in result:
            result["complete"] = result["complete"] and result["exact_nodes"]["complete"]
        result["applicability_rederivations"] = dict(schema=SCHEMA, cases=accepted)
    return result
