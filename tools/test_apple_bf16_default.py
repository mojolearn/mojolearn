"""Static default/gate policy checks; native compilation remains a root check."""
import ast
import re
from pathlib import Path

import pytest


SOURCE = Path(__file__).resolve().parents[1] / "gemm/afn_apple_fast.mojo"


def expression(name):
    lines = SOURCE.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("comptime " + name + " = "))
    value = lines[start].split(" = ", 1)[1]
    index = start + 1
    while value.count("(") > value.count(")"):
        value += " " + lines[index].strip()
        index += 1
    value = re.sub(r'get_defined_int\[([^]]+)\]\(\)', r'get_defined_int(\1)', value)
    value = re.sub(r'is_defined\[([^]]+)\]\(\)', r'is_defined(\1)', value)
    return compile(ast.parse(value, mode="eval"), str(SOURCE), "eval")


@pytest.mark.parametrize("vendor", ["apple", "nvidia", "amd", "cpu"])
@pytest.mark.parametrize("fast", [False, True])
def test_bf16_default_respects_actual_source_gate(vendor, fast):
    # The explicit BF16=0 opt-out (-D MOJOLEARN_AFN_GEMM_BF16_MMA=0) and the aggregate
    # AFN_GEMM_ALL / AFN_GEMM2 experiments were deleted 2026-11-15 (docs/TOMBSTONES.md):
    # BF16 is on exactly on a FAST Apple GPU build.
    defines = {"MOJOLEARN_COLUMN_CPU": 1} if vendor == "cpu" else {}
    namespace = dict(GLOBAL_NUMERIC_MODE=int(not fast), NUMERIC_FAST=0,
                     TARGET_COLUMN=vendor, COLUMN_APPLE="apple",
                     TARGET_HAS_APPLE_GPU=vendor in ("apple", "cpu"),
                     is_defined=lambda name: name in defines,
                     get_defined_int=lambda name, default: defines.get(name, default))
    namespace["AFN_GEMM_APPLE"] = eval(expression("AFN_GEMM_APPLE"), {"__builtins__": {}}, namespace)
    enabled = eval(expression("AFN_GEMM_BF16_MMA"), {"__builtins__": {}}, namespace)
    assert enabled is (vendor == "apple" and fast)
