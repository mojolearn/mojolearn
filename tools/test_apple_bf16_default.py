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
@pytest.mark.parametrize("override", [None, 0, 1])
def test_bf16_default_and_explicit_off_respect_actual_source_gate(vendor, fast, override):
    defines = {"MOJOLEARN_COLUMN_CPU": 1} if vendor == "cpu" else {}
    if override is not None:
        defines["MOJOLEARN_AFN_GEMM_BF16_MMA"] = override
    namespace = dict(GLOBAL_NUMERIC_MODE=int(not fast), NUMERIC_FAST=0,
                     TARGET_COLUMN=vendor, COLUMN_APPLE="apple",
                     has_apple_gpu_accelerator=lambda: vendor in ("apple", "cpu"),
                     is_defined=lambda name: name in defines,
                     get_defined_int=lambda name, default: defines.get(name, default))
    namespace["AFN_GEMM_APPLE"] = eval(expression("AFN_GEMM_APPLE"), {"__builtins__": {}}, namespace)
    # Aggregate experiments must not defeat an explicit BF16=0. These values
    # affect the actual expression only if a future edit adds them back.
    for all_on, second_generation_on in [(False, False), (True, False), (False, True), (True, True)]:
        namespace.update(AFN_GEMM_ALL=all_on, AFN_GEMM2_ON=second_generation_on)
        enabled = eval(expression("AFN_GEMM_BF16_MMA"), {"__builtins__": {}}, namespace)
        assert enabled is (vendor == "apple" and fast and override != 0)


def test_explicit_off_survives_the_real_build_flag_passthrough():
    from test_build_defines import flags

    value = "MOJOLEARN_AFN_GEMM_BF16_MMA=0,MOJOLEARN_AFN_GEMM_ALL=1"
    code, stdout, stderr = flags(value)
    assert code == 0, stderr
    assert stdout == "[-D MOJOLEARN_AFN_GEMM_BF16_MMA=0 -D MOJOLEARN_AFN_GEMM_ALL=1]"
