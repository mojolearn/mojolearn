#!/usr/bin/env python3
"""External CPU C-ABI smoke and refusal checks; no framework dependency."""
import argparse
import ctypes as c
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("library", type=Path)
    parser.add_argument("--json", required=True, type=Path)
    args = parser.parse_args()
    lib = c.CDLL(str(args.library.resolve()))
    lib.mojolearn_adapter_abi.restype = c.c_int64
    assert lib.mojolearn_adapter_abi() == 1
    call = lib.mojolearn_adapter_operation
    call.argtypes = [c.c_int64] * 11
    call.restype = c.c_int64
    x = (c.c_float * 6)(1, -2, .5, -.25, 2, -1)
    y = (c.c_float * 6)(.5, 1, 2, 1, .5, -.25)
    dst = (c.c_float * 6)()
    desc = (c.c_int64 * 22)(c.addressof(x), 0, 2, 2, 3, 1, 1, 1, 1, 1, 1,
                              c.addressof(y), 0, 2, 2, 3, 1, 1, 1, 1, 1, 1)
    out = (c.c_int64 * 11)(c.addressof(dst), 0, 2, 2, 3, 1, 1, 1, 1, 1, 1)
    error = c.create_string_buffer(2048)
    def run(code=2, count=2):
        return call(-1, code, c.addressof(desc), count, c.addressof(out), 0, 0, 0, 0,
                    c.addressof(error), len(error))
    assert run() == 0, error.value
    assert list(dst) == [1.5, -1., 2.5, .75, 2.5, -1.25], list(dst)
    checks = ["cpu_add"]
    for _ in range(20):
        assert run() == 0, error.value
    checks.append("repeated_calls")
    out[3], out[4] = 3, 2
    assert run() != 0 and b"output shape" in error.value, error.value
    out[3], out[4] = 2, 3
    checks.append("output_shape_refusal")
    assert run(47) != 0 and b"unknown operation" in error.value, error.value
    checks.append("unknown_code_refusal")
    assert run(32, 1) != 0 and b"tensor indices" in error.value, error.value
    checks.append("host_pointer_option_refusal")
    assert run() == 0 and error.value == b"", error.value
    checks.append("restored_after_error")
    args.json.write_text(json.dumps({"complete": True, "checks": checks, "device": "CPU",
                                    "scope": "native adapter ABI only"}, indent=2) + "\n")
    print(f"PASS {len(checks)} native adapter ABI checks")


if __name__ == "__main__":
    main()
