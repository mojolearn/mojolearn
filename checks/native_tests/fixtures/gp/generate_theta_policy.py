# SPDX-License-Identifier: Apache-2.0
"""Prepare the original GP theta inputs and independent stdlib exp oracle.

This is reference preparation, not product execution. Independent 128-word
chunks run across os.cpu_count() processes. Four small artifact writes follow
the completed arithmetic; no Mojolearn module or native provider is imported.
"""
from concurrent.futures import ProcessPoolExecutor
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import struct
import sys


def chunk(job):
    part, start, stop = job
    values = []
    for index in range(start, stop):
        theta = (index - 2048) / 32
        if part == "lower":
            value = math.nextafter(theta, -math.inf)
        elif part == "upper":
            value = math.nextafter(theta, math.inf)
        elif part == "exp":
            value = math.exp(theta)
        else:
            value = theta
        values.append(value)
    return part, start, struct.pack("<" + "d" * len(values), *values)


def main():
    directory = Path(__file__).resolve().parent
    repository = directory.parents[3]
    original = repository / "packaging/portable_math/test_host_math.py"
    source = original.read_text()
    node = next(item for item in ast.parse(source).body
                if isinstance(item, ast.FunctionDef) and item.name == "test_gp_theta_float32_policy")
    parts = ("base", "lower", "upper", "exp")
    jobs = [(part, start, min(start + 128, 4097))
            for part in parts for start in range(0, 4097, 128)]
    workers = os.cpu_count() or 1
    with ProcessPoolExecutor(max_workers=workers) as pool:
        prepared = list(pool.map(chunk, jobs))
    artifacts = {}
    for part in parts:
        blocks = sorted((start, data) for label, start, data in prepared if label == part)
        data = b"".join(data for _, data in blocks)
        if len(data) != 4097 * 8:
            raise ValueError("theta fixture extent")
        path = directory / f"theta_policy-{part}.f64"
        path.write_bytes(data)
        artifacts[path.name] = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    # Explicitly preserve the signed smallest binary64 neighbors of zero.
    lower = (directory / "theta_policy-lower.f64").read_bytes()
    upper = (directory / "theta_policy-upper.f64").read_bytes()
    if struct.unpack_from("<Q", lower, 2048 * 8)[0] != 0x8000000000000001:
        raise ValueError("negative binary64 neighbor of zero")
    if struct.unpack_from("<Q", upper, 2048 * 8)[0] != 1:
        raise ValueError("positive binary64 neighbor of zero")
    metadata = {
        "purpose": "independent reference preparation; no product imports or native execution",
        "original": "packaging/portable_math/test_host_math.py::test_gp_theta_float32_policy",
        "original_line": node.lineno,
        "original_function_ast_sha256": hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest(),
        "inputs": "i/32 for i=-2048..2048 inclusive; math.nextafter toward both infinities",
        "oracle": "Python stdlib math.exp followed by the original binary32 word comparison, tolerance2ULP",
        "ftz": {"input": [-100.0, -90.0, -88.0, 0.0], "expected": [0.0, 0.0, 0.0, 1.0]},
        "count_per_span": 4097, "chunk_size": 128, "independent_chunks": len(jobs),
        "workers": workers, "python": sys.version, "platform": platform.platform(),
        "artifacts": artifacts,
        "limits": "Oracle comes from stdlib libm, never native GP outputs or an old release hash."
    }
    (directory / "theta_policy-reference.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"status": "reference preparation complete", "artifacts": len(artifacts),
                      "bytes": sum(x["bytes"] for x in artifacts.values()),
                      "workers": workers, "chunks": len(jobs), "product_execution": False}))


if __name__ == "__main__":
    main()
