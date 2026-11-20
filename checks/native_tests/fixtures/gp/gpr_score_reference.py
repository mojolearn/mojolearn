"""Save the original GPR score workload and an independent NumPy GP oracle.

The original case uses one ordered RNG stream: y's noise follows X's samples.
Independent artifact writes use the host pool. This imports no Mojolearn code
and does not execute a product test, benchmark, compiler or identity check.
"""
from concurrent.futures import ThreadPoolExecutor
import ast
import hashlib
import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
SOURCE = ROOT / "python/mojolearn/tests/test_estimator_score_p2.py"
DESTINATION = Path(__file__).resolve().parent


def write_blob(item):
    name, array = item
    data = array.tobytes(order="C")
    (DESTINATION / name).write_bytes(data)
    return {"name": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def main():
    source = SOURCE.read_text()
    tree = ast.parse(source)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_reg_data")
    scope = {"np": np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), scope)
    x, y = scope["_reg_data"]()
    assert x.shape == (60, 4) and y.shape == (60,)
    # Default public kernel is Constant(1)*RBF(1), with no optimizer or
    # normalization. NumPy's dense solve is independent of the Mojo provider.
    xd, yd = x.astype(np.float64), y.astype(np.float64)
    delta = xd[:, None, :] - xd[None, :, :]
    kernel = np.exp(-0.5 * np.sum(delta * delta, axis=2))
    system = kernel + np.eye(60, dtype=np.float64) * (2.0 ** -20)
    prediction = kernel @ np.linalg.solve(system, yd)
    numerator = np.sum((yd - prediction) ** 2)
    denominator = np.sum((yd - yd.mean()) ** 2)
    reference = np.asarray([1.0 - numerator / denominator, numerator, denominator], dtype="<f8")
    jobs = [("gpr-score60-source.f32", x.astype("<f4")),
            ("gpr-score60-target.f32", y.astype("<f4")),
            ("gpr-score60-reference.f64", reference)]
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as pool:
        records = list(pool.map(write_blob, jobs))
    manifest = {"status": "independent fixture preparation only; no product execution",
                "source": str(SOURCE.relative_to(ROOT)), "function": "_reg_data",
                "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
                "function_ast_sha256": hashlib.sha256(ast.dump(function, include_attributes=False).encode()).hexdigest(),
                "rows": 60, "features": 4, "data_seed": 1,
                "kernel": "ConstantKernel(1.0, fixed) * RBF(1.0, fixed)",
                "alpha": 2.0 ** -20, "optimizer": None, "normalize_y": False,
                "original_score_abs_tolerance": 1e-3,
                "oracle": "NumPy float64 Gaussian covariance and dense linear solve, then the original float64 R2 formula",
                "numpy": np.__version__, "artifact_workers": os.cpu_count(), "files": records}
    (DESTINATION / "gpr-score-reference.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": manifest["status"], "files": len(records),
                      "bytes": sum(record["bytes"] for record in records), "rows": 60,
                      "features": 4, "reference_score": float(reference[0])}))


if __name__ == "__main__":
    main()
