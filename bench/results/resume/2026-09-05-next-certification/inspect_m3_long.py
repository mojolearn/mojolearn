"""Read retained intermediates against both existing references; no GPU work."""
import os
for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[name] = "1"
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    actual = args.case / "diagnostic-actual"
    oracle = args.case / "diagnostic-oracle"
    manifest = json.loads((oracle / "manifest.json").read_text())
    dump = json.loads((actual / "dump_manifest.json").read_text())
    records = []
    for name in dump["tensors"]:
        if name not in manifest["gradients"]:
            continue
        entry = manifest["gradients"][name]
        raw = (actual / entry["file"].replace(".f64", ".f32")).read_bytes()
        values = np.frombuffer(raw, dtype="<f4").astype(np.float64)
        assert values.size == np.prod(entry["shape"]) and np.isfinite(values).all()
        row = {"tensor": name, "public_leaf": name in dump["public_prefill_leaves"],
               "shape": entry["shape"], "native_sha256": hashlib.sha256(raw).hexdigest()}
        for label, file_key, sha_key, dtype in (
            ("float64", "file", "sha256", "<f8"),
            ("pytorch_float32", "ref32_file", "ref32_sha256", "<f4"),
        ):
            if file_key not in entry:
                continue
            reference_raw = (oracle / entry[file_key]).read_bytes()
            assert hashlib.sha256(reference_raw).hexdigest() == entry[sha_key]
            reference = np.frombuffer(reference_raw, dtype=dtype).astype(np.float64)
            assert values.size == reference.size and np.isfinite(reference).all()
            delta = np.abs(values - reference)
            bad = np.flatnonzero(delta > 1e-6 + 1e-5 * np.abs(reference))
            row[label] = {"bad_cells": int(bad.size), "max_abs": float(delta.max()),
                          "examples": [{"flat_index": int(i), "actual": float(values[i]),
                                        "reference": float(reference[i])} for i in bad[:3]]}
        records.append(row)
    result = {"scope": "Diagnostic comparison only; does not change the normative certificate",
              "case": manifest["case"], "rtol": 1e-5, "atol": 1e-6, "tensors": records}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for row in records:
        if any(row.get(k, {}).get("bad_cells", 0) for k in ("float64", "pytorch_float32")):
            print(row["tensor"], "f64_bad=", row.get("float64", {}).get("bad_cells"),
                  "ref32_bad=", row.get("pytorch_float32", {}).get("bad_cells"))


if __name__ == "__main__":
    main()
