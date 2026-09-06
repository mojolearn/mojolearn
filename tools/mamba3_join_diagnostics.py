#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Trace retained Mamba3 joins without running a model or changing its gate.

Exact arithmetic on captured operands distinguishes a bad join from error
already present in its inputs. This diagnostic is NOT a correctness
certificate: operand semantics and the unchanged whole-gradient gate still
matter, even when AMD and NVIDIA bytes match.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


# Left-associated, separate float32 additions, as in mamba3_backward.mojo.
# Single-operand entries are the scale->gamma/beta copies.
JOINS = {
    "partial.scale.gamma": ("partial.s15.scale",),
    "partial.scale.beta": ("partial.s15.scale",),
    "partial.gamma.total": ("partial.qkdot.gamma", "partial.scale.gamma"),
    "partial.dt.available_total": ("partial.dt.current_total", "partial.angle.dt"),
    "partial.dt.with_seg": ("partial.dt.available_total", "partial.dt.from_seg"),
    "partial.join.rot.q": ("partial.s16.rot.q", "partial.s17.readout.rot.q"),
    "partial.join.kscale": ("partial.s16.kscale", "partial.s17.recur.kscale"),
    "partial.join.value": ("partial.value.total", "partial.s17.recur.value"),
    "partial.join.dacs": ("partial.s17.readout.dacs", "partial.s17.recur.dacs"),
    "partial.join.adt.total": ("partial.seg.adt", "partial.join.adt.from_dacs"),
    "partial.join.gamma.total": ("partial.qkdot.gamma", "partial.join.s15.scale"),
    "partial.join.dt.available_total": (
        "partial.join.dt.current_total", "partial.join.dt.from_adt", "partial.join.angle.dt",
    ),
    "partial.value.total": ("partial.in_proj.x.from_skip", "partial.s16.value"),
    "partial.B_biased.total": ("partial.qkdot.B_biased", "partial.rotary.B_biased"),
    "partial.C_biased.total": ("partial.qkdot.C_biased", "partial.rotary.C_biased"),
    "partial.join.B_biased.total": ("partial.qkdot.B_biased", "partial.join.rotary.B_biased"),
    "partial.join.C_biased.total": ("partial.qkdot.C_biased", "partial.join.rotary.C_biased"),
}


def ftz(values):
    """IDENTICAL flushes float32 subnormals, preserving the sign of zero."""
    bits = np.asarray(values, dtype="<f4").copy().view(np.uint32)
    subnormal = ((bits & 0x7f800000) == 0) & ((bits & 0x007fffff) != 0)
    bits[subnormal] &= 0x80000000
    return bits.view(np.float32)


def pinned_join(operands):
    result = ftz(operands[0]).reshape(-1)
    for operand in operands[1:]:
        other = ftz(operand).reshape(-1)
        if result.shape != other.shape:
            raise ValueError("join operand cell count mismatch")
        result = ftz(result + other)
    return result


def errors(actual, expected):
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    bad = np.flatnonzero(delta > 1e-6 + 1e-5 * np.abs(expected.astype(np.float64)))
    return {"max_abs": float(delta.max()), "bad_cells": int(bad.size),
            "bad_indices": bad.tolist()}


def load_case(case_dir):
    actual_dir = case_dir / "failed-actual"
    oracle_dir = case_dir / "failed-oracle"
    dump = json.loads((actual_dir / "dump_manifest.json").read_text())
    oracle = json.loads((oracle_dir / "manifest.json").read_text())
    native = json.loads((case_dir / "native" / "manifest.json").read_text())
    if (dump.get("schema") != "mojolearn.mamba.gradient-dump.v1"
            or oracle.get("schema") != "mojolearn.mamba.gradient-oracle.v1"
            or dump.get("family") != "mamba3" or oracle.get("family") != "mamba3"
            or dump.get("case") != oracle.get("case")
            or dump.get("objective") != "signed_dyadic_weight_v1"
            or oracle.get("objective") != "sum(flat(block_output) * signed_dyadic_weight_v1)"):
        raise ValueError("incompatible Mamba3 capture provenance")
    if (native.get("schema") != "mojolearn.mamba.backward-bytes.v1"
            or native.get("mode") != "IDENTICAL"
            or native.get("family") != dump["family"]
            or native.get("case") != dump["case"]
            or native.get("objective") != dump["objective"]
            or not native.get("source_sha256")):
        raise ValueError("missing matching IDENTICAL native capture provenance")
    names = dump["tensors"]
    if not names or len(names) != len(set(names)):
        raise ValueError("empty or duplicate tensor inventory")
    values, references, rows = {}, {}, {}
    for name in names:
        entry = oracle["gradients"][name]
        raw = (actual_dir / entry["file"].replace(".f64", ".f32")).read_bytes()
        actual = np.frombuffer(raw, dtype="<f4")
        if actual.size != np.prod(entry["shape"]) or not np.isfinite(actual).all():
            raise ValueError(f"invalid native tensor: {name}")
        values[name] = actual
        rows[name] = {"sha256": hashlib.sha256(raw).hexdigest(), "shape": entry["shape"]}
        references[name] = {}
        for label, file_key, sha_key, dtype in (
            ("float64", "file", "sha256", "<f8"),
            ("float32", "ref32_file", "ref32_sha256", "<f4"),
        ):
            reference_raw = (oracle_dir / entry[file_key]).read_bytes()
            if hashlib.sha256(reference_raw).hexdigest() != entry[sha_key]:
                raise ValueError(f"reference digest mismatch: {name}/{label}")
            expected = np.frombuffer(reference_raw, dtype=dtype)
            if expected.size != actual.size or not np.isfinite(expected).all():
                raise ValueError(f"invalid reference: {name}/{label}")
            references[name][label] = expected
            rows[name][label] = errors(actual, expected)
    for name, entry in native["tensors"].items():
        if name not in rows or entry["sha256"] != rows[name]["sha256"]:
            raise ValueError(f"diagnostic/public capture digest mismatch: {name}")
    joins = {}
    for name, inputs in JOINS.items():
        result = pinned_join([values[key] for key in inputs])
        actual = values[name]
        if result.size != actual.size:
            raise ValueError(f"join output cell count mismatch: {name}")
        joins[name] = {
            "operands": list(inputs),
            "bitwise": bool(np.array_equal(result.view(np.uint32), actual.view(np.uint32))),
            "operand_semantics": {key: {kind: rows[key][kind] for kind in ("float64", "float32")}
                                  for key in inputs},
            # Sum the captured operands in float64 to expose cancellation;
            # this is only an attribution aid, not a replacement reference.
            "native_vs_exact_operand_sum": errors(
                actual, sum(values[key].astype(np.float64) for key in inputs)),
        }
    report = {"path": str(case_dir), "case": dump["case"],
              "source_sha256": native["source_sha256"], "tensors": rows, "joins": joins}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    left, right = load_case(args.left), load_case(args.right)
    if (left["case"] != right["case"] or left["source_sha256"] != right["source_sha256"]
            or set(left["tensors"]) != set(right["tensors"])):
        raise ValueError("source/case/tensor inventory mismatch")
    matches = {name: left["tensors"][name]["sha256"] == right["tensors"][name]["sha256"]
               and left["tensors"][name]["shape"] == right["tensors"][name]["shape"]
               for name in left["tensors"]}
    result = {
        "scope": "Retained-data diagnostic only; does not issue or change a certificate",
        "rtol": 1e-5, "atol": 1e-6,
        "cross_device_tensor_matches": matches, "left": left, "right": right,
    }
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Native tensor byte matches: {sum(matches.values())}/{len(matches)}")
    for label, report in (("left", left), ("right", right)):
        exact = sum(row["bitwise"] for row in report["joins"].values())
        failed = [name for name, row in report["tensors"].items() if row["float32"]["bad_cells"]]
        explained = [name for name in failed if name in report["joins"] and report["joins"][name]["bitwise"]]
        print(f"{label}: exact joins {exact}/{len(JOINS)}; unchanged direct failures {len(failed)}")
        print(f"{label}: failing outputs reproduced from native inputs: {', '.join(explained)}")
    if not all(matches.values()) or not all(
            row["bitwise"] for report in (left, right) for row in report["joins"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
