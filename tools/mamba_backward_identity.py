#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Retain native gradient bytes and compare named backward certificates.

Numerical oracle success and cross-device byte equality are separate gates.
This tool never rounds, normalizes, or recomputes a gradient before hashing it.
"""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys


SOURCE_PATHS = (
    "mamba/__init__.mojo", "mamba/checks", "mamba/impl",
    "mamba/corpus/gen_corpus.py", "checks/__init__.mojo",
    "checks/numerics.mojo", "checks/kernel_matrix.mojo",
    "core/__init__.mojo", "core/identity_trace.mojo",
    "gemm/__init__.mojo", "gemm/checks", "pixi.toml", "pixi.lock",
    "tools/mamba_backward_certify.sh", "tools/mamba_backward_identity.py",
    "tools/mamba_gradient_oracle.py", "tools/with_identical_mode.sh",
    "tools/with_build_lock.sh",
)
SCHEMA = "mojolearn.mamba.backward-bytes.v1"
PUBLIC_LEAVES = {
    "mamba1": ["x", "norm.weight", "in_proj.weight", "conv1d.weight",
               "conv1d.bias", "out_proj.weight", "D", "A_log",
               "dt_proj.weight", "dt_proj.bias", "x_proj.weight"],
    "mamba2": ["x", "block_norm.weight", "in_proj.weight", "conv1d.weight",
               "conv1d.bias", "dt_bias", "A_log", "D", "norm.weight",
               "out_proj.weight"],
    "mamba3": ["x", "block_norm.weight", "in_proj.weight", "dt_bias",
               "B_norm.weight", "C_norm.weight", "B_bias", "C_bias", "D",
               "out_proj.weight"],
}
CASES = {
    "mamba1": ("mamba1", "base_b2_l4_d8"),
    "mamba2": ("mamba2", "m2_base_b2_l4_d32"),
    "mamba3": ("mamba3", "m3_base_b2_l4_d32"),
    "mamba2-l257": ("mamba2", "m2_base_b1_l257_d64"),
    "mamba2-state": ("mamba2", "m2_base_b1_l257_d64"),
}
STATE_POLICY = (
    "incoming_state_before_chunk0; block_output_objective; "
    "final_state_cotangent=zero"
)


def validate_metadata(dump, reference):
    for key in ("family", "case"):
        if dump.get(key) != reference.get(key):
            raise ValueError(f"oracle/dump {key} mismatch")
    family = dump.get("family")
    if family not in PUBLIC_LEAVES:
        raise ValueError("unknown gradient family")
    if (dump.get("objective") != "signed_dyadic_weight_v1" or
            reference.get("objective") !=
            "sum(flat(block_output) * signed_dyadic_weight_v1)"):
        raise ValueError("oracle/dump objective mismatch")
    for manifest in (dump, reference):
        if manifest.get("public_prefill_leaves") != PUBLIC_LEAVES[family]:
            raise ValueError("missing or mismatched public leaf inventory")
        if manifest.get("state_boundary_leaves", []) != (
            ["initial_state"] if family == "mamba2" else []
        ):
            raise ValueError("state boundary inventory mismatch")
        if manifest.get("state_boundary_policy") != (
            STATE_POLICY if family == "mamba2" else None
        ):
            raise ValueError("state boundary policy mismatch")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_digest(root):
    files = []
    for name in SOURCE_PATHS:
        path = root / name
        if not path.exists():
            raise ValueError(f"missing source dependency: {name}")
        files.extend(path.rglob("*.mojo") if path.is_dir() else [path])
    records = [(p.relative_to(root).as_posix(), digest(p)) for p in files]
    return hashlib.sha256(
        json.dumps(sorted(records), separators=(",", ":")).encode()
    ).hexdigest()


def capture(actual, oracle, output, source_sha256):
    dump = json.loads((actual / "dump_manifest.json").read_text())
    reference = json.loads((oracle / "manifest.json").read_text())
    validate_metadata(dump, reference)
    leaves = dump.get("public_prefill_leaves", [])
    if not leaves or leaves != reference.get("public_prefill_leaves"):
        raise ValueError("missing or mismatched public leaf inventory")
    names = leaves + dump.get("state_boundary_leaves", [])
    if len(set(names)) != len(names):
        raise ValueError("duplicate native gradient leaf")
    output.mkdir(parents=True, exist_ok=False)
    tensors = {}
    for name in names:
        entry = reference["gradients"][name]
        filename = "grad." + name + ".f32"
        if Path(filename).name != filename or name not in dump["tensors"]:
            raise ValueError(f"invalid or undeclared native leaf: {name}")
        source = actual / filename
        shape = entry["shape"]
        if not shape or any(type(n) is not int or n < 1 for n in shape):
            raise ValueError(f"invalid shape: {name}")
        if source.stat().st_size != 4 * math.prod(shape):
            raise ValueError(f"wrong float32 byte count: {name}")
        shutil.copyfile(source, output / filename)
        tensors[name] = {
            "file": filename, "dtype": "<f4", "shape": shape,
            "sha256": digest(source),
        }
    manifest = {
        "schema": SCHEMA, "mode": "IDENTICAL",
        "source_sha256": source_sha256,
        "family": dump["family"], "case": dump["case"],
        "objective": dump["objective"],
        "public_prefill_leaves": leaves,
        "state_boundary_leaves": dump.get("state_boundary_leaves", []),
        "state_boundary_policy": dump.get("state_boundary_policy"),
        "tensors": tensors,
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )


def load_certificate(root):
    env = dict(line.split("=", 1) for line in
               (root / "environment.txt").read_text().splitlines() if "=" in line)
    if env.get("mode") != "IDENTICAL":
        raise ValueError(f"{root}: not an IDENTICAL certificate")
    if env.get("source_changed") or len(env.get("source_sha256", "")) != 64:
        raise ValueError(f"{root}: missing or unstable source fingerprint")
    if env.get("vendor") not in ("apple", "nvidia", "amd"):
        raise ValueError(f"{root}: missing hardware vendor")
    if not (root / "device.csv").read_text().strip():
        raise ValueError(f"{root}: missing hardware inventory")
    rows = (root / "results.tsv").read_text().splitlines()[1:]
    cases = {}
    for row in rows:
        fields = row.split("\t")
        if len(fields) != 5 or fields[1:3] != ["GREEN", "0"]:
            raise ValueError(f"{root}: incomplete or failed numerical gate")
        name = fields[0]
        if Path(name).name != name or name in cases:
            raise ValueError(f"{root}: invalid/duplicate case: {name}")
        for filename, expected_sha in zip(
            ("oracle_manifest.json", "dump_manifest.json"), fields[3:]
        ):
            if digest(root / name / filename) != expected_sha:
                raise ValueError(f"{name}: {filename} changed")
        directory = root / name / "native"
        manifest = json.loads((directory / "manifest.json").read_text())
        if manifest.get("schema") != SCHEMA or manifest.get("mode") != "IDENTICAL":
            raise ValueError(f"{name}: invalid byte manifest")
        if manifest.get("source_sha256") != env.get("source_sha256"):
            raise ValueError(f"{name}: source fingerprint mismatch")
        tensors = manifest["tensors"]
        required = manifest["public_prefill_leaves"] + manifest["state_boundary_leaves"]
        if (not required or len(set(required)) != len(required) or
                set(required) != set(tensors)):
            raise ValueError(f"{name}: incomplete byte inventory")
        dump = json.loads((root / name / "dump_manifest.json").read_text())
        reference = json.loads((root / name / "oracle_manifest.json").read_text())
        validate_metadata(dump, reference)
        if name not in CASES or (dump["family"], dump["case"]) != CASES[name]:
            raise ValueError(f"{name}: wrong certificate fixture")
        for key in ("family", "case", "objective", "public_prefill_leaves"):
            if manifest.get(key) != dump.get(key):
                raise ValueError(f"{name}: native/dump {key} mismatch")
        for key, default in (("state_boundary_leaves", []),
                             ("state_boundary_policy", None)):
            if manifest.get(key, default) != dump.get(key, default):
                raise ValueError(f"{name}: native/dump {key} mismatch")
        for tensor, entry in tensors.items():
            filename = entry["file"]
            if Path(filename).name != filename or entry["dtype"] != "<f4":
                raise ValueError(f"{name}/{tensor}: invalid native file")
            path = directory / filename
            if entry["shape"] != reference["gradients"][tensor]["shape"]:
                raise ValueError(f"{name}/{tensor}: oracle shape mismatch")
            if path.stat().st_size != 4 * math.prod(entry["shape"]):
                raise ValueError(f"{name}/{tensor}: wrong byte count")
            if digest(path) != entry["sha256"]:
                raise ValueError(f"{name}/{tensor}: retained bytes changed")
        cases[name] = manifest
    expected = set(CASES)
    if set(cases) != expected:
        raise ValueError(f"{root}: expected exactly {sorted(expected)}")
    return env, cases


def compare(roots):
    baseline_env, baseline = load_certificate(roots[0])
    for root in roots[1:]:
        env, cases = load_certificate(root)
        for key in ("commit", "source_sha256", "mode"):
            if not env.get(key) or env[key] != baseline_env.get(key):
                raise ValueError(f"{root}: {key} differs from baseline")
        for case, manifest in baseline.items():
            other = cases[case]
            for key in manifest.keys() - {"tensors"}:
                if manifest[key] != other.get(key):
                    raise ValueError(f"{root}/{case}: {key} differs")
            for name, entry in manifest["tensors"].items():
                if entry != other["tensors"].get(name):
                    raise ValueError(f"{root}/{case}/{name}: native bytes/shape differ")
        print(f"BITWISE PASS: {roots[0]} == {root}; "
              f"devices={baseline_env.get('vendor')},{env.get('vendor')}; "
              f"{sum(len(m['tensors']) for m in cases.values())} gradient tensors")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    source = sub.add_parser("source")
    source.add_argument("root", type=Path)
    record = sub.add_parser("capture")
    record.add_argument("actual", type=Path)
    record.add_argument("oracle", type=Path)
    record.add_argument("output", type=Path)
    record.add_argument("source_sha256")
    diff = sub.add_parser("compare")
    diff.add_argument("certificates", nargs="+", type=Path)
    check = sub.add_parser("validate")
    check.add_argument("certificate", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "source":
            print(source_digest(args.root))
        elif args.command == "capture":
            capture(args.actual, args.oracle, args.output, args.source_sha256)
        elif args.command == "validate":
            load_certificate(args.certificate)
            print(f"MAMBA BACKWARD BYTE CERTIFICATE VALID: {args.certificate}")
        else:
            if len(args.certificates) < 2:
                parser.error("compare requires at least two certificates")
            compare(args.certificates)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        print(f"MAMBA BACKWARD IDENTITY FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
