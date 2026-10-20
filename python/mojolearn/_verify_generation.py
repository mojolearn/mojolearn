# SPDX-License-Identifier: Apache-2.0
"""Installed-package checks of native causal generation and its output bytes.

This is a test controller, never part of model execution or a performance
measurement. A capture is explicitly unqualified until independently compared.
"""
import hashlib
import json
import platform
import re
import tempfile
from pathlib import Path

from . import _backend, _causal_lm_fixtures as fixtures
from . import _verify_native_runtime

PROFILE = "native-generation-v1"
CASES = (("llama", "fp32_v1"), ("mamba", "fp32_v1"),
         ("mamba2", "fp32_v1"), ("llama", "fixed15_v1"))
PARTS = frozenset(("logits", "greedy", "sampled", "streamed", "resumed", "eos"))
CHECKS = frozenset(("repeat", "stream", "resume", "eos", "no_python", "corruption_detected"))


def _digest(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def _source():
    root = Path(__file__).parent
    files = (Path(__file__), Path(fixtures.__file__), Path(_verify_native_runtime.__file__),
             root / "models" / "causal_lm.py", root / "models" / "tokenizer.py")
    return hashlib.sha256(b"".join(p.read_bytes() for p in files)).hexdigest()


def _case(path, family, profile, device):
    import numpy as np  # Independent controller joins/hashes outputs only.
    from .models import CausalLM
    cfg, tensors = fixtures.family_fixture(family, False)
    fixtures._write_checkpoint(path, cfg, tensors)
    model = CausalLM.load(path, device=device, numeric_profile=profile)
    binding = model._blocks[0]._extension()
    prefix = "neural_host" if device == "cpu" else ("transformer" if family == "llama" else "mamba")
    vendor = str(getattr(binding, prefix + "_vendor")())
    mode = int(getattr(binding, prefix + "_numeric_mode")())
    if mode != 1 or (device == "cpu" and vendor != "cpu") or (device == "gpu" and vendor not in {"metal", "cuda", "hip"}):
        raise ValueError(f"requested {device}/identical but native entry reports {vendor}/{mode}")
    ids = np.array([[1, 3, 5], [2, 4, 6]], dtype="i4")
    audits = []

    def native(operation, *args, **kwargs):
        audit = _verify_native_runtime.audit_native_call(operation, *args, **kwargs)
        audits.append(audit.report())
        return audit.assert_native()

    logits = native(model.forward, ids)
    greedy = native(model.generate, ids, 5)
    options = dict(greedy=False, seed=0xFEDCBA9876543210, temperature=0.8, top_k=9, top_p=0.91)
    sampled = native(model.generate, ids, 5, **options)
    repeated = native(model.generate, ids, 5, **options)
    stream = native(model.stream, ids, 5, **options)
    first = native(stream.read, 2)
    snapshot = native(stream.state_dict)
    tail = native(stream.read, 8)
    resumed = native(model.resume_generation, snapshot)
    restored_tail = native(resumed.read, 8)
    streamed = np.concatenate((ids, np.asarray(first), np.asarray(tail)), axis=1)
    restored = np.concatenate((ids, np.asarray(first), np.asarray(restored_tail)), axis=1)
    eos = native(model.generate, ids, 5, eos_token_id=list(range(model.vocab_size)), **options)
    corrupt = dict(snapshot)
    corrupt["produced"] += 1
    detected = False
    try:
        model.resume_generation(corrupt)
    except Exception as exc:
        detected = "checksum" in str(exc)
    return dict(family=family, numeric_profile=profile, vendor=vendor, mode=mode,
                checkpoints={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(path.iterdir())},
                parts={name: _digest(value) for name, value in (
                    ("logits", logits), ("greedy", greedy), ("sampled", sampled),
                    ("streamed", streamed), ("resumed", restored), ("eos", eos))},
                checks=dict(repeat=_digest(sampled) == _digest(repeated),
                            stream=_digest(sampled) == _digest(streamed),
                            resume=_digest(sampled) == _digest(restored),
                            eos=eos.shape == (2, 4), no_python=all(a["passed"] for a in audits),
                            corruption_detected=detected),
                audits=audits)


def capture(device="cpu", progress=None):
    if _backend.numeric_mode() != "identical":
        raise ValueError("generation verification requires identical mode")
    result = dict(profile=PROFILE, source_sha256=_source(), device=device,
                  platform=platform.platform(), status="INCOMPLETE", cases=[],
                  native_arithmetic_fault="PENDING", cross_vendor_evidence="PENDING")
    try:
        with tempfile.TemporaryDirectory() as directory:
            for family, profile in CASES:
                path = Path(directory) / (family + "-" + profile)
                result["cases"].append(_case(path, family, profile, device))
                if progress:
                    progress(result)
        result["status"] = "CAPTURED_UNQUALIFIED" if all(all(c["checks"].values()) for c in result["cases"]) else "FAILED"
        # Binary identity is retained separately: architecture-specific bytes
        # naturally differ. Source/recipes and outputs decide comparability.
        from .tokenizer import _binding
        runtime = _binding()
        result["generation_binary"] = dict(path=runtime.__file__, sha256=hashlib.sha256(Path(runtime.__file__).read_bytes()).hexdigest())
    except BaseException as exc:
        result["status"] = "FAILED"
        result["error"] = repr(exc)
        if progress:
            progress(result)
        raise
    if progress:
        progress(result)
    return result


def _cells(report):
    if report.get("profile") != PROFILE or report.get("status") != "CAPTURED_UNQUALIFIED":
        raise ValueError("incomplete or failed generation capture")
    if report.get("device") not in {"cpu", "gpu"}:
        raise ValueError("missing execution device")
    rows = {}
    for case in report.get("cases", []):
        key = (case["family"], case["numeric_profile"])
        if key in rows or key not in CASES:
            raise ValueError("duplicate or unexpected generation case")
        if set(case.get("parts", {})) != PARTS or set(case.get("checks", {})) != CHECKS or not all(case["checks"].values()):
            raise ValueError("missing or failed generation checks/outputs")
        if case.get("mode") != 1 or case.get("vendor") not in ({"cpu"} if report["device"] == "cpu" else {"metal", "cuda", "hip"}):
            raise ValueError("native vendor/mode readback differs from requested execution")
        if len(case.get("audits", [])) != 12 or not all(a.get("passed") for a in case["audits"]):
            raise ValueError("missing or failed native execution audits")
        values = [*case["parts"].values(), *case.get("checkpoints", {}).values()]
        if not case.get("checkpoints") or any(not isinstance(v, str) or not re.fullmatch("[0-9a-f]{64}", v) for v in values):
            raise ValueError("missing or invalid retained digest")
        rows[key] = (case["checkpoints"], case["parts"])
    if set(rows) != set(CASES):
        raise ValueError("generation capture is missing requested cases")
    return rows


def compare(left, right):
    if not left.get("source_sha256") or left.get("source_sha256") != right.get("source_sha256"):
        raise ValueError("generation source differs")
    return _cells(left) == _cells(right)


def cmd_verify_generation(args):
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    def save(report):
        out.write_text(json.dumps(report, indent=2) + "\n")

    try:
        report = capture(args.device, save)
        _cells(report)
        if args.expect:
            expected = json.loads(Path(args.expect).read_text())
            matched = compare(expected, report)
            report["comparison"] = "MATCH" if matched else "DIFFER"
            save(report)
            code = 0 if matched else 1
        else:
            code = 0
        print(json.dumps(dict(status=report["status"], cases=len(report["cases"]),
                              comparison=report.get("comparison", "PENDING"), out=str(out))))
        return code
    except Exception as exc:
        print(json.dumps(dict(status="FAILED", error=str(exc), out=str(out))))
        return 1


def add_parser(sub):
    parser = sub.add_parser("verify-generation", help="verify native generation, sampling, streaming and resume on this install")
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--out", required=True)
    parser.add_argument("--expect", help="independent capture to compare byte for byte")
    parser.set_defaults(func=cmd_verify_generation)
    return parser
