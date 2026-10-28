#!/usr/bin/env python3
"""Run only after integration builds; this lane does not execute GPU probes."""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import traceback


def qualify():
    import numpy as np
    from mojolearn._causal_lm_fixtures import family_fixture, _write_checkpoint
    from mojolearn.models import CausalLM

    # Force the existing per-layer route on both arms. This probe qualifies
    # the ABI boundary; these tiny fixtures are not performance evidence.
    os.environ["MOJOLEARN_CAUSAL_LM_RESIDENT_FORWARD"] = "0"
    os.environ.pop("MOJOLEARN_TRANSFORMER_LEGACY_SETUP", None)
    with tempfile.TemporaryDirectory(prefix="ml-native-dispatch-") as folder:
        cfg, tensors = family_fixture("llama", False)
        _write_checkpoint(folder, cfg, tensors)
        os.environ["MOJOLEARN_GENERATION_NATIVE_SESSION"] = "0"
        baseline = CausalLM.load(folder, device="gpu", numeric_profile="fp32_v1")
        os.environ["MOJOLEARN_GENERATION_NATIVE_SESSION"] = "1"
        candidate = CausalLM.load(folder, device="gpu", numeric_profile="fp32_v1")
        assert baseline._native_numeric_mode == candidate._native_numeric_mode == 1
        assert "native_session_provider" not in baseline._native_entries
        assert "native_session_provider" in candidate._native_entries
        assert candidate._native_vendor in {"metal", "cuda", "hip"}
        ids = np.array([[1, 4, 7], [2, 5, 8]], dtype="i4")
        states = [model.allocate_state(2, 48) for model in (baseline, candidate)]
        for _ in range(8):
            a = np.asarray(baseline.forward(ids, states[0])).copy()
            b = np.asarray(candidate.forward(ids, states[1])).copy()
            assert np.array_equal(a.view("u4"), b.view("u4"))
            assert states[0].positions == states[1].positions
            assert [layer.cached_tokens for layer in states[0].layers] == [
                layer.cached_tokens for layer in states[1].layers
            ] == [states[1].positions] * len(states[1].layers)
        logits_sha256 = hashlib.sha256(b.tobytes()).hexdigest()
        options = dict(greedy=False, seed=7, top_k=2)
        generated_a = np.asarray(baseline.generate(ids, 5, **options)).copy()
        generated_b = np.asarray(candidate.generate(ids, 5, **options)).copy()
        assert np.array_equal(generated_a, generated_b)
        # Exact retained argument counts after warm use: no added reference
        # per native call. This is a targeted lifetime check, not a leak proof.
        fields = object.__getattribute__(candidate.blocks[0], "__dict__")
        session = fields["_native_session"]
        linked = fields["_native_forward_owner"]
        gc.collect()
        before = (sys.getrefcount(session), sys.getrefcount(linked))
        for _ in range(4):
            candidate.forward(ids, states[1])
        gc.collect()
        assert before == (sys.getrefcount(session), sys.getrefcount(linked))

        # Negative control: the baseline dispatch actually reaches this old
        # callable; the candidate uses its cached native table instead.
        calls = []

        def forbidden(*_):
            calls.append(True)
            raise RuntimeError("poisoned CPython session dispatch")

        baseline._native_entries["forward_session"] = forbidden
        candidate._native_entries["forward_session"] = forbidden
        try:
            baseline.forward(ids, baseline.allocate_state(2, 12))
        except Exception as problem:
            assert "poisoned CPython session dispatch" in str(problem)
        else:
            raise AssertionError("baseline negative control did not fire")
        count = len(calls)
        candidate.forward(ids, candidate.allocate_state(2, 12))
        assert len(calls) == count

        # Public capacity refusal/recovery; it may refuse before the ABI and
        # therefore does NOT qualify failure cleanup inside native execution.
        bad = candidate.allocate_state(2, 1)
        try:
            candidate.forward(ids, bad)
        except Exception:
            pass
        else:
            raise AssertionError("cache capacity negative control did not fire")
        candidate.forward(ids, candidate.allocate_state(2, 12))
        candidate._native_entries["block_binding"].transformer_session_close(session)
        try:
            candidate.forward(ids, candidate.allocate_state(2, 12))
        except Exception as problem:
            assert "busy or closed" in str(problem)
        else:
            raise AssertionError("closed session negative control did not fire")
        fields["_native_session"] = candidate._native_entries["create"]()
        candidate.forward(ids, candidate.allocate_state(2, 12))
        assert fields["_native_forward_owner"] is not linked
        return {"status": "passed", "vendor": candidate._native_vendor,
                "logits_sha256": logits_sha256, "sampled_tokens": generated_b.tolist(),
                "checks": ["8 repeated bitwise forwards", "5-token sampled generation", "state positions",
                           "session/link reference stability", "poisoned dispatch negative control",
                           "public capacity refusal followed by valid execution", "closed session and owner replacement"],
                "not_qualified": ["thread contention", "failure cleanup inside native execution", "all refusal paths", "full-workload performance",
                                  "cross-vendor identity until all vendor outputs are compared"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = qualify()
    except BaseException:
        report = {"status": "failed", "traceback": traceback.format_exc()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "traceback"}))
    return int(report["status"] != "passed")


if __name__ == "__main__":
    raise SystemExit(main())
