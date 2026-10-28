# SPDX-License-Identifier: Apache-2.0
"""Light same-bits and same-refusals fixture for causal LM generation guards.

Test controller only (never runtime). Runs tiny deterministic CausalLM
checkpoints on the CPU column in identical mode and prints one JSON report:
output digests per public operation, and the exception type and message of
every guarded refusal. Run it against two builds of the host bindings
(MOJOLEARN_HOST_DIR) and compare the reports.

usage: MOJOLEARN_VENDOR=cpu MOJOLEARN_HOST_DIR=<dir> python overhead_generate_guards_fixture.py <out.json>
"""
import hashlib
import json
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np  # controller-only: hashes returned bytes

import mojolearn  # noqa: F401
from mojolearn import _causal_lm_fixtures as fixtures
from mojolearn._array import Array
from mojolearn.models import CausalLM

FAMILIES = ("llama", "mamba", "mamba2")
OPTIONS = dict(greedy=False, seed=0xFEDCBA9876543210, temperature=0.8, top_k=9, top_p=0.91)


def digest(value):
    a = np.asarray(value)
    return f"{a.dtype.str}{list(a.shape)}:" + hashlib.sha256(a.tobytes()).hexdigest()


def outcome(fn):
    try:
        out = fn()
    except Exception as exc:  # noqa: BLE001 - the report records the type
        return dict(raised=type(exc).__name__, message=str(exc))
    return dict(raised=None, digest=digest(out) if out is not None else None)


class patched:
    """Set one Array class attribute for the duration of a block, then restore it."""

    def __init__(self, name, value):
        self.name, self.value = name, value

    def __enter__(self):
        self.had = self.name in Array.__dict__
        self.saved = Array.__dict__.get(self.name)
        setattr(Array, self.name, self.value)

    def __exit__(self, *exc):
        if self.had:
            setattr(Array, self.name, self.saved)
        else:
            delattr(Array, self.name)
        return False


def same_bits(model):
    ids = np.array([[1, 3, 5], [2, 4, 6]], dtype="i4")
    out = {}
    out["forward"] = digest(model.forward(ids))
    state = model.allocate_state(2, 16)
    _ = model.forward(ids, state)
    out["step"] = digest(model.step(np.array([7, 8], dtype="i4"), state))
    out["generate_greedy"] = digest(model.generate(ids, 6))
    out["generate_greedy_eos"] = digest(model.generate(ids, 6, eos_token_id=[11, 17, 23, 29, 31, 37]))
    out["generate_sampled"] = digest(model.generate(ids, 6, **OPTIONS))
    out["generate_sampled_eos"] = digest(model.generate(ids, 6, eos_token_id=list(range(0, 64, 3)), **OPTIONS))
    stream = model.stream(ids, 6, **OPTIONS)
    parts = [np.asarray(stream.read(2)), np.asarray(stream.read(1)), np.asarray(stream.read(8))]
    out["stream_reads"] = [digest(p) for p in parts]
    out["stream_info"] = stream.info()
    greedy_stream = model.stream(ids, 4)
    out["stream_greedy"] = digest(np.concatenate([np.asarray(greedy_stream.read(1)) for _ in range(4)], axis=1))
    return out


def refusals(model):
    ids = np.array([[1, 3, 5], [2, 4, 6]], dtype="i4")
    r = {}
    r["float_ids"] = outcome(lambda: model.generate(ids.astype("f4"), 3))
    r["rank1_ids"] = outcome(lambda: model.generate(ids[0], 3))
    r["negative_max_new"] = outcome(lambda: model.generate(ids, -1))
    r["sampling_without_seed"] = outcome(lambda: model.generate(ids, 3, greedy=False))
    r["out_of_vocab"] = outcome(lambda: model.generate(ids + 1000, 3))
    for hook in ("__getattr__", "__getattribute__", "__setattr__", "__del__"):
        def run(hook=hook):
            with patched(hook, (lambda self, *a: object.__getattribute__(self, a[0])) if hook == "__getattribute__"
                         else (lambda self, *a: None)):
                return model.generate(ids, 3)
        r["hook_" + hook] = outcome(run)
    shape = Array.__dict__["_addr"]

    def descriptor():
        with patched("_addr", property(lambda self: shape.__get__(self, Array))):
            return model.generate(ids, 3)
    r["descriptor_replaced"] = outcome(descriptor)
    r["after_restore"] = outcome(lambda: model.generate(ids, 3))

    def stream_between_reads():
        stream = model.stream(ids, 6)
        first = stream.read(1)
        with patched("__getattr__", lambda self, name: None):
            second = outcome(lambda: stream.read(1))
        third = outcome(lambda: stream.read(1))
        return dict(first=digest(first), mutated=second, after=third, info=stream.info())
    r["stream_mutated_between_reads"] = stream_between_reads()
    r["mutated_during_generate"] = mutated_during(model, ids)
    return r


def mutated_during(model, ids):
    # A second thread installs a hook while generation runs: it can only run
    # while generation releases the GIL. The report records which outcome
    # happened; any refusal must be the guard's TypeError.
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    go = threading.Event()

    def mutate():
        go.wait()
        Array.__getattr__ = lambda self, name: None

    t = threading.Thread(target=mutate)
    t.start()
    try:
        go.set()
        result = outcome(lambda: model.generate(ids, 40))
    finally:
        t.join()
        if "__getattr__" in Array.__dict__:
            del Array.__getattr__
        sys.setswitchinterval(old)
    return result


def main():
    report = dict(families={})
    with tempfile.TemporaryDirectory() as directory:
        for family in FAMILIES:
            path = Path(directory) / family
            cfg, tensors = fixtures.family_fixture(family, False)
            fixtures._write_checkpoint(path, cfg, tensors)
            model = CausalLM.load(path, device="cpu", numeric_profile="fp32_v1")
            report["families"][family] = dict(same_bits=same_bits(model), refusals=refusals(model))
    Path(sys.argv[1]).write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    print("wrote", sys.argv[1])


if __name__ == "__main__":
    main()
