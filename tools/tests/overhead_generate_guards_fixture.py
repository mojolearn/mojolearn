# SPDX-License-Identifier: Apache-2.0
"""Light same-bits and same-refusals fixture for causal LM generation guards.

Test controller only (never runtime). Runs tiny deterministic CausalLM
checkpoints on the CPU column in identical mode and prints one JSON report:
output digests per public operation, and the exception type and message of
every guarded refusal. Run it against two builds of the host bindings
(MOJOLEARN_HOST_DIR) and compare the reports.

usage: MOJOLEARN_VENDOR=cpu MOJOLEARN_HOST_DIR=<dir> python overhead_generate_guards_fixture.py <out.json>
       python overhead_generate_guards_fixture.py compare <before.json> <after.json> <rows.json>
"""
import array
import hashlib
import json
import sys
import tempfile
import threading
from pathlib import Path

import mojolearn  # noqa: F401
from mojolearn import _causal_lm_fixtures as fixtures
from mojolearn._array import Array
from mojolearn.models import CausalLM

FAMILIES = ("llama", "mamba", "mamba2")
OPTIONS = dict(greedy=False, seed=0xFEDCBA9876543210, temperature=0.8, top_k=9, top_p=0.91)


def digest(value):
    view = memoryview(value)
    return f"{view.format}{list(view.shape)}:" + hashlib.sha256(view.tobytes()).hexdigest()


def ids2(values, rows, code="i"):
    flat = memoryview(array.array(code, values))
    return flat.cast("B").cast(code, (rows, len(values) // rows))


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
    ids = ids2([1, 3, 5, 2, 4, 6], 2)
    out = {}
    out["forward"] = outcome(lambda: model.forward(ids))

    def step():
        state = model.allocate_state(2, 16)
        _ = model.forward(ids, state)
        return model.step(array.array("i", [7, 8]), state)
    out["step"] = outcome(step)
    out["generate_greedy"] = outcome(lambda: model.generate(ids, 6))
    out["generate_greedy_eos"] = outcome(lambda: model.generate(ids, 6, eos_token_id=[11, 17, 23, 29, 31, 37]))
    out["generate_sampled"] = outcome(lambda: model.generate(ids, 6, **OPTIONS))
    out["generate_sampled_eos"] = outcome(lambda: model.generate(ids, 6, eos_token_id=list(range(0, 64, 3)), **OPTIONS))

    def stream(**options):
        handle = model.stream(ids, 6, **options)
        return dict(reads=[outcome(lambda n=n: handle.read(n)) for n in (2, 1, 8)], info=handle.info())
    out["stream_sampled"] = stream(**OPTIONS)
    out["stream_greedy"] = stream()
    return out


def refusals(model):
    ids = ids2([1, 3, 5, 2, 4, 6], 2)
    r = {}
    r["float_ids"] = outcome(lambda: model.generate(ids2([1.0, 3.0, 5.0, 2.0, 4.0, 6.0], 2, "f"), 3))
    r["rank1_ids"] = outcome(lambda: model.generate(array.array("i", [1, 3, 5]), 3))
    r["negative_max_new"] = outcome(lambda: model.generate(ids, -1))
    r["sampling_without_seed"] = outcome(lambda: model.generate(ids, 3, greedy=False))
    r["out_of_vocab"] = outcome(lambda: model.generate(ids2([1001, 3, 5, 2, 4, 6], 2), 3))
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
    try:
        r["stream_mutated_between_reads"] = stream_between_reads()
    except Exception as exc:  # noqa: BLE001
        r["stream_mutated_between_reads"] = dict(raised=type(exc).__name__, message=str(exc))
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


def compare(before_path, after_path):
    """Exit 0 when every digest and refusal matches. The threaded mutation case
    races the scheduler, so only its exception type (or success) is compared."""
    before = json.loads(Path(before_path).read_text())
    after = json.loads(Path(after_path).read_text())
    for report in (before, after):
        for family in report["families"].values():
            race = family.get("refusals", {}).get("mutated_during_generate")
            if race is not None:
                race.pop("message", None)
                race.pop("digest", None)
    rows = []

    def walk(a, b, key):
        if isinstance(a, dict) and isinstance(b, dict) and set(a) == set(b) and "raised" not in a:
            for k in sorted(a):
                walk(a[k], b[k], key + "/" + k)
        else:
            rows.append(dict(op=key, before=a, after=b, same=a == b))
    walk(before, after, "")
    same = all(r["same"] for r in rows)
    print(json.dumps(dict(same=same, rows=len(rows), different=[r["op"] for r in rows if not r["same"]])))
    return rows, same


def main():
    if sys.argv[1] == "compare":
        rows, same = compare(sys.argv[2], sys.argv[3])
        Path(sys.argv[4]).write_text(json.dumps(rows, indent=1, default=str) + "\n")
        return 0 if same else 1
    report = dict(families={})
    with tempfile.TemporaryDirectory() as directory:
        for family in FAMILIES:
            path = Path(directory) / family
            cfg, tensors = fixtures.family_fixture(family, False)
            fixtures._write_checkpoint(path, cfg, tensors)
            try:
                model = CausalLM.load(path, device="cpu", numeric_profile="fp32_v1")
                report["families"][family] = dict(same_bits=same_bits(model), refusals=refusals(model))
            except Exception as exc:  # noqa: BLE001 - recorded, and the run exits nonzero
                report["families"][family] = dict(failed=type(exc).__name__ + ": " + str(exc))
    Path(sys.argv[1]).write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    print("wrote", sys.argv[1])
    return 1 if any("failed" in v for v in report["families"].values()) else 0


if __name__ == "__main__":
    sys.exit(main())
