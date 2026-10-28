# SPDX-License-Identifier: Apache-2.0
"""Light same-bits and same-refusals fixture for admit-once module lookups.

Test controller only (never runtime). Runs tiny deterministic neural
operations on the CPU column in identical mode and writes one JSON report:
the output digest of each public operation, and the exception type and
message of each refusal (wrong dtype, wrong shape, unfitted, bad class,
missing module registry, replaced module, changed export, an export changed
while the operation waits with the GIL released). Run it against two builds
of the host bindings (MOJOLEARN_HOST_DIR) and compare the reports; every
entry must be equal.

usage: MOJOLEARN_VENDOR=cpu MOJOLEARN_HOST_DIR=<dir> python overhead_module_lookup_fixture.py <out.json>
"""
import builtins
import hashlib
import json
import sys
import tempfile
import threading
import time
import types
from pathlib import Path

import numpy as np  # controller-only: builds inputs and hashes returned bytes

import mojolearn
from mojolearn import TransformerBlock, TransformerState


def digest(value):
    a = np.asarray(value)
    return f"{a.dtype.str}{list(a.shape)}:" + hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def outcome(fn):
    try:
        out = fn()
    except Exception as exc:  # noqa: BLE001 - the report records the type
        return dict(raised=type(exc).__name__, message=str(exc))
    if out is None:
        return dict(raised=None)
    if isinstance(out, dict):
        return dict(raised=None, digest={k: (digest(v) if hasattr(v, "shape") or isinstance(v, (list, tuple))
                                             else repr(v)) for k, v in sorted(out.items())})
    return dict(raised=None, digest=digest(out))


class export_changed:
    """Replace one module export by an equivalent wrapper for a block, then restore it."""

    def __init__(self, module, name):
        self.module, self.name = module, name

    def __enter__(self):
        self.saved = getattr(self.module, self.name)
        saved = self.saved
        setattr(self.module, self.name, lambda *a, **k: saved(*a, **k))

    def __exit__(self, *exc):
        setattr(self.module, self.name, self.saved)
        return False


class module_replaced:
    """Replace one sys.modules entry (or remove it) for a block, then restore it."""

    def __init__(self, name, remove=False):
        self.name, self.remove = name, remove

    def __enter__(self):
        self.saved = sys.modules[self.name]
        if self.remove:
            del sys.modules[self.name]
        else:
            sys.modules[self.name] = types.ModuleType(self.name)

    def __exit__(self, *exc):
        sys.modules[self.name] = self.saved
        return False


def connection_refusals(run):
    """The module-connection refusals of one operation, then a clean rerun."""
    r = {}
    with export_changed(builtins, "ascii"):
        r["builtins_export_changed"] = outcome(run)
    with export_changed(types, "SimpleNamespace"):
        r["types_export_changed"] = outcome(run)
    with module_replaced("types"):
        r["types_module_replaced"] = outcome(run)
    with module_replaced("mojolearn._native_loaded_modules", remove=True):
        r["registry_missing"] = outcome(run)
    r["after_restore"] = outcome(run)
    return r


def changed_while_waiting(lock, run):
    """Hold the operation's lock in a second thread, so the operation waits for
    it with the GIL released; change a builtins export during that wait, then
    release. The report records what the operation did after the wait."""
    if lock is None or not hasattr(lock, "acquire"):
        return dict(skipped="operation has no runtime lock")
    held = threading.Event()
    done = threading.Event()
    saved = builtins.ascii

    def holder():
        lock.acquire()
        held.set()
        time.sleep(0.3)
        builtins.ascii = lambda *a, **k: saved(*a, **k)
        lock.release()
        done.set()

    t = threading.Thread(target=holder)
    t.start()
    held.wait()
    try:
        result = outcome(run)
    finally:
        t.join()
        builtins.ascii = saved
    result["changed_during_wait"] = done.is_set()
    return result


def transformer_block():
    d, it = 16, 32
    rng = np.random.default_rng(27)
    names = TransformerBlock._W_NAMES
    shapes = ((d,), (d,), (d, d), (d, d), (d, d), (d, d), (it, d), (it, d), (d, it))
    weights = {name: rng.normal(0, .01, shape).astype(np.float32) for name, shape in zip(names, shapes)}
    weights[names[0]].fill(1)
    weights[names[1]].fill(1)
    return TransformerBlock(weights, n_heads=1, numeric_mode="identical")


def transformer_case():
    import array
    blk = transformer_block()
    x = np.random.default_rng(3).standard_normal((2, 5, 16)).astype(np.float32)
    out = {}
    out["forward"] = outcome(lambda: blk.forward(x))

    def with_state():
        keys = array.array("f", [0.0] * (2 * 1 * 8 * 16))
        values = array.array("f", [0.0] * (2 * 1 * 8 * 16))
        state = TransformerState(2, 1, 16, 8, keys, values, 0, 0)
        y = blk.forward(x, state)
        return dict(y=np.asarray(y), keys=np.asarray(state.keys()), values=np.asarray(state.values()))
    out["forward_state"] = outcome(with_state)
    ref = {}
    ref["float64_input"] = outcome(lambda: blk.forward(x.astype(np.float64)))
    ref["wrong_width"] = outcome(lambda: blk.forward(np.zeros((2, 5, 8), np.float32)))
    ref["rank2_input"] = outcome(lambda: blk.forward(x[0]))
    ref["bad_class"] = outcome(lambda: TransformerBlock.forward(object(), x))
    ref.update(connection_refusals(lambda: blk.forward(x)))
    fields = getattr(blk, "_native_block_fields", None)
    lock = fields[0].get("_runtime_lock") if fields else None
    ref["export_changed_while_waiting"] = changed_while_waiting(lock, lambda: blk.forward(x))
    return dict(same_bits=out, refusals=ref)


def mamba2_case():
    rng = np.random.default_rng(5)
    dm, di, nh = 32, 64, 1
    cd, dip = di + 256, 2 * di + 256 + nh
    shapes = {"block_norm.weight": (dm,), "in_proj.weight": (dip, dm), "conv1d.weight": (cd, 1, 4),
              "conv1d.bias": (cd,), "dt_bias": (nh,), "A_log": (nh,), "D": (nh,), "norm.weight": (di,),
              "out_proj.weight": (dm, di)}
    w = {k: (np.ones(s, np.float32) if k.endswith("norm.weight")
             else (rng.standard_normal(s) / 8).astype(np.float32)) for k, s in shapes.items()}
    blk = mojolearn.Mamba2Block(w)
    x = rng.standard_normal((2, 9, 32)).astype(np.float32)
    out = {}
    out["forward"] = outcome(lambda: blk.forward(x))

    def prefill_step():
        st = blk.allocate_state(2)
        head = blk.forward(np.ascontiguousarray(x[:, :8]), st)
        tail = blk.step(np.ascontiguousarray(x[:, 8:]), st)
        return dict(head=np.asarray(head), tail=np.asarray(tail))
    out["prefill_step"] = outcome(prefill_step)
    ref = {}
    ref["float64_input"] = outcome(lambda: blk.forward(x.astype(np.float64)))
    ref["wrong_width"] = outcome(lambda: blk.forward(np.zeros((2, 9, 16), np.float32)))
    ref["bad_class"] = outcome(lambda: type(blk).forward(object(), x))
    ref.update(connection_refusals(lambda: blk.forward(x)))
    return dict(same_bits=out, refusals=ref)


def byte_lm_case():
    from mojolearn import SmallByteLanguageModelTrainer

    def trainer():
        return SmallByteLanguageModelTrainer(np.zeros(34944, dtype=np.float32), data_schedule={
            'corpus_sha256': '1' * 64, 'token_schedule_sha256': '2' * 64,
            'batch_offsets': [0, 32], 'planned_steps': 4})
    ids = (np.arange(66, dtype=np.int32) % 256).reshape(2, 33)

    def step():
        model = trainer()
        metrics = model.train_step(ids)
        state = model.state_dict()
        return dict(metrics=repr(metrics), parameters=np.asarray(state["parameters"]),
                    m=np.asarray(state["m"]), v=np.asarray(state["v"]))
    out = {"train_step": outcome(step)}
    model = trainer()
    ref = {}
    ref["float_ids"] = outcome(lambda: model.train_step(ids.astype(np.float32)))
    ref["rank1_ids"] = outcome(lambda: model.train_step(ids[0]))
    ref["out_of_range_ids"] = outcome(lambda: model.train_step(ids + 300))
    ref["bad_class"] = outcome(lambda: SmallByteLanguageModelTrainer.train_step(object(), ids))
    ref.update(connection_refusals(lambda: trainer().train_step(ids)))
    return dict(same_bits=out, refusals=ref)


def sequential_case():
    rng = np.random.default_rng(11)
    X = rng.standard_normal((24, 4)).astype(np.float32)
    y = (X @ np.array([0.5, -1.0, 0.25, 2.0], np.float32)).astype(np.float32)

    def fitted():
        m = mojolearn.Sequential().linear(4).relu().linear(1).compile(loss="mse", optimizer="sgd", lr=0.03)
        m.fit(X, y, epochs=2, batch_size=8, seed=0)
        return m
    out = {}
    out["fit_predict"] = outcome(lambda: dict(loss=np.asarray(fitted().loss_curve_, np.float64),
                                              predict=np.asarray(fitted().predict(X))))
    model = fitted()
    ref = {}
    ref["unfitted_predict"] = outcome(lambda: mojolearn.Sequential().linear(1).compile(
        optimizer="sgd", lr=0.03).predict(X))
    ref["wrong_width"] = outcome(lambda: model.predict(np.zeros((3, 7), np.float32)))
    ref["string_input"] = outcome(lambda: model.predict([["a", "b", "c", "d"]]))
    ref["bad_class"] = outcome(lambda: type(model).predict(object(), X))
    ref.update(connection_refusals(lambda: model.predict(X)))
    return dict(same_bits=out, refusals=ref)


def causal_lm_case():
    from mojolearn import _causal_lm_fixtures as fixtures
    from mojolearn.models import CausalLM
    ids = np.array([[1, 3, 5], [2, 4, 6]], dtype="i4")
    report = {}
    with tempfile.TemporaryDirectory() as directory:
        for family in ("llama", "mamba2"):
            path = Path(directory) / family
            cfg, tensors = fixtures.family_fixture(family, False)
            fixtures._write_checkpoint(path, cfg, tensors)
            model = CausalLM.load(path, device="cpu", numeric_profile="fp32_v1")
            out = {"forward": outcome(lambda: model.forward(ids)),
                   "generate": outcome(lambda: model.generate(ids, 4))}
            ref = {"float_ids": outcome(lambda: model.forward(ids.astype("f4"))),
                   "rank1_ids": outcome(lambda: model.forward(ids[0])),
                   "missing_checkpoint": outcome(lambda: CausalLM.load(Path(directory) / "absent", device="cpu"))}
            ref.update(connection_refusals(lambda: model.forward(ids)))
            report[family] = dict(same_bits=out, refusals=ref)
    return report


def main():
    report = {}
    for name, case in (("transformer", transformer_case), ("mamba2", mamba2_case), ("byte_lm", byte_lm_case),
                       ("sequential", sequential_case), ("causal_lm", causal_lm_case)):
        try:
            report[name] = case()
        except Exception as exc:  # noqa: BLE001 - a setup failure is reported, never hidden
            report[name] = dict(setup_failed=type(exc).__name__, message=str(exc))
    Path(sys.argv[1]).write_text(json.dumps(report, indent=1, sort_keys=True, default=str) + "\n")
    print("wrote", sys.argv[1])


if __name__ == "__main__":
    main()
