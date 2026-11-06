"""Remote GPU correctness/exit gate for resident IsolationForest handles.

Run with the branch-built package in PYTHONPATH. The parent process bounds
shutdown time; a success line from a child that hangs during exit is a failure.
"""
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys


#: Instance fields an older build kept for the resident forest. The token,
#: the training matrix it was minted for and its release lifetime are now one
#: native side record of the estimator (bindings/trees_side_records.mojo,
#: bindings/trees_public_iforest.mojo `_run`), keyed by a weak reference and
#: invisible to Python, so this gate checks the observable behaviour.
OLD_FIELDS = ("_if_token", "_if_lifetime", "_if_finalizer")


def _bits(model, x):
    import numpy as np
    return np.asarray(model.score_samples(x)).copy().view(np.uint32)


def _fields(model):
    """The instance dictionary as comparable values (keys and value types;
    float and int values by value)."""
    out = {}
    for key, value in vars(model).items():
        out[key] = value if isinstance(value, (bool, int, float, str, type(None))) else type(value).__name__
    return out


def child():
    import weakref

    import numpy as np
    from mojolearn import IsolationForest, _backend

    binding = _backend.binding("_mojolearn_svm", "identical")
    probe = getattr(binding, "iforest_resident", None)
    assert probe is not None and int(probe()) == 1, "resident path was not enabled"

    x = ((np.arange(257 * 7).reshape(257, 7) % 41) - 20).astype(np.float32)
    x2 = x[::-1].copy()
    model = IsolationForest(n_estimators=3, max_samples=64, random_state=17)
    model.fit(x)
    assert not any(k in vars(model) for k in OLD_FIELDS), "the resident token is in __dict__"
    before = _fields(model)
    first = _bits(model, x)
    # Repeated scoring reuses the held forest: the same bits, and the
    # instance dictionary is left as the fit left it (no token written).
    for _ in range(3):
        assert np.array_equal(first, _bits(model, x))
    assert _fields(model) == before, "scoring changed the instance dictionary"

    restored = pickle.loads(pickle.dumps(model))
    assert not any(k in vars(restored) for k in OLD_FIELDS), "a pickle carried a resident field"
    assert np.array_equal(first, _bits(restored, x)), "an unpickled model scores differently"
    assert not any(k in vars(restored) for k in OLD_FIELDS)

    # A refit on other data drops the held forest: its scores are a fresh
    # model's on that data, and a refit back gives the first bits again.
    model.fit(x2)
    fresh = IsolationForest(n_estimators=3, max_samples=64, random_state=17).fit(x2)
    assert np.array_equal(_bits(fresh, x), _bits(model, x)), "refit kept the old forest"
    model.fit(x)
    assert np.array_equal(first, _bits(model, x)), "refit back does not reproduce the first forest"

    # Exceed the native cache capacity; an evicted estimator must reconstruct
    # its own forest and register the replacement token for orderly teardown.
    held = [model, restored, fresh]
    for seed in range(9):
        held.append(IsolationForest(n_estimators=2, max_samples=32,
                                    random_state=seed).fit(x))
    assert np.array_equal(first, _bits(model, x))
    assert np.array_equal(first, _bits(restored, x))

    # The side record holds no reference to its estimator: dropping the
    # estimator collects it (and with it the record and its lifetime, which
    # releases the native forest).
    gone = weakref.ref(restored)
    held.remove(restored)
    del restored
    gc.collect()
    assert gone() is None, "the resident record kept its estimator alive"
    # Retain a cycle at interpreter exit to exercise the exit release path.
    model._gate_cycle = held
    globals()["_retained_models"] = held
    digest = hashlib.sha256(first.tobytes()).hexdigest()
    print(json.dumps({"status": "PASS", "digest": digest,
                      "reuse": True, "refit": True, "pickle": True,
                      "eviction": True, "collection": True}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.child:
        child()
        return
    if args.out is None:
        parser.error("--out is required")
    args.out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, MOJOLEARN_NUMERIC_MODE="identical")
    with (args.out / "child.log").open("w") as log:
        try:
            process = subprocess.run([sys.executable, __file__, "--child"],
                                     env=env, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=180)
            rc = process.returncode
        except subprocess.TimeoutExpired:
            rc = 124
    result = {"status": "PASS" if rc == 0 else "FAIL", "exit_code": rc,
              "shutdown_completed": rc == 0}
    (args.out / "result.json").write_text(json.dumps(result) + "\n")
    print(json.dumps(result))
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
