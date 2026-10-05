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


def child():
    import numpy as np
    from mojolearn import IsolationForest

    x = ((np.arange(257 * 7).reshape(257, 7) % 41) - 20).astype(np.float32)
    model = IsolationForest(n_estimators=3, max_samples=64, random_state=17)
    model.fit(x)
    assert model.__dict__.get("_if_token", 0), "resident path was not enabled"
    first = np.asarray(model.score_samples(x)).copy()
    assert np.array_equal(first.view(np.uint32),
                          np.asarray(model.score_samples(x)).view(np.uint32))

    restored = pickle.loads(pickle.dumps(model))
    assert "_if_token" not in restored.__dict__
    assert "_if_finalizer" not in restored.__dict__
    assert np.array_equal(first.view(np.uint32),
                          np.asarray(restored.score_samples(x)).view(np.uint32))
    assert restored._if_token != model._if_token
    old_finalizer = model._if_finalizer
    model.fit(x)
    assert not old_finalizer.alive, "refit retained the old native handle"
    assert model._if_finalizer.alive

    # Exceed the native cache capacity; a evicted estimator must reconstruct
    # its own forest and register the replacement token for orderly teardown.
    held = [model, restored]
    for seed in range(9):
        held.append(IsolationForest(n_estimators=2, max_samples=32,
                                    random_state=seed).fit(x))
    assert np.array_equal(first.view(np.uint32),
                          np.asarray(model.score_samples(x)).view(np.uint32))
    finalizer = restored._if_finalizer
    held.remove(restored)
    del restored
    gc.collect()
    assert not finalizer.alive, "collection retained the native handle"
    # Retain a cycle at interpreter exit to exercise weakref's atexit path.
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
