#!/usr/bin/env python3
"""Diagnostic (r6-proving, 2026-11-23): where a forest, its loaded copy and its CPU twin HostForest part in `digest_`.

The L40S proving run of 0.8.67.dev0+4519097d380a failed tests/api/test_digest.py
`test_a_model_and_its_loaded_copies_have_one_digest[RandomForestRegressor]` (GPU device, n_estimators=5, 400 x 6):
`host_model(path).digest_ != model.digest_` while `type(model).load(path).digest_ == model.digest_`. The same check
passed for device="cpu" (tests/proving/test_proving_digest.py) and for the GPU at 20 trees on 1000 x 8 (feature proof).

For every case below this prints one line per side (model before save, model after the loads, loaded copy, twin) with
its digest, then the SHA-256 of every field the forest digest steps read (python/mojolearn/_digest_fields.py
`forest_steps`) and of every npz member of the saved file, so one run names the side that moved and the field.
Testing tool only (AGENTS.md "No Python in the runtime": a test controller); it times nothing.

Run from the installed wheel's venv: `python tools/ops/forest_digest_diff.py [out.txt]`. The cases are independent
and each fit takes milliseconds on one GPU, so they run one after another in one process (the GPU is one device).
"""
import hashlib
import io
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np

import mojolearn

FIELDS = ("estimator", "family", "_inference_engine", "inference_engine", "inference_engine_", "numeric_mode",
          "_fit_numeric_mode", "_offsets", "_colid", "_quesval", "_left_child", "_leaves", "n_features_in_",
          "_n_trees", "_num_outputs", "classes_", "n_classes_", "device")


def _data(rows, cols):
    rng = np.random.default_rng(0)
    X = rng.random((rows, cols), dtype=np.float32)
    y = X @ np.arange(1, cols + 1, dtype=np.float32)
    return X, y, (y > np.median(y)).astype(np.int32)


def _describe(value):
    if value is None:
        return "None"
    if isinstance(value, (str, bool, int, float)):
        return type(value).__name__ + ":" + repr(value)
    try:
        array = np.asarray(value)
    except Exception as error:                      # a value NumPy cannot view: say what it is
        return type(value).__name__ + " (" + type(error).__name__ + ")"
    raw = np.ascontiguousarray(array).tobytes()
    return f"{type(value).__name__} {array.dtype.str}{list(array.shape)} {hashlib.sha256(raw).hexdigest()[:16]}"


def _fields(model):
    held = vars(model)
    return {name: _describe(held[name]) if name in held else "-" for name in FIELDS}


def _members(path):
    out = {}
    with zipfile.ZipFile(path) as archive:
        for name in sorted(archive.namelist()):
            raw = archive.read(name)
            try:
                array = np.load(io.BytesIO(raw), allow_pickle=False)
                out[name] = f"{array.dtype.str}{list(array.shape)} {hashlib.sha256(array.tobytes()).hexdigest()[:16]}"
            except Exception:
                out[name] = f"raw {hashlib.sha256(raw).hexdigest()[:16]}"
    return out


def case(write, name, params, target, device, rows, cols, folder):
    X, y, c = _data(rows, cols)
    klass = getattr(mojolearn, name)
    kept = dict(params, **({"device": "cpu"} if device == "cpu" else {}))
    model = klass(**kept)
    model.fit(X) if target is None else model.fit(X, y if target == "r" else c)
    head = f"{name} {params} device={device} {rows}x{cols}"
    before = model.digest_
    path = str(Path(folder) / f"{name}-{device}-{rows}.npz")
    model.save(path)
    loaded = klass.load(path)
    twin = mojolearn.host_model(path)
    sides = {"model-before": (before, _fields(model)), "loaded": (loaded.digest_, _fields(loaded)),
             "twin": (getattr(twin, "digest_", "no digest_"), _fields(twin)),
             "model-after": (model.digest_, _fields(model))}
    verdict = "SAME" if len({digest for digest, _ in sides.values()}) == 1 else "DIFFERENT"
    write(f"CASE {head} {verdict}")
    for side, (digest, _) in sides.items():
        write(f"  {side:13s} {digest}")
    for field in FIELDS:
        seen = {side: fields[field] for side, (_, fields) in sides.items()}
        if len(set(seen.values())) > 1:
            write(f"  FIELD {field}: " + " | ".join(f"{side}={text}" for side, text in seen.items()))
    for member, text in _members(path).items():
        write(f"  MEMBER {member} {text}")


CASES = [
    ("RandomForestRegressor", dict(n_estimators=5, random_state=0), "r"),
    ("RandomForestRegressor", dict(n_estimators=20, random_state=0), "r"),
    ("RandomForestClassifier", dict(n_estimators=5, random_state=0), "c"),
    ("RandomForestClassifier", dict(n_estimators=5, random_state=0, oob_score=True), "c"),
    ("ExtraTreesRegressor", dict(n_estimators=5, random_state=0), "r"),
    ("ExtraTreesClassifier", dict(n_estimators=5, random_state=0), "c"),
    ("IsolationForest", dict(n_estimators=5, random_state=0), None),
]


def main():
    out = open(sys.argv[1], "w") if len(sys.argv) > 1 else sys.stdout
    write = lambda line: print(line, file=out, flush=True)  # noqa: E731
    devices = ["gpu", "cpu"] if mojolearn.vendor() != "cpu" else ["cpu"]
    failed = 0
    with tempfile.TemporaryDirectory() as folder:
        for name, params, target in CASES:            # glue: a fixed list of tiny fits on one GPU
            for device in devices:
                for rows, cols in ((400, 6), (1000, 8)):
                    try:
                        case(write, name, params, target, device, rows, cols, folder)
                    except Exception as error:     # one case's refusal is reported, the others still run
                        failed += 1
                        write(f"CASE {name} {params} device={device} {rows}x{cols} ERROR "
                              f"{type(error).__name__}: {error}")
    write(f"SUMMARY cases={len(CASES) * len(devices) * 2} errors={failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
