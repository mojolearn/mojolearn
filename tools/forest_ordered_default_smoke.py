#!/usr/bin/env python3
"""Smoke: default AUTO selects ordered resident and matches sequential exactly."""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))


def load_data(name, rows):
    import speed_gbdt_arm as data
    if name == "taxi":
        d = data.load_taxi("shipped", regression=False)
        cols = [data.TAXI_FEATURES.index(c) for c in data.TAXI_NUMERIC]
        return np.ascontiguousarray(d.X_train[:rows, cols], dtype=np.float32)
    d = data.load_istella("shipped", regression=False)
    return np.ascontiguousarray(d.X_train[:rows], dtype=np.float32)


def sha(value):
    a = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(a.tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--rows", type=int, default=100_000)
    ap.add_argument("--json", required=True)
    ap.add_argument("--expected-mode", choices=("fast", "identical"), default="identical")
    ap.add_argument("--expected-vendor")
    args = ap.parse_args()
    import mojolearn as ml
    if ml.numeric_mode() != args.expected_mode:
        raise RuntimeError("numeric mode %r, expected %r" %
                           (ml.numeric_mode(), args.expected_mode))
    if args.expected_vendor is not None and ml.vendor() != args.expected_vendor:
        raise RuntimeError("vendor %r, expected %r" %
                           (ml.vendor(), args.expected_vendor))
    records = []
    for dataset in ("taxi", "istella"):
        x = load_data(dataset, args.rows)
        for kind, cls in (("rf", ml.RandomForestClassifier),
                          ("et", ml.ExtraTreesClassifier)):
            model = cls.load(os.path.join(args.models, "%s-%s.npz" % (dataset, kind)))
            if int(model._bind().forest_ordered_resident()) != 1:
                raise RuntimeError("default binding did not select ordered resident")
            model.inference_engine = "sequential"
            references = {"predict": model.predict(x), "proba": model.predict_proba(x)}
            model.inference_engine = "auto"
            # IDENTICAL auto predicts AS sequential through an ordered
            # resident snapshot (2026-09-22); FAST auto is parallel_groves.
            resident = (model._ordered_resident_auto()
                        if args.expected_mode == "identical"
                        else model._prediction_engine() == "parallel_groves")
            if not resident:
                raise RuntimeError("AUTO did not select resident inference")
            for op, reference in references.items():
                call = model.predict if op == "predict" else model.predict_proba
                expected = sha(reference)
                hashes = [sha(call(x)) for _ in range(3)]
                if any(value != expected for value in hashes):
                    raise RuntimeError("default AUTO output differs: %s/%s/%s" %
                                       (dataset, kind, op))
                records.append({"dataset": dataset, "kind": kind,
                                "operation": op, "rows": len(x),
                                "hash": expected, "repeats": 3})
                print("DEFAULT_EXACT", dataset, kind, op, expected[:16])
    with open(args.json, "w") as fh:
        json.dump(records, fh, indent=2, sort_keys=True)
    print("PASS", ml.vendor(), ml.numeric_mode(), "ordered resident AUTO",
          len(records), "cells")


if __name__ == "__main__":
    main()
