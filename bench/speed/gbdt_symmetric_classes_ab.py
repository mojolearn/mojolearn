#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Interleaved FAST symmetric-GBDT predict_classes output-boundary gate."""
import argparse
import hashlib
import json
import platform
import statistics
import time

import numpy as np

import mojolearn as ml
from mojolearn._labels import argmax_rows


def digest(value):
    return hashlib.sha256(value.tobytes()).hexdigest()


def one_case(rows, features, classes, depth, seed, distribution, rounds):
    rng = np.random.default_rng(seed)
    train_rows = 8064 if distribution == "tie-heavy" else 8192
    train = rng.normal(size=(train_rows, features)).astype("<f4")
    if distribution == "tie-heavy":
        patterns = (np.arange(train_rows) // classes) % 16
        for feature in range(features):
            train[:, feature] = ((patterns >> (feature % 4)) & 1).astype(np.float32)
    score = train[:, 0] + np.float32(0.25) * train[:, min(1, features - 1)]
    if classes == 2:
        threshold = np.float32(1.2 if distribution == "imbalanced" else 0.0)
        target = ((np.arange(train_rows) & 1) if distribution == "tie-heavy"
                  else score > threshold).astype("<f4")
        loss = "Logloss"
    else:
        if distribution == "imbalanced":
            cuts = np.linspace(-0.3, 2.3, classes - 1, dtype=np.float32)
        else:
            cuts = np.quantile(score, np.arange(1, classes) / classes).astype("<f4")
        target = ((np.arange(train_rows) % classes) if distribution == "tie-heavy"
                  else np.searchsorted(cuts, score, side="right")).astype("<f4")
        loss = "MultiClass"
    model = ml.GradientBoosting(
        loss=loss, n_estimators=8, max_depth=depth, bootstrap_type="No",
        random_strength=0.0, numeric_mode="fast",
    ).fit(np.asfortranarray(train), target)
    query = rng.normal(size=(rows, features)).astype("<f4")
    if distribution == "tie-heavy":
        query.fill(np.float32(0.0))
    query_score = query[:, 0] + np.float32(0.25) * query[:, min(1, features - 1)]
    if classes == 2:
        query_target = (((np.arange(rows) & 1) if distribution == "tie-heavy"
                         else query_score > threshold)).astype(np.int64)
    else:
        query_target = (((np.arange(rows) % classes) if distribution == "tie-heavy"
                         else np.searchsorted(cuts, query_score, side="right"))).astype(np.int64)

    def old():
        return argmax_rows(model.predict_proba(query))

    def new():
        return model.predict_classes(query)

    expected = old()
    actual = new()
    if expected.tobytes() != actual.tobytes():
        raise AssertionError("direct labels differ from predict_proba argmax")
    times = {"old_ms": [], "new_ms": []}
    hashes = set()
    for i in range(rounds):
        for name, fn in (("old_ms", old), ("new_ms", new)) if i % 2 == 0 else (("new_ms", new), ("old_ms", old)):
            start = time.perf_counter()
            value = fn()
            times[name].append((time.perf_counter() - start) * 1000.0)
            hashes.add(digest(value))
    if len(hashes) != 1:
        raise AssertionError("label hash changed across arms or rounds")
    prediction = np.asarray(actual)
    return {
        "rows": rows, "features": features, "classes": classes,
        "depth": depth, "seed": seed, "distribution": distribution,
        "model_sha256": hashlib.sha256(str(model.model_).encode()).hexdigest(),
        "labels_sha256": hashes.pop(),
        "accuracy": float(np.mean(prediction == query_target)),
        **times,
        "old_median_ms": statistics.median(times["old_ms"]),
        "new_median_ms": statistics.median(times["new_ms"]),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=500_000)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--output")
    args = parser.parse_args()
    cases = []
    for classes in (2, 3, 8):
        for seed, features, depth, distribution in (
            (3, 8, 3, "balanced"),
            (17, 32, 6, "imbalanced"),
            (41, 8, 6, "balanced"),
        ):
            case = one_case(args.rows, features, classes, depth, seed, distribution, args.rounds)
            cases.append(case)
            print(json.dumps(case), flush=True)
    report = {
        "hardware": platform.platform(),
        "numpy": np.__version__,
        "numeric_mode": "fast",
        "cases": cases,
    }
    if args.output:
        with open(args.output, "w", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")


if __name__ == "__main__":
    main()
