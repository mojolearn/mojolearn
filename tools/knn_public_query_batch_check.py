#!/usr/bin/env python3
"""Public Python IDENTICAL automatic batching versus explicit256, all words.

Run with the rebuilt base binding and matching Python package on PYTHONPATH.
--expected-vendor {cuda,metal} and --expected-tile {512,256} are required;
actual binding vendor/mode are checked independently of those expectations.
No timing, opponent calls, tree estimators, or non-IDENTICAL mode execution.
"""
import argparse
import hashlib
import json
import os


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-vendor", choices=("cuda", "metal"), required=True)
    parser.add_argument("--expected-tile", type=int, choices=(256, 512), required=True)
    args = parser.parse_args()
    os.environ["MOJOLEARN_NUMERIC_MODE"] = "identical"
    import numpy as np
    from mojolearn.neighbors import NearestNeighbors, KNeighborsClassifier, KNeighborsRegressor

    n, q, d, k = 513, 513, 17, 15
    features = np.arange(d, dtype=np.int64)[None, :]
    rows = np.arange(n, dtype=np.int64)[:, None] % 257
    query_rows = np.arange(q, dtype=np.int64)[:, None]
    # Distinct dyadic query/index mixers, duplicate index rows for ties.
    index = (((rows * 37 + features * 19 + rows * features * 3) % 509 - 254) / 256).astype(np.float32)
    queries = (((query_rows * 71 + features * 13 + query_rows * features * 5 + 43) % 503 - 251) / 256).astype(np.float32)
    labels = ((np.arange(n, dtype=np.int64) * 7) % 3 - 1).astype(np.int32)
    targets = (((np.arange(n, dtype=np.int64) * 23) % 127 - 63) / 32).astype(np.float32)
    original = tuple(array.tobytes() for array in (index, queries, labels, targets))
    evidence = []

    def check_model(model, expected_tile):
        assert model.vendor_used() == args.expected_vendor, model.vendor_used()
        assert int(model._bind().mojolearn_numeric_mode()) == 1, "binding is not IDENTICAL"
        assert model.used_query_tile_ == expected_tile, (type(model).__name__, model.used_query_tile_, expected_tile)

    def compare(label, first, second):
        first, second = np.asarray(first), np.asarray(second)
        assert first.shape == second.shape and first.dtype == second.dtype, label
        assert first.tobytes() == second.tobytes(), f"{label}: output words differ"
        evidence.append({"output": label, "shape": list(first.shape), "dtype": str(first.dtype),
                         "sha256": hashlib.sha256(first.tobytes()).hexdigest()})

    for cls, y in ((NearestNeighbors, None), (KNeighborsClassifier, labels), (KNeighborsRegressor, targets)):
        auto = cls(n_neighbors=k, numeric_mode="identical")
        explicit = cls(n_neighbors=k, query_tile=256, numeric_mode="identical")
        assert auto.query_tile == 0, f"{cls.__name__} Python default is not automatic"
        auto.fit(index) if y is None else auto.fit(index, y)
        explicit.fit(index) if y is None else explicit.fit(index, y)
        if cls is NearestNeighbors:
            actual_d, actual_i = auto.kneighbors(queries)
            expected_d, expected_i = explicit.kneighbors(queries)
            compare("nearest distances", actual_d, expected_d)
            compare("nearest indices", actual_i, expected_i)
        else:
            compare(cls.__name__ + " predict", auto.predict(queries), explicit.predict(queries))
        check_model(auto, args.expected_tile)
        check_model(explicit, 256)
        if cls is KNeighborsClassifier:
            compare("classifier classes", auto.classes_, explicit.classes_)
            compare("classifier probabilities", auto.predict_proba(queries), explicit.predict_proba(queries))
            check_model(auto, args.expected_tile)
            check_model(explicit, 256)
    assert original == tuple(array.tobytes() for array in (index, queries, labels, targets)), "input mutated"
    print(json.dumps({"verdict": "PUBLIC PYTHON QUERY BATCH PASS", "vendor": args.expected_vendor,
                      "numeric_mode": "identical", "automatic_tile": args.expected_tile,
                      "explicit_tile": 256, "fixture": {"index": n, "queries": q, "features": d, "k": k},
                      "outputs": evidence}, indent=2))


if __name__ == "__main__":
    main()
