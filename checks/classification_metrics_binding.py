#!/usr/bin/env python3
"""GPU public classification gate against sklearn 1.8 and hand-count oracles.

Requires all requested compiled metrics artifacts; never substitutes host
results for missing GPU entry points. Use the local sklearn 1.8 environment
through the repository's GPU-capable Python launcher under the build lock.
"""
import argparse
import json
import math
import warnings

import numpy as np
try:
    import sklearn
    from sklearn import metrics as reference
except ImportError:
    sklearn = reference = None

import mojolearn
from mojolearn import metrics, _backend

CODES = {"fast": 0, "identical": 1, "deterministic": 2}
SCORES = ("precision_score", "recall_score", "f1_score")


def hand_confusion(y, p, labels, normalize):
    rows = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for a, b in zip(y, p):
        if a in rows and b in rows:
            matrix[rows[a]][rows[b]] += 1
    if normalize is None:
        return np.array(matrix, dtype=np.int64)
    total = sum(map(sum, matrix))
    return np.array([
        [value / denominator if denominator else 0.0
         for j, value in enumerate(row)
         for denominator in [sum(row) if normalize == "true" else
             sum(r[j] for r in matrix) if normalize == "pred" else total]]
        for row in matrix], dtype=np.float64)


def hand_scores(y, p, labels, average, zero_division, pos_label):
    selected = [pos_label] if average == "binary" else list(labels)
    # Count independently over ALL observations. Restricting a confusion
    # matrix first would incorrectly discard false positives/negatives when
    # labels is a subset of the observed classes.
    counts = [(sum(a == label and b == label for a, b in zip(y, p)),
               sum(a == label for a in y), sum(b == label for b in p))
              for label in selected]
    fill = 0.0 if zero_division == "warn" else float(zero_division)
    if average == "micro":
        counts = [tuple(sum(c[i] for c in counts) for i in range(3))]
    values = [[], [], []]
    for tp, true, pred in counts:
        values[0].append(tp / pred if pred else fill)
        values[1].append(tp / true if true else fill)
        values[2].append(2 * tp / (true + pred) if true + pred else fill)
    if average is None:
        return [np.array(v) for v in values]
    if average in ("binary", "micro"):
        return [v[0] for v in values]
    if average == "macro":
        return [math.fsum(v) / len(v) for v in values]
    support = sum(c[1] for c in counts)
    if not support:
        return [math.fsum(v) / len(v) for v in values]
    return [math.fsum(v * c[1] for v, c in zip(vals, counts)) / support
            for vals in values]


def fixtures():
    y = np.array([0, 0, 0, 1, 1, 2, 2, 2, 2, 1, 0], dtype=np.int64)
    p = np.array([0, 1, 2, 1, 2, 0, 0, 2, 1, 0, 2], dtype=np.int64)
    for name, mapping in [
        ("integer", np.array([0, 1, 2, 9], np.int64)),
        ("large-integer", np.array([2**60 + 1, 2**60 + 3, 2**60 + 5,
                                     2**60 + 7], np.int64)),
        ("string", np.array(["alpha", "beta", "gamma", "absent"])),
    ]:
        yield name, mapping[y], mapping[p], mapping[[2, 0, 3]]
    # Full output-size and tail exercise, with deliberately asymmetric errors.
    i = np.arange(513)
    yield "ragged", i % 5, (i * 3 + i // 7) % 5, np.array([4, 1, 9])
    yield "zero-divisions", np.zeros(7, np.int64), np.ones(7, np.int64), np.array([0, 1, 2])


def captured(fn, *args, **kwargs):
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        value = fn(*args, **kwargs)
    return value, bool(seen)


def fingerprint(value):
    array = np.asarray(value)
    return (array.dtype.str, array.shape, array.tobytes().hex())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", choices=tuple(CODES), default=list(CODES))
    parser.add_argument("--require-sklearn", action="store_true")
    args = parser.parse_args()
    if args.require_sklearn:
        assert sklearn is not None, "--require-sklearn needs sklearn 1.8"
    if sklearn is not None:
        assert sklearn.__version__.split(".")[:2] == ["1", "8"], sklearn.__version__
    version = sklearn.__version__ if sklearn is not None else "not installed; hand oracle"
    seen = {}
    checks = 0

    def compare(mode, key, value, expected, hand):
        nonlocal checks
        np.testing.assert_allclose(expected, hand, rtol=1e-14, atol=1e-14,
                                   err_msg=f"independent oracle {key}")
        np.testing.assert_allclose(value, expected, rtol=2e-6, atol=2e-7,
                                   err_msg=f"GPU {mode} {key}")
        stamp = fingerprint(value)
        identity = (mode, key)
        if identity in seen:
            assert seen[identity] == stamp, ("repeat bits", identity)
        seen[identity] = stamp
        checks += 1

    for mode in [*args.modes, *reversed(args.modes)]:
        binding = metrics._get_binding(mode)
        readback = binding.metrics_numeric_mode
        vendor = _backend.read_vendor(binding)
        assert vendor in ("metal", "cuda", "hip"), vendor
        assert readback() == CODES[mode], (mode, readback())
        for label, y, p, subset in fixtures():
            before_y, before_p = y.copy(), p.copy()
            for selected in (None, subset):
                labels = np.unique(np.concatenate([y, p])) if selected is None else selected
                key = f"{label}/{'all' if selected is None else 'subset'}"
                for normalize in (None, "true", "pred", "all"):
                    opts = dict(labels=selected, normalize=normalize)
                    actual = metrics.confusion_matrix(y, p, numeric_mode=mode, **opts)
                    hand = hand_confusion(y.tolist(), p.tolist(), labels.tolist(), normalize)
                    expected = reference.confusion_matrix(y, p, **opts) if reference else hand
                    if normalize is None:
                        np.testing.assert_array_equal(actual, expected)
                    compare(mode, f"{key}/confusion/{normalize}", actual, expected, hand)
                for average in (None, "micro", "macro", "weighted"):
                    for zero in (0, 1, "warn"):
                        hand = hand_scores(y.tolist(), p.tolist(), labels.tolist(), average, zero, 1)
                        for name, truth in zip(SCORES, hand):
                            opts = dict(labels=selected, average=average, zero_division=zero)
                            actual, warned = captured(getattr(metrics, name), y, p,
                                                       numeric_mode=mode, **opts)
                            expected = truth
                            if reference:
                                expected, ref_warned = captured(getattr(reference, name), y, p, **opts)
                                assert warned == ref_warned, (mode, key, name, average, zero, "warnings")
                            compare(mode, f"{key}/{name}/{average}/{zero}", actual, expected, truth)
            np.testing.assert_array_equal(y, before_y)
            np.testing.assert_array_equal(p, before_p)
        for label, y, p, pos in [
            ("binary-string", ["yes", "no", "yes", "no"], ["no", "no", "yes", "yes"], "yes"),
            ("binary-large", [2**60+1, 2**60+3], [2**60+3, 2**60+3], 2**60+1),
            ("absent-positive", [0, 0], [0, 0], 1),
        ]:
            for zero in (0, 1, "warn"):
                hand = hand_scores(y, p, [pos], "binary", zero, pos)
                for name, truth in zip(SCORES, hand):
                    opts = dict(pos_label=pos, zero_division=zero)
                    actual, warned = captured(getattr(metrics, name), y, p, numeric_mode=mode, **opts)
                    expected = truth
                    if reference:
                        expected, ref_warned = captured(getattr(reference, name), y, p, **opts)
                        assert warned == ref_warned, (mode, label, name, zero, "warnings")
                    compare(mode, f"{label}/{name}/{zero}", actual, expected, truth)
        # Zero selected true support still permits nonzero selected prediction
        # support. sklearn's weighted fallback averages the per-label results.
        y, p, labels = [1, 1], [0, 1], [0, 2]
        for zero in (0, 1, "warn"):
            hand = hand_scores(y, p, labels, "weighted", zero, 1)
            for name, truth in zip(SCORES, hand):
                opts = dict(labels=labels, average="weighted", zero_division=zero)
                actual, warned = captured(getattr(metrics, name), y, p, numeric_mode=mode, **opts)
                expected = truth
                if reference:
                    expected, ref_warned = captured(getattr(reference, name), y, p, **opts)
                    assert warned == ref_warned, (mode, name, zero, "zero-support warnings")
                compare(mode, f"zero-selected-support/{name}/{zero}", actual, expected, truth)
        # Atomic integer counts must be independent of row arrival order.
        index = np.arange(513)
        y, p = index % 5, (index * 3 + index // 7) % 5
        permutations = (index[::-1], (index * 17) % len(index))
        for name in ("confusion_matrix", *SCORES):
            options = ([dict(normalize=v) for v in (None, "true", "pred", "all")]
                       if name == "confusion_matrix" else
                       [dict(average=a, zero_division=0) for a in ("macro", "weighted")])
            for opts in options:
                baseline = getattr(metrics, name)(y, p, numeric_mode=mode, **opts)
                for order in permutations:
                    reordered = getattr(metrics, name)(y[order], p[order], numeric_mode=mode, **opts)
                    assert fingerprint(baseline) == fingerprint(reordered), (mode, name, opts, "row order")
                    checks += 1
        previous = _backend.default_mode()
        try:
            mojolearn.set_numeric_mode(mode)
            for name in SCORES:
                opts = dict(average="macro", zero_division=0)
                explicit = getattr(metrics, name)([0, 1, 1], [1, 1, 1], numeric_mode=mode, **opts)
                implicit = getattr(metrics, name)([0, 1, 1], [1, 1, 1], **opts)
                assert fingerprint(explicit) == fingerprint(implicit), (mode, name, "dynamic default")
                checks += 1
            explicit = metrics.confusion_matrix([0, 1], [1, 1], numeric_mode=mode)
            implicit = metrics.confusion_matrix([0, 1], [1, 1])
            assert fingerprint(explicit) == fingerprint(implicit), (mode, "confusion dynamic default")
            checks += 1
        finally:
            mojolearn.set_numeric_mode(previous)
        print(json.dumps({"mode": mode, "compiled_mode": readback(), "vendor": vendor,
                          "checks": checks, "sklearn": version, "status": "PASS"}), flush=True)
    print(json.dumps({"status": "PASS", "checks": checks,
                      "repeated_values": len(seen), "modes": args.modes,
                      "fingerprints": {"/".join(k): v for k, v in seen.items()}}), flush=True)


if __name__ == "__main__":
    main()
