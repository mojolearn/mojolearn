"""Small independent public GPU ranking smoke, not cross-vendor qualification."""
import math
import warnings

import numpy as np

from mojolearn import metrics


def reference(y, scores, positive):
    labels = [label == positive for label in y]
    values = [float(value) for value in scores]
    thresholds = sorted(set(values))
    positives = sum(labels)
    precision, recall = [], []
    for threshold in thresholds:
        selected = [i for i, value in enumerate(values) if value >= threshold]
        tp = sum(labels[i] for i in selected)
        precision.append(tp / len(selected))
        recall.append(tp / positives if positives else 1.0)
    precision.append(1.0)
    recall.append(0.0)
    auc = None
    if 0 < positives < len(labels):
        # Pairwise win/tie oracle is independent of the native sort/scan algorithm.
        credit = sum(
            1.0 if a > b else 0.5 if a == b else 0.0
            for a, yes in zip(values, labels) if yes
            for b, no in zip(values, labels) if not no
        )
        auc = credit / (positives * (len(labels) - positives))
    return auc, precision, recall, thresholds


def main():
    tiny = np.nextafter(np.float32(0), np.float32(1))
    cases = [
        ([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8], 1),
        ([0, 1, 0, 1], [2, 2, 2, 2], 1),
        ([0, 1, 0, 1, 0, 1], [-0.0, 0.0, -tiny, tiny, -2, 2], 1),
        (["a", "z", "a", "z"], [-4, -3, -2, -1], "z"),
        ([2**62, 2**62 + 1, 2**62 + 1], [3, 2, 1], 2**62 + 1),
        ([i % 2 for i in range(513)], [(i * 37) % 19 - 9 for i in range(513)], 1),
        ([1, 1, 1], [0, 1, 2], 1),
        ([0, 0, 0], [0, 1, 2], 1),
    ]
    checks = 0
    for mode in ("fast", "deterministic", "identical"):
        for y, values, positive in cases:
            scores = np.asarray(values, dtype=np.float32)
            auc, p, r, t = reference(y, scores, positive)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                actual = metrics.precision_recall_curve(
                    y, scores, pos_label=positive, numeric_mode=mode
                )
            for observed, expected in zip(actual[:2], (p, r)):
                np.testing.assert_allclose(observed, expected, rtol=3e-6, atol=1e-7)
                checks += 1
            # Equality of values (including subnormals) matters; zero sign does not.
            np.testing.assert_array_equal(actual[2], np.asarray(t, np.float32))
            checks += 1
            if auc is not None:
                observed = metrics.roc_auc_score(y, scores, numeric_mode=mode)
                assert math.isclose(observed, auc, rel_tol=3e-6, abs_tol=1e-7), (
                    mode, y, observed, auc
                )
                checks += 1
        print(f"PASS binary ranking {mode}", flush=True)
    print(f"PASS {checks} public GPU binary-ranking smoke checks")


if __name__ == "__main__":
    main()
