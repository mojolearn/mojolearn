"""Small public GPU log-loss smoke; not a cross-vendor qualification gate."""
import math

import numpy as np

from mojolearn import metrics


def main():
    eps = float(np.finfo(np.float32).eps)
    cases = [
        ([0, 1], np.array([0.25, 0.75], np.float32), None, [0.75, 0.75]),
        ([0, 1], np.array([1, 0], np.float32), None, [eps, eps]),
        (["b", "a"], np.array([[0.75, 0.25], [0.5, 0.5]], np.float32),
         ["b", "a"], [0.75, 0.5]),
        ([2, 0, 1], np.array([[0.25, 0.25, 0.5], [0.5, 0.25, 0.25],
                             [0.25, 0.5, 0.25]], np.float32), None, [0.5] * 3),
        ([1] * 257, np.full((257, 1), 0.75, np.float32), [0, 1], [0.75] * 257),
    ]
    checks = 0
    for mode in ("fast", "deterministic", "identical"):
        for y, p, labels, selected in cases:
            total = math.fsum(-math.log(min(1 - eps, max(eps, x))) for x in selected)
            for normalize in (False, True):
                actual = metrics.log_loss(y, p, labels=labels, normalize=normalize,
                                          numeric_mode=mode)
                expected = total / len(y) if normalize else total
                assert math.isclose(actual, expected, rel_tol=3e-5, abs_tol=2e-6), (
                    mode, actual, expected)
                checks += 1
        print(f"PASS log_loss {mode}", flush=True)
    print(f"PASS {checks} public GPU log-loss smoke checks")


if __name__ == "__main__":
    main()
