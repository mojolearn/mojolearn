"""lane/cnn-apple (2026-09-28): the FAST tier's paired quality set, run at the
before and the after commit on the same Mac.

5 seeds x 2 seeded image datasets (CIFAR-shaped 3x32x32, 10 classes):
  'blobs'   class = a per-class template + Gaussian noise (sigma 1.5)
  'stripes' class = the orientation/frequency of a stripe pattern + noise
CNNClassifier (32,64), batch 128, 3 epochs, 2048 train / 1024 test rows:
test accuracy, test log loss, and a digest of the probabilities.
5 seeds x 2 Conv2d shapes: max relative error of the forward against a
float64 NumPy reference (the PyTorch semantics), and a digest of y.
Lines: XCNN-Q <kind> <dataset|shape> seed <s> ... digest <hex>.
usage: python fastq.py <mode> <repo root>"""
import hashlib
import sys

import numpy as np

sys.path.insert(0, sys.argv[2] + "/python")
import mojolearn as ml  # noqa: E402

mode = sys.argv[1]
if "legacy" in __import__("os").environ.get("CNN_FASTQ_EXTRA", ""):  # lane/cnn-apple2 before arm
    from mojolearn import _expansion_cnn as _xc
    _xc._LEGACY_STEP = True


def digest(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def data(kind, seed, n):
    rng = np.random.default_rng(1000 + seed)
    y = rng.integers(0, 10, n)
    if kind == "blobs":
        t = np.random.default_rng(7).standard_normal((10, 3, 32, 32)).astype(np.float32)
        x = t[y] + 1.5 * rng.standard_normal((n, 3, 32, 32)).astype(np.float32)
    else:
        hh, ww = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
        ang = (y % 5) * np.pi / 5
        freq = 0.3 + 0.25 * (y // 5)
        ph = rng.uniform(0, 2 * np.pi, n)
        pat = np.sin(freq[:, None, None] * (np.cos(ang)[:, None, None] * hh + np.sin(ang)[:, None, None] * ww)
                     + ph[:, None, None])
        x = np.repeat(pat[:, None], 3, axis=1) + 1.0 * rng.standard_normal((n, 3, 32, 32))
    return x.reshape(n, -1).astype(np.float32), y


for kind in ("blobs", "stripes"):
    for seed in range(5):
        X, y = data(kind, seed, 3072)
        clf = ml.CNNClassifier((3, 32, 32), conv_channels=(32, 64), batch_size=128, max_iter=3,
                               learning_rate=0.01, random_state=seed, numeric_mode=mode)
        clf.fit(X[:2048], y[:2048])
        p = clf.predict_proba(X[2048:])
        acc = float((clf.classes_[p.argmax(1)] == y[2048:]).mean())
        ll = float(-np.log(np.clip(p[np.arange(1024), y[2048:]], 1e-12, 1)).mean())
        print(f"XCNN-Q fit {kind} seed {seed} acc {acc:.4f} logloss {ll:.4f} digest {digest(p)}", flush=True)


def conv_ref(x, w, b):
    x = x.astype(np.float64)
    n, c, h, wd = x.shape
    oc, _, kh, kw = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)))
    cols = np.stack([xp[:, :, i:i + h, j:j + wd] for i in range(kh) for j in range(kw)], axis=2)
    cols = cols.reshape(n, c * kh * kw, h * wd)
    y = np.einsum("ok,nkp->nop", w.reshape(oc, -1).astype(np.float64), cols) + b.astype(np.float64)[None, :, None]
    return y.reshape(n, oc, h, wd)


for (C, OC, H) in [(3, 64, 32), (64, 64, 16)]:
    for seed in range(5):
        rng = np.random.default_rng(seed)
        x = rng.standard_normal((64, C, H, H)).astype(np.float32)
        conv = ml.Conv2d(C, OC, 3, padding=1, random_state=seed, numeric_mode=mode)
        y = conv.forward(x)
        ref = conv_ref(x, conv.weight_, conv.bias_)
        err = float(np.max(np.abs(y - ref)) / np.max(np.abs(ref)))
        print(f"XCNN-Q conv C{C} OC{OC} H{H} seed {seed} maxrelerr {err:.3e} digest {digest(y)}", flush=True)
