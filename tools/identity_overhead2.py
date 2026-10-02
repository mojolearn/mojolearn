"""Identity digests for lane gap-neural-overhead2 (2026-10-02).

Runs the 15 neural board lanes' ops (cross-entropy, embedding, conv1d,
conv2d, max/avg pool 1d/2d, batchnorm 1d/2d, dropout2d, resnet-block, gcn,
graphsage, moe) on small fixed inputs and prints one `DIGEST <op> <sha16>`
line each. Every op runs THREE rounds on the same layer (the second and
third exercise the cached device copies, resident intermediates and pools),
and the digest covers every round's outputs, gradients and public array
attributes. Inputs come from integer arithmetic only (no platform libm), so
the digests are comparable across hosts and vendors: run it on the branch
and on main, default vendor and MOJOLEARN_VENDOR=cpu; all four must agree.

    cd python && python ../tools/identity_overhead2.py
"""
import hashlib
import sys

import numpy as np

import mojolearn as ml

ROUNDS = 3


def fx(shape, seed):
    """Floats in [-1, 1) from a multiplicative hash: exact in float32."""
    n = int(np.prod(shape))
    i = np.arange(n, dtype=np.uint64)
    h = (i * np.uint64(2654435761) + np.uint64(seed * 40503 + 12345)) & np.uint64(0xFFFFFFFF)
    v = (h >> np.uint64(8)).astype(np.float64) / float(1 << 23) - 1.0
    return v.astype(np.float32).reshape(shape)


def ix(shape, hi, seed):
    n = int(np.prod(shape))
    i = np.arange(n, dtype=np.uint64)
    h = (i * np.uint64(2246822519) + np.uint64(seed * 3266489917 + 7)) & np.uint64(0xFFFFFFFF)
    return (h % np.uint64(hi)).astype(np.int64).reshape(shape)


class D:
    def __init__(self):
        self.h = hashlib.sha256()

    def add(self, tag, a):
        if a is None:
            return
        if isinstance(a, tuple):
            for k, e in enumerate(a):
                self.add("%s.%d" % (tag, k), e)
            return
        a = np.ascontiguousarray(np.asarray(a))
        self.h.update(("%s|%s|%s|" % (tag, a.dtype.str, a.shape)).encode())
        self.h.update(a.tobytes())

    def attrs(self, tag, layer):
        for k in sorted(dir(layer)):
            if not k.endswith("_") or k.startswith("_"):
                continue
            try:
                v = getattr(layer, k)
            except Exception:  # noqa: BLE001  (an attribute only set after a backward)
                continue
            if isinstance(v, np.ndarray):
                self.add("%s.%s" % (tag, k), v)

    def hex(self):
        return self.h.hexdigest()[:16]


def layer_op(make, x, extra=(), backward=True, takes_x=False):
    d = D()
    layer = make()
    for r in range(ROUNDS):
        y = layer.forward(x, *extra)
        y = y[0] if isinstance(y, tuple) else y
        d.add("y%d" % r, y)
        if backward:
            dy = fx(np.shape(y), 900 + r)
            g = layer.backward(dy, x) if takes_x else layer.backward(dy)
            d.add("g%d" % r, g)
        d.attrs("a%d" % r, layer)
    return d.hex()


def op_cross_entropy():
    d = D()
    logits = fx((64, 257), 1) * 8
    t = ix((64,), 257, 2)
    t[5] = -100
    for r in range(ROUNDS):
        for red in ("mean", "sum", "none"):
            loss, dl = ml.cross_entropy(logits, t, reduction=red, return_grad=True)
            d.add("%d%s" % (r, red), (np.asarray(loss), dl))
        loss, dl = ml.cross_entropy(logits, t, label_smoothing=0.1, return_grad=True)
        d.add("%dls" % r, (np.asarray(loss), dl))
    return d.hex()


def op_embedding():
    d = D()
    e = ml.Embedding(300, 48, weight=fx((300, 48), 3))
    ids = ix((16, 9), 300, 4)
    for r in range(ROUNDS):
        y = e.forward(ids)
        d.add("y%d" % r, y)
        d.add("g%d" % r, e.backward(ids, fx(np.shape(y), 5 + r)))
    return d.hex()


def cls(name):
    return getattr(ml, name)


def op_moe():
    d = D()
    E, D_, F = 4, 32, 48
    m = cls("MoEBlock")(D_, F, num_experts=E, top_k=2)
    sd = {"router": fx((E, D_), 11), "gate_up_proj": fx((E, 2 * F, D_), 12) * 0.2,
          "down_proj": fx((E, D_, F), 13) * 0.2}
    m.load_state_dict(sd)
    x = fx((2, 24, D_), 14)
    for r in range(ROUNDS):
        y = m.forward(x)
        y = y[0] if isinstance(y, tuple) else y
        d.add("y%d" % r, y)
        d.attrs("a%d" % r, m)
    # a weight assignment must reach the next forward (the version bump)
    m.down_proj = fx((E, D_, F), 15) * 0.2
    y = m.forward(x)
    d.add("yv", y[0] if isinstance(y, tuple) else y)
    return d.hex()


def graph(n, k, seed):
    src = np.repeat(np.arange(n, dtype=np.int64), k)
    dst = ix((n * k,), n, seed)
    return np.stack([src, dst])


OPS = {
    "cross-entropy": op_cross_entropy,
    "embedding": op_embedding,
    "conv1d": lambda: layer_op(lambda: cls("Conv1d")(4, 6, 3, padding=1, random_state=1), fx((3, 4, 40), 21),
                               takes_x=True),
    "conv2d": lambda: layer_op(lambda: cls("Conv2d")(3, 5, 3, padding=1, random_state=2), fx((2, 3, 12, 12), 22),
                               takes_x=True),
    "maxpool2d": lambda: layer_op(lambda: cls("MaxPool2d")(2), fx((2, 3, 12, 12), 23)),
    "avgpool2d": lambda: layer_op(lambda: cls("AvgPool2d")(2), fx((2, 3, 12, 12), 24)),
    "maxpool1d": lambda: layer_op(lambda: cls("MaxPool1d")(2), fx((3, 4, 40), 25)),
    "avgpool1d": lambda: layer_op(lambda: cls("AvgPool1d")(2), fx((3, 4, 40), 26)),
    "batchnorm1d": lambda: layer_op(lambda: cls("BatchNorm1d")(6), fx((16, 6), 27)),
    "batchnorm2d": lambda: layer_op(lambda: cls("BatchNorm2d")(5), fx((4, 5, 6, 6), 28)),
    "dropout2d": lambda: layer_op(lambda: cls("Dropout2d")(0.5, random_state=3), fx((4, 8, 5, 5), 29)),
    "resnet-block": lambda: layer_op(lambda: cls("BasicBlock")(8, 8, random_state=4), fx((2, 8, 10, 10), 30)),
    "gcn": lambda: layer_op(lambda: cls("GCNConv")(12, 7, random_state=5), fx((50, 12), 31),
                            extra=(graph(50, 4, 32),)),
    "graphsage": lambda: layer_op(lambda: cls("SAGEConv")(12, 7, random_state=6), fx((50, 12), 33),
                                  extra=(graph(50, 4, 34),)),
    "moe": op_moe,
}


def main():
    want = sys.argv[1].split(",") if len(sys.argv) > 1 else list(OPS)
    bad = 0
    for name in want:
        try:
            print("DIGEST %s %s" % (name, OPS[name]()), flush=True)
        except Exception as e:  # noqa: BLE001
            bad += 1
            print("DIGEST %s ERROR %s: %s" % (name, type(e).__name__, str(e)[:200]), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
