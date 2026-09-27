# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Every x_cnn entry, called again and again in ONE process (the directive of
2026-09-27: a binding that builds a DeviceContext per call hangs on its second
GPU call, and the lane checks call each entry once). Since DEVIATION 5718 the
GPU binding also reuses cached workspace buffers across calls and hands out
resident arrays, so each entry runs at a LARGE shape, a SMALL one and the large
one again (the workspace grows, is reused smaller, is reused again) and must
return the same bits for the same input every time; the resident (`_r`)
entries must return the address entries' bits. Runs on whichever column the
process loads (the GPU binding, or the CPU twin under MOJOLEARN_VENDOR=cpu)."""
import numpy as np


def _x(shape, seed):
    return np.random.default_rng(seed).standard_normal(shape).astype(np.float32)


def _same(a, b):
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def _thrice(run, big, small):
    """run(size) -> tuple of arrays; big, small, big again: the two big calls agree."""
    first = run(big)
    run(small)
    again = run(big)
    for a, b in zip(first, again):
        _same(a, b)
    return first


def test_layers_repeat():
    import mojolearn as ml

    def conv(n):
        c = ml.Conv2d(5, 7, 3, padding=1, random_state=1)
        x = _x((n, 5, 12, 12), n)
        y = c.forward(x)
        dx = c.backward(_x(y.shape, n + 1))
        return y, dx, c.grad_weight_, c.grad_bias_
    _thrice(conv, 24, 2)

    def pools(n):
        x = _x((n, 3, 8, 8), n)
        out = []
        for layer in (ml.MaxPool2d(2), ml.AvgPool2d(3, stride=2, padding=1)):
            y = layer.forward(x)
            out += [y, layer.backward(_x(y.shape, 3))]
        return tuple(out)
    _thrice(pools, 40, 1)

    def bn_blk(n):
        x = _x((n, 4, 6, 6), n)
        bn = ml.BatchNorm2d(4)
        y = bn.forward(x)
        blk = ml.BasicBlock(4, 6, stride=2, random_state=2)
        z = blk.forward(x)
        return y, bn.backward(_x(y.shape, 5)), z, blk.backward(_x(z.shape, 6))
    _thrice(bn_blk, 32, 2)

    def graph(n):
        rng = np.random.default_rng(n)
        e = rng.integers(0, n, (2, 4 * n))
        h = _x((n, 5), n)
        out = []
        for layer in (ml.GCNConv(5, 3, random_state=0), ml.SAGEConv(5, 3, random_state=0)):
            y = layer.forward(h, e)
            out += [y, layer.backward(_x(y.shape, 9))]
        return tuple(out)
    _thrice(graph, 50, 3)


def _fit(n, **kw):
    import mojolearn as ml
    X = _x((n, 2 * 6 * 6), n)
    y = (X[:, :36].sum(1) > 0).astype(int) + (X[:, 36:].sum(1) > 0).astype(int)
    m = ml.CNNClassifier((2, 6, 6), conv_channels=(3, 4), batch_size=32, max_iter=2, random_state=0, **kw).fit(X, y)
    return (np.asarray(m.losses_), *m.weights(), m.predict_proba(X))


def test_trainer_repeat():
    for kw in (dict(), dict(optimizer="adam", learning_rate=0.01)):
        _thrice(lambda n: _fit(n, **kw), 100, 7)


def test_resident_entries_match_address_entries():
    """The `_r` entries on resident arrays give the address entries' bits,
    each called twice."""
    import mojolearn as ml
    from mojolearn._expansion_cnn import _Res
    conv = ml.Conv2d(3, 4, 3, padding=1, random_state=3)
    pool = ml.MaxPool2d(2)
    b = conv._binding()
    x = _x((6, 3, 8, 8), 1)
    prm = conv._params(x.shape)
    cshape = conv._out_shape(x.shape)
    pprm = pool._params(cshape)
    oshape = pool._out_shape(cshape)
    out = np.empty(oshape, np.float32)
    idx = np.empty(oshape, np.int32)
    b.x_cnn_conv_block_forward(x.ctypes.data, conv.weight_.ctypes.data, conv.bias_.ctypes.data, out.ctypes.data,
                               idx.ctypes.data, prm, pprm)
    g = _x(oshape, 2)
    dx, dw, db = np.empty(x.shape, np.float32), np.empty(conv.weight_.shape, np.float32), np.empty(4, np.float32)
    b.x_cnn_conv_block_backward(x.ctypes.data, conv.weight_.ctypes.data, conv.bias_.ctypes.data, g.ctypes.data,
                                idx.ctypes.data, [dx.ctypes.data, dw.ctypes.data, db.ctypes.data], prm, pprm)
    for _ in (0, 1):
        with _Res(b) as R:
            h = {k: R.new(v.size) for k, v in dict(x=x, w=conv.weight_, b=conv.bias_, out=out, idx=idx, g=g, dx=dx,
                                                  dw=dw, db=db).items()}
            for k, v in dict(x=x, w=conv.weight_, b=conv.bias_, g=g).items():
                R.put(h[k], v)
            cols, yconv = R.new(6 * 8 * 8 * 27), R.new(6 * 4 * 8 * 8)
            sv = [cols, yconv] if _ else []  # the second round with the saved arrays
            b.x_cnn_conv_block_forward_r(h["x"], h["w"], h["b"], h["out"], h["idx"], prm, pprm, sv)
            b.x_cnn_conv_block_backward_r(h["x"], h["w"], h["b"], h["g"], h["idx"], [h["dx"], h["dw"], h["db"]] + sv,
                                          prm, pprm)
            _same(R.get(h["out"], out.shape), out)
            _same(R.get(h["idx"], idx.shape).view(np.int32), idx)
            _same(R.get(h["dx"], dx.shape), dx)
            _same(R.get(h["dw"], dw.shape), dw)
            _same(R.get(h["db"], db.shape), db)
        rows = np.array([5, 0, 3], np.int32)
        with _Res(b) as R:
            src, dst = R.new(x.size), R.new(3 * 192)
            R.put(src, x)
            b.x_cnn_res_gather(dst, src, rows.ctypes.data, [3, 192])
            _same(R.get(dst, (3, 3, 8, 8)), x[rows])
