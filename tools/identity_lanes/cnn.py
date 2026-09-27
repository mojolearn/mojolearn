# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE CNN LANE'S IDENTITY LANES (docs/lanes/ALGORITHM_EXPANSION_BRIEFS.md).
#
# Owned by the `cnn` expansion lane. tools/identity_break.py executes this
# file in ITS OWN namespace after every helper and registry exists
# (`_load_lane_fragments`), so a lane here is written exactly like a lane there:
#
#     @lane("cnn-example")
#     def _(ml, X, yc, yr, Xh=None):
#         m = ml.Example().fit(X)
#         return _fit(dict(out=_h(m.transform(X[:256]))), m, lambda e: (e.transform(Xh[:256]),))
#
#     _batch_decl(_rows_calls("transform", sl=slice(0, 256)), "cnn-example")
#
# Rules the loader enforces: spell every lane name literally; add only your
# own lanes (to LANES and the per-lane registries); rebind no existing name;
# prefix your own helpers with `_cnn_`. No imports are needed: np, _h,
# _fit, _rows_calls and the rest are this module's.


def _cnn_grad(X, shape):
    """A deterministic upstream gradient of `shape` from the fixture's own
    later rows (plain copies, no host arithmetic)."""
    flat = np.ascontiguousarray(X[256:, :16]).ravel()
    size = int(np.prod(shape))
    reps = -(-size // flat.size)
    return np.ascontiguousarray(np.tile(flat, reps)[:size].reshape(shape))


def _cnn_fwd_bwd(conv, x, X):
    out = conv.forward(x)
    dx = conv.backward(_cnn_grad(X, out.shape))
    return out, dx, conv.grad_weight_, conv.grad_bias_


@lane("x-cnn-conv2d")
def _(ml, X, yc, yr, Xh=None):
    """Conv2d forward + backward on 256 rows as (1, 4, 4) images: a 3x3
    padded conv, and a strided, dilated, rectangular one."""
    x = np.ascontiguousarray(X[:256, :16]).reshape(256, 1, 4, 4)
    a = ml.Conv2d(1, 4, 3, padding=1, random_state=3, input_shape=(1, 4, 4))
    b = ml.Conv2d(1, 3, (3, 2), stride=(2, 1), padding=(2, 1), dilation=(2, 1), random_state=4,
                  input_shape=(1, 4, 4))
    pa = _cnn_fwd_bwd(a, x, X)
    pb = _cnn_fwd_bwd(b, x, X)
    return _fit(dict(a_out=_h(pa[0]), a_dx=_h(pa[1]), a_dw=_h(pa[2]), a_db=_h(pa[3]),
                     b_out=_h(pb[0]), b_dx=_h(pb[1]), b_dw=_h(pb[2]), b_db=_h(pb[3])),
                a, lambda e: (e.transform(np.ascontiguousarray(Xh[:256, :16])),))


@lane("x-cnn-conv1d")
def _(ml, X, yc, yr, Xh=None):
    """Conv1d forward + backward on 256 rows as (2, 8) sequences: strided
    padded, and dilated without a bias."""
    x = np.ascontiguousarray(X[:256, :16]).reshape(256, 2, 8)
    a = ml.Conv1d(2, 3, 3, stride=2, padding=1, random_state=5, input_shape=(2, 8))
    b = ml.Conv1d(2, 4, 2, padding=2, dilation=3, bias=False, random_state=6, input_shape=(2, 8))
    pa = _cnn_fwd_bwd(a, x, X)
    pb = _cnn_fwd_bwd(b, x, X)
    return _fit(dict(a_out=_h(pa[0]), a_dx=_h(pa[1]), a_dw=_h(pa[2]), a_db=_h(pa[3]),
                     b_out=_h(pb[0]), b_dx=_h(pb[1]), b_dw=_h(pb[2])),
                a, lambda e: (e.transform(np.ascontiguousarray(Xh[:256, :16])),))


_batch_decl(_rows_calls("transform", sl=np.s_[:256, :16]), "x-cnn-conv2d", "x-cnn-conv1d")


@lane("x-cnn-pool")
def _(ml, X, yc, yr, Xh=None):
    """MaxPool and AvgPool forward + backward on 256 rows as (2, 2, 4)
    images and (2, 8) sequences: overlapping windows (the gathers sum more
    than one window), padding, dilation, count_include_pad both ways. The
    `ties` fixture puts exact ties inside the max windows."""
    x = np.ascontiguousarray(X[:256, :16]).reshape(256, 2, 2, 4)
    s = np.ascontiguousarray(X[:256, :16]).reshape(256, 2, 8)
    parts = {}
    layers = dict(
        mx=(ml.MaxPool2d((2, 3), stride=1, padding=(1, 1), input_shape=(2, 2, 4)), x),
        mxd=(ml.MaxPool2d(2, stride=1, padding=1, dilation=(1, 2)), x),
        av=(ml.AvgPool2d(3, stride=1, padding=1), x),
        avx=(ml.AvgPool2d((2, 3), stride=(1, 2), padding=(1, 1), count_include_pad=False), x),
        mx1=(ml.MaxPool1d(3, stride=2, padding=1), s),
        av1=(ml.AvgPool1d(4, stride=2, padding=2, count_include_pad=False), s),
    )
    for k, (layer, inp) in layers.items():
        out = layer.forward(inp)
        parts[k + "_out"] = _h(out)
        parts[k + "_dx"] = _h(layer.backward(_cnn_grad(X, out.shape)))
        if hasattr(layer, "indices_"):
            parts[k + "_idx"] = _h(layer.indices_)
    first = layers["mx"][0]
    return _fit(parts, first, lambda e: (e.transform(np.ascontiguousarray(Xh[:256, :16])),))


_batch_decl(_rows_calls("transform", sl=np.s_[:256, :16]), "x-cnn-pool")


@lane("x-cnn-trainer")
def _(ml, X, yc, yr, Xh=None):
    """CNNClassifier: two SGD epochs (momentum, weight decay) of 64-row
    batches over 256 rows as (1, 4, 4) images, one conv block (conv, relu,
    max pool) and the linear head; the batch losses, every trained array
    and the probabilities on the training rows."""
    x = np.ascontiguousarray(X[:256, :16])
    m = ml.CNNClassifier(input_shape=(1, 4, 4), conv_channels=(4,), kernel_size=3, pool_size=2,
                         learning_rate=0.05, momentum=0.9, weight_decay=1e-4, batch_size=64, max_iter=2,
                         random_state=0).fit(x, yc[:256])
    return _fit(dict(loss=_h(np.asarray(m.losses_, dtype=np.float64)), weights=_h(*m.weights()),
                     proba=_h(m.predict_proba(x))),
                m, lambda e: (e.predict_proba(np.ascontiguousarray(Xh[:256, :16])),))


_batch_decl(_rows_calls("predict_proba", sl=np.s_[:256, :16]), "x-cnn-trainer")


@lane("x-cnn-batchnorm")
def _(ml, X, yc, yr, Xh=None):
    """BatchNorm2d over 256 rows as (4, 2, 2) maps: two training passes
    (batch statistics, running updates) with backward, then eval mode with
    backward; BatchNorm1d over (N, C). The `dupes` fixture's constant and
    all-zero columns give zero-variance channels."""
    x = np.ascontiguousarray(X[:256, :16]).reshape(256, 4, 2, 2)
    bn = ml.BatchNorm2d(4, momentum=0.3, input_shape=(4, 2, 2))
    bn.weight_ = np.ascontiguousarray((np.arange(4, dtype=np.float32) - 1.5) / np.float32(2))
    bn.bias_ = np.ascontiguousarray(np.arange(4, dtype=np.float32) / np.float32(4))
    parts = {}
    for k in range(2):
        y = bn.forward(x[128 * k:128 * (k + 1)])
        parts[f"t{k}_y"] = _h(y)
        parts[f"t{k}_dx"] = _h(bn.backward(_cnn_grad(X, y.shape)), bn.grad_weight_, bn.grad_bias_)
    parts["running"] = _h(bn.running_mean_, bn.running_var_)
    bn.eval()
    y = bn.forward(x)
    parts["e_y"] = _h(y)
    parts["e_dx"] = _h(bn.backward(_cnn_grad(X, y.shape)), bn.grad_weight_, bn.grad_bias_)
    b1 = ml.BatchNorm1d(16)
    y1 = b1.forward(np.ascontiguousarray(X[:256, :16]))
    parts["b1_y"] = _h(y1, b1.running_mean_, b1.running_var_)
    parts["b1_dx"] = _h(b1.backward(_cnn_grad(X, y1.shape)), b1.grad_weight_, b1.grad_bias_)
    return _fit(parts, bn, lambda e: (e.transform(np.ascontiguousarray(Xh[:256, :16])),))


_batch_decl(_rows_calls("transform", sl=np.s_[:256, :16]), "x-cnn-batchnorm")


@lane("x-cnn-dropout2d")
def _(ml, X, yc, yr, Xh=None):
    """Dropout2d (p = 0.3) on 256 rows as (4, 2, 2) maps, two calls (two
    masks) with backward, then eval mode."""
    x = np.ascontiguousarray(X[:256, :16]).reshape(256, 4, 2, 2)
    d = ml.Dropout2d(p=0.3, random_state=11, input_shape=(4, 2, 2))
    parts = {}
    for k in range(2):
        y = d.forward(x)
        parts[f"y{k}"] = _h(y, d.mask_)
        parts[f"dx{k}"] = _h(d.backward(_cnn_grad(X, y.shape)))
    d.eval()
    parts["eval"] = _h(d.forward(x))
    d.train()
    d.calls_ = 0
    return _fit(parts, d, lambda e: (e.transform(np.ascontiguousarray(Xh[:256, :16])),))


@lane("x-cnn-globalpool")
def _(ml, X, yc, yr, Xh=None):
    """Global average and max pooling (AdaptiveAvgPool2d(1),
    AdaptiveMaxPool2d(1)) and a (2, 2) adaptive average, forward + backward,
    on 256 rows as (1, 4, 4) maps."""
    x = np.ascontiguousarray(X[:256, :16]).reshape(256, 1, 4, 4)
    parts = {}
    for k, layer in dict(gavg=ml.AdaptiveAvgPool2d(1, input_shape=(1, 4, 4)), gmax=ml.AdaptiveMaxPool2d(1),
                         avg22=ml.AdaptiveAvgPool2d(2)).items():
        y = layer.forward(x)
        parts[k] = _h(y, layer.backward(_cnn_grad(X, y.shape)))
    first = ml.AdaptiveAvgPool2d(1, input_shape=(1, 4, 4))
    return _fit(parts, first, lambda e: (e.transform(np.ascontiguousarray(Xh[:256, :16])),))


@lane("x-cnn-resnet-block")
def _(ml, X, yc, yr, Xh=None):
    """torchvision's BasicBlock, forward + backward in training mode, on 256
    rows as (1, 4, 4) maps: a downsampling block (1 -> 3 planes, stride 2,
    the conv1x1 + BN shortcut) then an identity block (3 -> 3)."""
    x = np.ascontiguousarray(X[:256, :16]).reshape(256, 1, 4, 4)
    b1 = ml.BasicBlock(1, 3, stride=2, random_state=21, input_shape=(1, 4, 4))
    b2 = ml.BasicBlock(3, 3, random_state=23)
    y = b2.forward(b1.forward(x))
    dx = b1.backward(b2.backward(_cnn_grad(X, y.shape)))
    grads = [getattr(l, g) for blk in (b1, b2) for l, _, g in blk.parameters()]
    running = [a for blk in (b1, b2) for bn in blk._bns() for a in (bn.running_mean_, bn.running_var_)]
    b1.eval()
    return _fit(dict(y=_h(y), dx=_h(dx), grads=_h(*grads), running=_h(*running)),
                b1, lambda e: (e.transform(np.ascontiguousarray(Xh[:256, :16])),))


_batch_decl(_rows_calls("transform", sl=np.s_[:256, :16]), "x-cnn-globalpool", "x-cnn-resnet-block")
