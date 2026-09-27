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
