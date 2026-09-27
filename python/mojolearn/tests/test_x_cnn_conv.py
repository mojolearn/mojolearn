# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Correctness sanity of the CNN lane's convolutions against a float64 NumPy
reference of PyTorch's nn.Conv2d / nn.Conv1d semantics (tolerance, not
identity; identity is tools/algos_lane_check.sh x-cnn-conv2d,x-cnn-conv1d).
The reference functions here are also checked against torch itself where
torch is installed (`test_reference_matches_torch`)."""
import numpy as np
import pytest


def ref_conv2d(x, w, b, stride, padding, dilation):
    x = x.astype(np.float64)
    w = w.astype(np.float64)
    n, c, h, wd = x.shape
    oc, _, kh, kw = w.shape
    sh, sw = stride
    ph, pw = padding
    dh, dw = dilation
    xp = np.zeros((n, c, h + 2 * ph, wd + 2 * pw))
    xp[:, :, ph:ph + h, pw:pw + wd] = x
    oh = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ow = (wd + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    cols = np.empty((n, oh, ow, c, kh, kw))
    for i in range(kh):
        for j in range(kw):
            cols[:, :, :, :, i, j] = xp[:, :, i * dh:i * dh + sh * (oh - 1) + 1:sh,
                                        j * dw:j * dw + sw * (ow - 1) + 1:sw].transpose(0, 2, 3, 1)
    out = cols.reshape(n * oh * ow, -1) @ w.reshape(oc, -1).T
    if b is not None:
        out = out + b.astype(np.float64)
    return out.reshape(n, oh, ow, oc).transpose(0, 3, 1, 2), cols, xp.shape


def ref_conv2d_backward(x, w, g, stride, padding, dilation):
    n, c, h, wd = x.shape
    oc, _, kh, kw = w.shape
    _, cols, pshape = ref_conv2d(x, w, None, stride, padding, dilation)
    oh, ow = g.shape[2], g.shape[3]
    G = g.astype(np.float64).transpose(0, 2, 3, 1).reshape(-1, oc)
    dw = (G.T @ cols.reshape(G.shape[0], -1)).reshape(w.shape)
    db = G.sum(0)
    dcols = (G @ w.astype(np.float64).reshape(oc, -1)).reshape(n, oh, ow, c, kh, kw)
    dxp = np.zeros(pshape)
    sh, sw = stride
    dh, dwl = dilation
    for i in range(kh):
        for j in range(kw):
            dxp[:, :, i * dh:i * dh + sh * (oh - 1) + 1:sh, j * dwl:j * dwl + sw * (ow - 1) + 1:sw] += \
                dcols[:, :, :, :, i, j].transpose(0, 3, 1, 2)
    ph, pw = padding
    return dxp[:, :, ph:ph + h, pw:pw + wd], dw, db


CASES2D = [
    dict(n=3, c=2, h=7, w=6, oc=4, k=(3, 3), s=(1, 1), p=(1, 1), d=(1, 1), bias=True),
    dict(n=2, c=3, h=9, w=8, oc=5, k=(3, 2), s=(2, 1), p=(2, 1), d=(2, 1), bias=True),
    dict(n=4, c=1, h=5, w=5, oc=2, k=(2, 2), s=(2, 2), p=(0, 0), d=(1, 1), bias=False),
    dict(n=1, c=4, h=6, w=10, oc=3, k=(1, 3), s=(1, 3), p=(0, 2), d=(1, 2), bias=True),
]


def _case_arrays(cs, seed):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((cs["n"], cs["c"], cs["h"], cs["w"])).astype(np.float32)
    w = (rng.standard_normal((cs["oc"], cs["c"]) + cs["k"]) * 0.3).astype(np.float32)
    b = rng.standard_normal(cs["oc"]).astype(np.float32) if cs["bias"] else None
    return x, w, b


@pytest.mark.parametrize("cs", CASES2D)
def test_conv2d_forward_backward(cs):
    import mojolearn as ml
    x, w, b = _case_arrays(cs, 0)
    conv = ml.Conv2d(cs["c"], cs["oc"], cs["k"], stride=cs["s"], padding=cs["p"], dilation=cs["d"],
                     bias=cs["bias"]).set_weights(w, b)
    out = conv.forward(x)
    ref, _, _ = ref_conv2d(x, w, b, cs["s"], cs["p"], cs["d"])
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)
    g = np.random.default_rng(1).standard_normal(out.shape).astype(np.float32)
    dx = conv.backward(g)
    rdx, rdw, rdb = ref_conv2d_backward(x, w, g, cs["s"], cs["p"], cs["d"])
    np.testing.assert_allclose(dx, rdx, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(conv.grad_weight_, rdw, rtol=1e-5, atol=1e-5)
    if cs["bias"]:
        np.testing.assert_allclose(conv.grad_bias_, rdb, rtol=1e-5, atol=1e-5)


def test_conv1d_forward_backward():
    import mojolearn as ml
    rng = np.random.default_rng(2)
    x = rng.standard_normal((3, 2, 11)).astype(np.float32)
    w = rng.standard_normal((4, 2, 3)).astype(np.float32)
    b = rng.standard_normal(4).astype(np.float32)
    conv = ml.Conv1d(2, 4, 3, stride=2, padding=2, dilation=2).set_weights(w, b)
    out = conv.forward(x)
    ref, _, _ = ref_conv2d(x[:, :, None, :], w[:, :, None, :], b, (1, 2), (0, 2), (1, 2))
    np.testing.assert_allclose(out, ref[:, :, 0, :], rtol=1e-5, atol=1e-5)
    g = rng.standard_normal(out.shape).astype(np.float32)
    dx = conv.backward(g)
    rdx, rdw, rdb = ref_conv2d_backward(x[:, :, None, :], w[:, :, None, :], g[:, :, None, :],
                                        (1, 2), (0, 2), (1, 2))
    np.testing.assert_allclose(dx, rdx[:, :, 0, :], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(conv.grad_weight_, rdw[:, :, 0, :], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(conv.grad_bias_, rdb, rtol=1e-5, atol=1e-5)


def test_reference_matches_torch():
    torch = pytest.importorskip("torch")
    F = torch.nn.functional
    for cs in CASES2D:
        x, w, b = _case_arrays(cs, 0)
        tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        tw = torch.tensor(w, dtype=torch.float64, requires_grad=True)
        tb = None if b is None else torch.tensor(b, dtype=torch.float64, requires_grad=True)
        out = F.conv2d(tx, tw, tb, stride=cs["s"], padding=cs["p"], dilation=cs["d"])
        ref, _, _ = ref_conv2d(x, w, b, cs["s"], cs["p"], cs["d"])
        np.testing.assert_allclose(out.detach().numpy(), ref, rtol=1e-10, atol=1e-10)
        g = np.random.default_rng(1).standard_normal(ref.shape).astype(np.float32)
        out.backward(torch.tensor(g, dtype=torch.float64))
        rdx, rdw, rdb = ref_conv2d_backward(x, w, g, cs["s"], cs["p"], cs["d"])
        np.testing.assert_allclose(tx.grad.numpy(), rdx, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(tw.grad.numpy(), rdw, rtol=1e-10, atol=1e-10)
        if tb is not None:
            np.testing.assert_allclose(tb.grad.numpy(), rdb, rtol=1e-10, atol=1e-10)


_NP_MODE = {"zeros": "constant", "reflect": "reflect", "replicate": "edge", "circular": "wrap"}


def ref_conv_options(x, w, b, stride, pads, dilation, groups, mode, g):
    """(out, dx, dw, db) of an explicitly padded (top, bottom, left, right),
    grouped conv, float64; the pad's backward sums every padded position
    back onto its source pixel."""
    n, c, h, wd = x.shape
    t, bo, l, r = pads
    kw = dict(constant_values=-1) if mode == "zeros" else {}
    idx_h = np.pad(np.arange(h), (t, bo), mode=_NP_MODE[mode], **kw)
    idx_w = np.pad(np.arange(wd), (l, r), mode=_NP_MODE[mode], **kw)
    xp = np.zeros((n, c, len(idx_h), len(idx_w)))
    for i, a in enumerate(idx_h):
        for j, bb in enumerate(idx_w):
            if a >= 0 and bb >= 0:
                xp[:, :, i, j] = x[:, :, a, bb]
    cg, og = c // groups, w.shape[0] // groups
    outs, dws, dxps = [], [], np.zeros(xp.shape)
    for k in range(groups):
        o, _, _ = ref_conv2d(xp[:, k * cg:(k + 1) * cg], w[k * og:(k + 1) * og], None, stride, (0, 0), dilation)
        outs.append(o)
        dxg, dwg, _ = ref_conv2d_backward(xp[:, k * cg:(k + 1) * cg], w[k * og:(k + 1) * og], g[:, k * og:(k + 1) * og],
                                          stride, (0, 0), dilation)
        dxps[:, k * cg:(k + 1) * cg] = dxg
        dws.append(dwg)
    out = np.concatenate(outs, 1) + (0 if b is None else b.astype(np.float64).reshape(1, -1, 1, 1))
    dx = np.zeros(x.shape)
    for i, a in enumerate(idx_h):
        for j, bb in enumerate(idx_w):
            if a >= 0 and bb >= 0:
                dx[:, :, a, bb] += dxps[:, :, i, j]
    return out, dx, np.concatenate(dws, 0), g.astype(np.float64).sum((0, 2, 3))


OPTION_CASES = [
    dict(mode="reflect", padding=(2, 1), groups=1, k=(3, 3), s=(1, 1), d=(1, 1), c=2, oc=3),
    dict(mode="replicate", padding=(1, 2), groups=2, k=(3, 2), s=(2, 1), d=(1, 1), c=4, oc=6),
    dict(mode="circular", padding=(1, 1), groups=1, k=(3, 3), s=(1, 1), d=(2, 1), c=2, oc=2),
    dict(mode="zeros", padding="same", groups=4, k=(4, 3), s=(1, 1), d=(1, 2), c=4, oc=8),
    dict(mode="reflect", padding="same", groups=1, k=(2, 2), s=(1, 1), d=(1, 1), c=3, oc=2),
    dict(mode="zeros", padding="valid", groups=3, k=(2, 2), s=(1, 1), d=(1, 1), c=3, oc=3),
]


def _pads(cs):
    kh, kw = cs["k"]
    if cs["padding"] == "same":
        th, tw = cs["d"][0] * (kh - 1), cs["d"][1] * (kw - 1)
        return (th // 2, th - th // 2, tw // 2, tw - tw // 2)
    if cs["padding"] == "valid":
        return (0, 0, 0, 0)
    ph, pw = cs["padding"]
    return (ph, ph, pw, pw)


@pytest.mark.parametrize("cs", OPTION_CASES)
def test_conv2d_options(cs):
    import mojolearn as ml
    rng = np.random.default_rng(7)
    x = rng.standard_normal((2, cs["c"], 7, 8)).astype(np.float32)
    conv = ml.Conv2d(cs["c"], cs["oc"], cs["k"], stride=cs["s"], padding=cs["padding"], dilation=cs["d"],
                     groups=cs["groups"], padding_mode=cs["mode"], random_state=3)
    out = conv.forward(x)
    g = rng.standard_normal(out.shape).astype(np.float32)
    dx = conv.backward(g)
    rout, rdx, rdw, rdb = ref_conv_options(x, conv.weight_, conv.bias_, cs["s"], _pads(cs), cs["d"], cs["groups"],
                                           cs["mode"], g)
    np.testing.assert_allclose(out, rout, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(dx, rdx, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(conv.grad_weight_, rdw, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(conv.grad_bias_, rdb, rtol=1e-5, atol=1e-5)


def test_option_reference_matches_torch():
    torch = pytest.importorskip("torch")
    for cs in OPTION_CASES:
        rng = np.random.default_rng(7)
        x = rng.standard_normal((2, cs["c"], 7, 8)).astype(np.float32)
        m = torch.nn.Conv2d(cs["c"], cs["oc"], cs["k"], stride=cs["s"], padding=cs["padding"], dilation=cs["d"],
                            groups=cs["groups"], padding_mode=cs["mode"]).double()
        tx = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        out = m(tx)
        g = rng.standard_normal(tuple(out.shape)).astype(np.float32)
        out.backward(torch.tensor(g, dtype=torch.float64))
        w = m.weight.detach().numpy().astype(np.float32)
        b = m.bias.detach().numpy().astype(np.float32)
        # the reference in float64 over the float32-rounded weights
        with torch.no_grad():
            m.weight.copy_(torch.tensor(w, dtype=torch.float64))
            m.bias.copy_(torch.tensor(b, dtype=torch.float64))
        tx2 = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        m.zero_grad()
        out2 = m(tx2)
        out2.backward(torch.tensor(g, dtype=torch.float64))
        rout, rdx, rdw, rdb = ref_conv_options(x, w, b, cs["s"], _pads(cs), cs["d"], cs["groups"], cs["mode"], g)
        np.testing.assert_allclose(out2.detach().numpy(), rout, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(tx2.grad.numpy(), rdx, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(m.weight.grad.numpy(), rdw, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(m.bias.grad.numpy(), rdb, rtol=1e-10, atol=1e-10)
