# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""identity_break lane body for the six training primitives routed through
mojolearn.training (workstream D, 2026-09-14), for the harness's owner to
merge into tools/identity_break.py. A function lane in the
cross-entropy-arms style: weights from the hashed stream (`_hw`), ids from
the fixture bytes (`_ids`), activations from the fixture slab (`_seq`).
The embedding backward's run-sorted fold sees duplicate ids because the
byte stream repeats; the linear backward's weight gradient contracts over
the 64 tokens (clause 9.2).
"""


@lane("training-primitives")
def _(ml, X, yc, yr, Xh=None):
    """embedding_forward/backward (the gather and the run-sorted fold),
    rms_norm_forward/backward, linear_forward/backward (the identical
    GEMM at OP_NT and the two backward products). Every output hashed;
    the probe runs the three forwards on a held-out slab."""
    T = ml.training
    V, D, N, K = 64, 32, 64, 16
    w_emb = _hw((V, D), "prim:emb", -0.25, 0.25)
    ids = (_ids(X, 1, N).reshape(N) % V).astype(np.int32)
    x = np.ascontiguousarray(_seq(X, 1, N, D).reshape(N, D))
    g = np.ascontiguousarray(np.abs(_hw((D,), "prim:rms", 0.5, 1.5)))
    w_lin = _hw((K, D), "prim:lin", -0.25, 0.25)
    dy = _hw((N, D), "prim:dy", -1.0, 1.0)
    dc = _hw((N, K), "prim:dc", -1.0, 1.0)
    emb = T.embedding_forward(w_emb, ids)
    demb = T.embedding_backward(dy, ids, V)
    rms = T.rms_norm_forward(x, g, 1e-5)
    dx, dg = T.rms_norm_backward(dy, x, g, 1e-5)
    lin = T.linear_forward(x, w_lin)
    da, dw = T.linear_backward(dc, x, w_lin)
    parts = dict(emb=_h(np.asarray(emb)), demb=_h(np.asarray(demb)), rms=_h(np.asarray(rms)),
                 drms=_h(np.asarray(dx), np.asarray(dg)), lin=_h(np.asarray(lin)), dlin=_h(np.asarray(da), np.asarray(dw)))
    xh = np.ascontiguousarray(_seq(Xh, 1, N, D).reshape(N, D))
    idh = (_ids(Xh, 1, N).reshape(N) % V).astype(np.int32)
    return _fit(parts, T, lambda e: (np.asarray(e.embedding_forward(w_emb, idh)), np.asarray(e.rms_norm_forward(xh, g, 1e-5)),
                                     np.asarray(e.linear_forward(xh, w_lin))))
