# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# ONE LANE PER PUBLIC TRAINING / SEQUENCE FUNCTION (mojolearn.training.*, the
# top-level clip_grad_norm_, cross_entropy, layer_norm_forward and
# layer_norm_backward), sub-lane vdv-cov-func, 2026-11-24.
#
# Every lane calls its function through `ml` with no device= and hands the call
# to `_functions_resident` (tools/identity_lanes/functions.py). The training
# functions use the "train" form (float32 inputs as TrainDeviceArray on the GPU
# side, results downloaded once by `to_host()`), the layer norms the "seq" form
# (SequenceDeviceTensor for the activations x and dy; weight and bias are the
# caller's host parameters, as LayerNorm keeps them). A function that writes its input in place
# (clip_grad_norm_) gets fresh copies per call, so the two runs see the same
# values. Nested runners are called through a lambda so the static endpoint
# walker (tools/verification_endpoint_calls.py) reaches their public calls.


@lane("fn-training-linear-forward")
def _(ml, X, yc, yr, Xh=None):
    a = _functions_mat(X, 64, 32, "fn-lin-f:a")
    w = _functions_mat(X[64:], 16, 32, "fn-lin-f:w")
    return _functions_resident(ml, "training.linear_forward", lambda p: dict(
        y=_functions_d(ml.training.linear_forward(p(a), p(w)))), form="train")


@lane("fn-training-linear-backward")
def _(ml, X, yc, yr, Xh=None):
    a = _functions_mat(X, 64, 32, "fn-lin-b:a")
    w = _functions_mat(X[64:], 16, 32, "fn-lin-b:w")
    dc = _functions_mat(X[128:], 64, 16, "fn-lin-b:dc")
    return _functions_resident(ml, "training.linear_backward", lambda p: dict(
        grads=_functions_d(ml.training.linear_backward(p(dc), p(a), p(w)))), form="train")


@lane("fn-training-embedding-forward")
def _(ml, X, yc, yr, Xh=None):
    w = _functions_mat(X, 64, 32, "fn-emb-f:w")
    ids = np.ascontiguousarray((_ids(X, 1, 96).reshape(96) % 64).astype(np.int32))
    return _functions_resident(ml, "training.embedding_forward", lambda p: dict(
        y=_functions_d(ml.training.embedding_forward(p(w), p(ids)))), form="train")


@lane("fn-training-embedding-backward")
def _(ml, X, yc, yr, Xh=None):
    dy = _functions_mat(X, 96, 32, "fn-emb-b:dy")
    ids = np.ascontiguousarray((_ids(X, 1, 96).reshape(96) % 64).astype(np.int32))
    return _functions_resident(ml, "training.embedding_backward", lambda p: dict(
        dw=_functions_d(ml.training.embedding_backward(p(dy), p(ids), 64))), form="train")


@lane("fn-training-rms-norm-forward")
def _(ml, X, yc, yr, Xh=None):
    x = _functions_mat(X, 64, 32, "fn-rms-f:x")
    g = _functions_pos(X, 32, 5, "fn-rms-f:g")
    return _functions_resident(ml, "training.rms_norm_forward", lambda p: dict(
        y=_functions_d(ml.training.rms_norm_forward(p(x), p(g), 1e-5))), form="train")


@lane("fn-training-rms-norm-backward")
def _(ml, X, yc, yr, Xh=None):
    x = _functions_mat(X, 64, 32, "fn-rms-b:x")
    g = _functions_pos(X, 32, 5, "fn-rms-b:g")
    dy = _functions_mat(X[64:], 64, 32, "fn-rms-b:dy")
    return _functions_resident(ml, "training.rms_norm_backward", lambda p: dict(
        grads=_functions_d(ml.training.rms_norm_backward(p(dy), p(x), p(g), 1e-5))), form="train")


@lane("fn-training-cross-entropy")
def _(ml, X, yc, yr, Xh=None):
    """cross_entropy (as training.cross_entropy and mojolearn.cross_entropy):
    mean with smoothing and its gradient, sum, and an ignored target."""
    logits = _functions_mat(X, 64, 40, "fn-ce:logits")
    t = np.ascontiguousarray(_ids(X, 1, 64, 40).reshape(-1).astype(np.int64))
    ti = t.copy()
    ti[::7] = -100
    return _functions_resident(ml, "training.cross_entropy", lambda p: dict(
        mean=_functions_d(ml.training.cross_entropy(p(logits), p(t), label_smoothing=0.1, return_grad=True)),
        total=_functions_d(ml.training.cross_entropy(p(logits), p(t), reduction="sum")),
        ignored=_functions_d(ml.training.cross_entropy(p(logits), p(ti))),
        top=_functions_d(ml.cross_entropy(p(logits), p(t), label_smoothing=0.1, return_grad=True))), form="train")


@lane("fn-training-chunked-lm-head-loss")
def _(ml, X, yc, yr, Xh=None):
    h = _functions_mat(X, 32, 16, "fn-chunk:h")
    w = _functions_mat(X[32:], 300, 16, "fn-chunk:w")
    t = np.ascontiguousarray(_ids(X, 1, 32, 300).reshape(-1))
    return _functions_resident(ml, "training.chunked_lm_head_loss", lambda p: dict(
        loss=_functions_d(ml.training.chunked_lm_head_loss(p(h), p(w), p(t))),
        grads=_functions_d(ml.training.chunked_lm_head_loss(p(h), p(w), p(t), return_grad=True))), form="train")


@lane("fn-training-clip-grad-norm")
def _(ml, X, yc, yr, Xh=None):
    """clip_grad_norm_ (as training.clip_grad_norm_ and mojolearn.clip_grad_norm_)
    over three gradients, the L2 norm (the one norm the profile has): the total
    norm and the gradients after a clipping call (max_norm 0.5) and after a
    call under the total (max_norm 1e6, gradients left as they are), each call
    on fresh copies (the function writes in place). norm_type=inf is refused by
    name (bindings/optimizer_public_door.mojo door_clip_options) and is held as
    a refusal here, not hashed: hashing it made the clean cell REFUSED on every
    side (coverage-gate-training-step_scalars-cpu, d1641666a4)."""
    gs = [_functions_mat(X, 16, 8, "fn-clip:g0"), _functions_mat(X[16:], 4, 3, "fn-clip:g1"),
          np.ascontiguousarray(_functions_cont(X, 5, 6, "fn-clip:g2") * np.float32(4))]

    def clip(p):
        a = [p(g.copy()) for g in gs]
        b = [p(g.copy()) for g in gs]
        c = [p(g.copy()) for g in gs]
        try:
            ml.training.clip_grad_norm_([p(g.copy()) for g in gs], 0.25, norm_type=float("inf"))
        except Exception as error:  # the host door raises NotImplementedError; any other refusal is a defect
            if type(error).__name__ != "NotImplementedError" and "NotImplementedError" not in str(error):
                raise
        else:
            raise AssertionError("fn-training-clip-grad-norm: norm_type=inf was accepted; the profile refuses it by name")
        return dict(l2=_functions_d(ml.training.clip_grad_norm_(a, 0.5), a),
                    unclipped=_functions_d(ml.training.clip_grad_norm_(b, 1e6), b),
                    top=_functions_d(ml.clip_grad_norm_(c, 0.5), c))

    return _functions_resident(ml, "training.clip_grad_norm_", lambda p: clip(p), form="train")


@lane("fn-training-accumulate-grads")
def _(ml, X, yc, yr, Xh=None):
    """accumulate_grads over four microbatches of three tensors, aligned and not."""
    base = _functions_mat(X, 16, 8, "fn-acc:base")
    micro = [[np.ascontiguousarray(base * np.float32(k + 1) + np.float32(1.0)),
              np.ascontiguousarray(base[:4, :3] - np.float32(k)),
              np.array([k + 1, -(k + 1), 2 * k + 1], dtype=np.float32)] for k in range(4)]  # glue: four microbatches

    def acc(p):
        m = [[p(t) for t in mb] for mb in micro]
        return dict(aligned=_functions_d(ml.training.accumulate_grads(m, 1024)),
                    unaligned=_functions_d(ml.training.accumulate_grads(m, None)))

    return _functions_resident(ml, "training.accumulate_grads", lambda p: acc(p), form="train")


@lane("fn-training-mlp-inference-sessions")
def _(ml, X, yc, yr, Xh=None):
    w1 = _hw((24, 8), "fn-mlp:w1", -0.5, 0.5)
    b1 = _hw((24,), "fn-mlp:b1", -0.1, 0.1)
    w2 = _hw((3, 24), "fn-mlp:w2", -0.5, 0.5)
    b2 = _hw((3,), "fn-mlp:b2", -0.1, 0.1)
    sessions = [_functions_mat(X, 5, 8, "fn-mlp:s0"), _functions_mat(X[5:], 1, 8, "fn-mlp:s1"),
                _functions_mat(X[6:], 64, 8, "fn-mlp:s2")]
    return _functions_resident(ml, "training.mlp_inference_sessions", lambda p: dict(
        logits=_functions_d(ml.training.mlp_inference_sessions([p(s) for s in sessions], p(w1), p(b1), p(w2), p(b2)))),
        form="train")


@lane("fn-training-residual-dropout")
def _(ml, X, yc, yr, Xh=None):
    v = _functions_mat(X, 16, 8, "fn-rd:v")
    r = _functions_mat(X[16:], 16, 8, "fn-rd:r")
    return _functions_resident(ml, "training.residual_dropout", lambda p: dict(
        drop=_functions_d(ml.training.residual_dropout(p(v), p(r), p=0.375, seed=42, stream=3, offset=17)),
        plain=_functions_d(ml.training.residual_dropout(p(v), p(r), p=0.0, seed=42))), form="train")


@lane("fn-training-residual-dropout-backward")
def _(ml, X, yc, yr, Xh=None):
    g = _functions_mat(X, 16, 8, "fn-rdb:g")
    return _functions_resident(ml, "training.residual_dropout_backward", lambda p: dict(
        grads=_functions_d(ml.training.residual_dropout_backward(p(g), p=0.375, seed=42, stream=3, offset=17))),
        form="train")


@lane("fn-layer-norm-forward")
def _(ml, X, yc, yr, Xh=None):
    """layer_norm_forward over (2, 8) with weight and bias, and over the last
    axis with neither."""
    x3 = _functions_mat(X, 128, 16, "fn-ln-f:x").reshape(128, 2, 8)
    wgt = _functions_pos(X, 16, 5, "fn-ln-f:w").reshape(2, 8)
    bias = _functions_cont(X, 16, 6, "fn-ln-f:b").reshape(2, 8)
    x2 = _functions_mat(X[128:], 128, 16, "fn-ln-f:x2")
    return _functions_resident(ml, "layer_norm_forward", lambda p: dict(
        affine=_functions_d(ml.layer_norm_forward(p(x3), (2, 8), weight=wgt, bias=bias, eps=1e-3)),
        plain=_functions_d(ml.layer_norm_forward(p(x2)))), form="seq")


@lane("fn-layer-norm-backward")
def _(ml, X, yc, yr, Xh=None):
    x3 = _functions_mat(X, 128, 16, "fn-ln-b:x").reshape(128, 2, 8)
    dy3 = _functions_mat(X[128:], 128, 16, "fn-ln-b:dy").reshape(128, 2, 8)
    wgt = _functions_pos(X, 16, 5, "fn-ln-b:w").reshape(2, 8)
    bias = _functions_cont(X, 16, 6, "fn-ln-b:b").reshape(2, 8)
    return _functions_resident(ml, "layer_norm_backward", lambda p: dict(
        affine=_functions_d(ml.layer_norm_backward(p(dy3), p(x3), (2, 8), weight=wgt, bias=bias, eps=1e-3)),
        plain=_functions_d(ml.layer_norm_backward(p(dy3), p(x3), (2, 8)))), form="seq")


_batch_decl(_FUNCTIONS_BATCH_NA, "fn-training-linear-forward", "fn-training-linear-backward",
            "fn-training-embedding-forward", "fn-training-embedding-backward", "fn-training-rms-norm-forward",
            "fn-training-rms-norm-backward", "fn-training-cross-entropy", "fn-training-chunked-lm-head-loss",
            "fn-training-clip-grad-norm", "fn-training-accumulate-grads", "fn-training-mlp-inference-sessions",
            "fn-training-residual-dropout", "fn-training-residual-dropout-backward", "fn-layer-norm-forward",
            "fn-layer-norm-backward")
