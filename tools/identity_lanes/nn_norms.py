# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE NORMALIZATION LAYERS' IDENTITY LANES (lane nn-norms, 2026-11-23): Sequential.group_norm and
# instance_norm1d / instance_norm2d (autograd/nn_norms.mojo, layer codes 54 and 55).
#
# A fragment of tools/identity_break.py (executed in its namespace, `_load_lane_fragments`; the rules are
# tools/identity_lanes/sequence.py's header). Each lane builds a small Sequential with the layers, fits one
# epoch of one full unshuffled batch on 32 fixture sequences (SGD with momentum: the momentum buffer is the
# gradient) and digests the epoch loss, the gradient, the updated parameters (tracked running statistics
# included) and the probabilities on the training and held-out sequences. Registered in
# python/mojolearn/_surface_sequence.py (x_sequence training lanes) and the routine profile
# (python/mojolearn/_verification_profiles.py ROUTINE_TAPE).


def _nn_norms_data(X, yc, Xh):
    """(32 sequences of 4 channels and 8 steps, their class codes, 32 held-out sequences)."""
    x = np.ascontiguousarray(X[:256, :4], dtype=np.float32).reshape(32, 8, 4).transpose(0, 2, 1)
    xh = np.ascontiguousarray((X if Xh is None else Xh)[:256, :4], dtype=np.float32).reshape(32, 8, 4)
    xh = xh.transpose(0, 2, 1)
    y = np.ascontiguousarray(yc[7:256:8], dtype=np.int64)
    return np.ascontiguousarray(x), y, np.ascontiguousarray(xh)


def _nn_norms_step(ml, m, x, y, xh):
    B = _sequence_tape_binding(ml)
    m = m.compile(loss="cross_entropy", optimizer="momentum", lr=0.05, momentum=0.9)
    m = m.fit(x, y, epochs=1, batch_size=32, shuffle=False, seed=0)
    d = lambda a: _sequence_tape_digest(B, a)
    return _fit(dict(loss=d(np.asarray(m.loss_curve_, dtype=np.float32)), grad=d(m.optimizer_m_),
                     params=d(m.params_), proba=d(m.predict_proba(x)), held_proba=d(m.predict_proba(xh))))


@lane("sequence-norms-resnet1d-gn")
def _(ml, X, yc, yr, Xh=None):
    """A ResNet-1D with group norm (a conv-gn-relu stem, a residual block of two conv-gn layers with a 1x1
    conv-gn shortcut; 4 groups of 2 channels, and one no-affine group norm of 1 group before the head): one
    seeded training step."""
    x, y, xh = _nn_norms_data(X, yc, Xh)
    S = ml.Sequential
    body = (S().conv1d(8, 3, padding=1, bias=False).group_norm(4, 8).relu()
            .conv1d(8, 3, padding=1, bias=False).group_norm(4, 8))
    m = (S(input_shape=(4, 8), numeric_mode="identical").conv1d(8, 5, padding=2, bias=False).group_norm(4, 8).relu()
         .residual_block(body, S().conv1d(8, 1, bias=False).group_norm(4, 8)).relu()
         .group_norm(1, 8, affine=False).adaptive_avg_pool1d(1).flatten().linear(2))
    return _nn_norms_step(ml, m, x, y, xh)


@lane("sequence-norms-instance")
def _(ml, X, yc, yr, Xh=None):
    """A 1-D CNN with instance norm: conv1d, instance_norm1d (affine, tracked running statistics: training
    updates them, the probabilities use them), relu, conv1d, instance_norm1d (no affine, untracked), relu,
    adaptive_avg_pool1d, linear: one seeded training step; the running statistics are part of the parameters'
    digest."""
    x, y, xh = _nn_norms_data(X, yc, Xh)
    S = ml.Sequential
    m = (S(input_shape=(4, 8), numeric_mode="identical").conv1d(8, 3, padding=1, bias=False)
         .instance_norm1d(8, affine=True, track_running_stats=True).relu()
         .conv1d(8, 3, padding=1, bias=False).instance_norm1d(8).relu()
         .adaptive_avg_pool1d(1).flatten().linear(2))
    return _nn_norms_step(ml, m, x, y, xh)


_batch_decl("n/a:training-step (the lane checks one seeded training step through the tape; its forward "
            "outputs are train parts, digested whole)", "sequence-norms-resnet1d-gn", "sequence-norms-instance")
