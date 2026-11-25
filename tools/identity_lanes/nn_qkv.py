# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# ATTENTION OVER EXPLICIT q, k, v: ops.Session.attention (lane nn-blocks-qkv, 2026-11-25).
#
# Verify through `device=` (docs/plans/VERIFY_THROUGH_DEVICE.md): the lane is device-agnostic (contract 1, as
# tools/identity_lanes/nn_programs.py): the harness's view gives `ops.Session` the side, so the same body runs
# once with device="cpu" and once with device="gpu" and both digests are compared with the reference.
#
# Loaded by tools/identity_break.py `_load_lane_fragments` in the harness namespace: np, os, tempfile, _h, _hw,
# _fit, _same_bytes, _batch_decl and `lane` are the harness's, read when the lane runs; helpers here are prefixed
# `_nn_qkv_`. Reference OWED.
#
# Fault injection (tools/identity_lanes/coverage_sabotage.tsv, the two nn-blocks-qkv arms): the causal mask one
# key late on the device moves part `causal` on the GPU side; a fully masked row of NaN on the host moves part
# `gqa_cross` on the CPU side (its batch row 1, query 2 has no key left).


def _nn_qkv_keep(B, Lq, Lk):
    """A boolean keep mask (B, 1, Lq, Lk) as int8: batch row 0 keeps every key; batch row 1 drops its last two
    keys for every query and drops every key of query 2 (a fully masked row)."""
    keep = np.ones((B, 1, Lq, Lk), dtype=np.int8)
    keep[1, 0, :, Lk - 2:] = 0
    keep[1, 0, 2, :] = 0
    return keep


@lane("ops-attention-qkv")
def _(ml, X, yc, yr, Xh=None):
    """ops.Session.attention, recorded and captured. causal: causal self-attention (2, 2, 6, 4) and the gradients
    of q, k and v of sum(out * seed). gqa_cross: cross-attention with grouped heads (4 query heads over 2
    key/value heads, Lq 5, Lk 7, dv 3) under a boolean ids mask (batch row 1: two keys dropped and query 2
    fully masked, whose output row must be zeros) and its three gradients. additive: inputs without a heads axis
    (2, 4, 6) x (2, 5, 6), a trainable float mask (4, 5) added to the scores (one -inf), scale 0.3, and the
    gradients of q and of the mask through Session.vjp. program: the gqa_cross
    call captured (q, k, v and the ids mask as inputs), run on the same arrays (must equal the eager output),
    saved and loaded (must equal), and run on held-out queries. Parts: every output and gradient."""
    B, H, Hkv, Lq, Lk, d, dv = 2, 4, 2, 5, 7, 4, 3
    cq = _hw((2, 2, 6, 4), "ops-attention-qkv:cq", -1.0, 1.0)
    ck = _hw((2, 2, 6, 4), "ops-attention-qkv:ck", -1.0, 1.0)
    cv = _hw((2, 2, 6, 4), "ops-attention-qkv:cv", -1.0, 1.0)
    cs = _hw((2, 2, 6, 4), "ops-attention-qkv:cseed", -1.0, 1.0)
    gq = _hw((B, H, Lq, d), "ops-attention-qkv:gq", -1.0, 1.0)
    gk = _hw((B, Hkv, Lk, d), "ops-attention-qkv:gk", -1.0, 1.0)
    gv = _hw((B, Hkv, Lk, dv), "ops-attention-qkv:gv", -1.0, 1.0)
    gs = _hw((B, H, Lq, dv), "ops-attention-qkv:gseed", -1.0, 1.0)
    keep = _nn_qkv_keep(B, Lq, Lk)
    parts = {}
    with ml.ops.Session(record=True) as s:
        q, k, v = (s.tensor(a, requires_grad=True) for a in (cq, ck, cv))
        out = s.attention(q, k, v, causal=True)
        s.backward(s.sum(s.mul(out, s.tensor(cs))))
        parts["causal"] = _h(np.asarray(s.numpy(out)), np.asarray(s.grad(q)), np.asarray(s.grad(k)),
                             np.asarray(s.grad(v)))
    with ml.ops.Session(record=True) as s:
        q, k, v = (s.tensor(a, requires_grad=True) for a in (gq, gk, gv))
        out = s.attention(q, k, v, mask=s.ids(keep))
        y = np.asarray(s.numpy(out))
        if not np.all(y[1, :, 2, :] == 0.0):
            raise AssertionError("Session.attention: a fully masked query row must give zeros")
        s.backward(s.sum(s.mul(out, s.tensor(gs))))
        eager = y.copy()
        parts["gqa_cross"] = _h(y, np.asarray(s.grad(q)), np.asarray(s.grad(k)), np.asarray(s.grad(v)))
    aq = _hw((2, 4, 6), "ops-attention-qkv:aq", -1.0, 1.0)
    ak = _hw((2, 5, 6), "ops-attention-qkv:ak", -1.0, 1.0)
    av = _hw((2, 5, 6), "ops-attention-qkv:av", -1.0, 1.0)
    am = _hw((4, 5), "ops-attention-qkv:am", -0.5, 0.5)
    am[3, 1] = -np.inf
    with ml.ops.Session(record=True) as s:
        hq, hm = s.tensor(aq, requires_grad=True), s.tensor(am, requires_grad=True)
        out = s.attention(hq, s.tensor(ak), s.tensor(av), mask=hm, scale=0.3)
        seed = s.tensor(_hw((2, 4, 6), "ops-attention-qkv:aseed", -1.0, 1.0))
        dq, dm = s.vjp(out, [hq, hm], seed)
        parts["additive"] = _h(np.asarray(s.numpy(out)), np.asarray(s.numpy(dq)), np.asarray(s.numpy(dm)))
    held_q = _hw((B, H, Lq, d), "ops-attention-qkv:held-q", -1.0, 1.0)
    with ml.ops.Session() as s:
        q, k, v = s.tensor(gq), s.tensor(gk), s.tensor(gv)
        m = s.ids(keep)
        prog = s.capture([q, k, v, m], [s.attention(q, k, v, mask=m)])
        ran = np.asarray(prog.run(gq, gk, gv, keep))
        _same_bytes("Session.attention", eager, "Program.run of the captured call", ran.reshape(eager.shape))
        held = np.asarray(prog.run(held_q, gk, gv, keep))
        with tempfile.TemporaryDirectory(prefix="ib-ops-attention-qkv-") as tmp:
            path = os.path.join(tmp, "attention.mlp")
            prog.save(path)
            again = np.asarray(s.load_program(path).run(gq, gk, gv, keep))
        _same_bytes("Program.run", ran, "Session.load_program(save).run", again)
        parts["program"] = _h(ran, held)
    return _fit(parts)


_batch_decl("n/a:program-endpoints (a session operation, recorded and captured; outputs and gradients digested "
            "whole)", "ops-attention-qkv")
