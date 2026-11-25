# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# LANE session-ops-rest's IDENTITY LANES (2026-11-25): Sequential's norms, prelu and mish, multi-head attention and
# positional encoding, losses and raw state-space scans as ops.Session operations (autograd/session_ops_rest.mojo),
# forward and backward, and captured Programs of them.
#
# Verify through `device=` (docs/plans/VERIFY_THROUGH_DEVICE.md): each lane opens its sessions with no `device=` and
# the harness's view gives every Session the side, so the same lane runs the host column (device="cpu") and the
# device column (device="gpu") and the two digests are compared with the reference. Session and Program have no
# digest: the lanes hash the selected outputs, gradients and state directly.
#
# Six lanes, one per implementation family; each op of a family is its own part, so a fault in one op names it.
# Fixtures are the smallest that reach each op's edges (a channel group of two, a one-element spatial axis, a
# causal mask, a short key length, a target with a repeated label and a blank-only row, a sign change inside
# smooth L1's threshold, an empty probability), not big tensors.
#
# Loaded by tools/identity_break.py `_load_lane_fragments` in the harness namespace: np, _h, _fit, _hw, _same_bytes,
# _batch_decl and `lane` are the harness's; helpers here are prefixed `_sor_`. Reference OWED (recorded once, at the
# release's one verification). Sabotage arms (tools/identity_lanes/coverage_sabotage.tsv, session-ops-rest group):
# the gate arithmetic faults of each side (sequence/ops.mojo add and fma3 one ulp up, host bindings or GPU bindings
# only) move every lane here on that side and leave the other side's cell.


def _sor_first(r):
    """A Program's one output (run returns the outputs; a single output may come back bare)."""
    if isinstance(r, (list, tuple)):
        return np.asarray(r[0])
    return np.asarray(r)


def _sor_backward(s, outs, key):
    """sum(c_k * out_k) over the outputs with fixed weights, then backward; returns the loss handle."""
    loss = None
    for k, o in enumerate(outs):  # driver-loop: over the lane's few outputs
        c = _hw(tuple(s.shape(o)), f"{key}:c{k}", -1.0, 1.0)
        t = s.sum(s.mul(o, s.tensor(c)))
        loss = t if loss is None else s.add(loss, t)
    s.backward(loss)
    return loss


@lane("ops-layer-norms")
def _(ml, X, yc, yr, Xh=None):
    """ops.Session batch_norm, group_norm and instance_norm (Sequential's norm layers' bodies). x (3, 4, 2, 1): a
    spatial axis of one, so batch_norm folds over 6 values a channel, group_norm (2 groups of 2 channels) over 4,
    instance_norm over 2. batch_norm and instance_norm train with running statistics (state tensors updated in
    place, the momentum 0.25), then batch_norm runs in eval from the updated statistics. Parts per op: forward,
    gradients of x, weight and bias, and the running statistics after the call; then a Program of the eval
    batch_norm and group_norm run on new inputs."""
    x0 = _hw((3, 4, 2, 1), "sor-norm:x", -2.0, 2.0)
    w0 = _hw((4,), "sor-norm:w", 0.5, 1.5)
    b0 = _hw((4,), "sor-norm:b", -0.5, 0.5)
    parts = {}
    with ml.ops.Session(record=True, numeric_mode="identical") as s:
        x = s.tensor(x0, requires_grad=True)
        w = s.tensor(w0, requires_grad=True)
        b = s.tensor(b0, requires_grad=True)
        rm = s.state(np.zeros(4, np.float32))
        rv = s.state(np.ones(4, np.float32))
        im = s.state(np.zeros(4, np.float32))
        iv = s.state(np.ones(4, np.float32))
        outs = {
            "batch_norm": s.batch_norm(x, rm, rv, w, b, training=True, momentum=0.25),
            "group_norm": s.group_norm(x, 2, w, b),
            "instance_norm": s.instance_norm(x, im, iv, w, b, use_input_stats=True, momentum=0.25),
        }
        for name, o in outs.items():  # driver-loop: over the three ops' outputs
            parts[name] = _h(np.asarray(s.numpy(o)))
        _sor_backward(s, list(outs.values()), "sor-norm")
        parts["grad"] = _h(*[np.asarray(s.grad(t)) for t in (x, w, b)])  # glue: one array per input
        parts["running"] = _h(*[np.asarray(s.numpy(t)) for t in (rm, rv, im, iv)])  # glue: the four statistics
        stats = [np.asarray(s.numpy(t)).reshape(-1) for t in (rm, rv)]  # glue: the trained statistics
    with ml.ops.Session(numeric_mode="identical") as s:
        x = s.tensor(x0)
        w, b = s.tensor(w0), s.tensor(b0)
        rm, rv = s.tensor(stats[0]), s.tensor(stats[1])
        ev = s.batch_norm(x, rm, rv, w, b, training=False)
        gn = s.group_norm(x, 2, w, b)
        eager = [np.asarray(s.numpy(ev)), np.asarray(s.numpy(gn))]
        prog = s.capture(inputs=[x], outputs=[ev, gn])
        again = prog.run(x0)
        _same_bytes("Program batch_norm", np.asarray(again[0]), "Session.batch_norm", eager[0])
        x2 = np.ascontiguousarray((x0[::-1] * np.float32(-0.75)).astype(np.float32))
        other = prog.run(x2)
        parts["program"] = _h(*eager, np.asarray(other[0]), np.asarray(other[1]))
    return _fit(parts)


@lane("ops-layer-activations")
def _(ml, X, yc, yr, Xh=None):
    """ops.Session prelu (one slope per channel, and one shared slope) and mish on x (2, 3, 2) with values of both
    signs; parts: each forward, then the gradients of x and the two slopes."""
    x0 = _hw((2, 3, 2), "sor-act:x", -3.0, 3.0)
    a0 = _hw((3,), "sor-act:a", 0.05, 0.4)
    parts = {}
    with ml.ops.Session(record=True, numeric_mode="identical") as s:
        x = s.tensor(x0, requires_grad=True)
        a = s.tensor(a0, requires_grad=True)
        a1 = s.tensor(np.asarray([0.2], np.float32), requires_grad=True)
        outs = {"prelu": s.prelu(x, a), "prelu_shared": s.prelu(x, a1), "mish": s.mish(x)}
        for name, o in outs.items():  # driver-loop: over the three outputs
            parts[name] = _h(np.asarray(s.numpy(o)))
        _sor_backward(s, list(outs.values()), "sor-act")
        parts["grad"] = _h(*[np.asarray(s.grad(t)) for t in (x, a, a1)])  # glue: one array per input
    return _fit(parts)


@lane("ops-layer-attention")
def _(ml, X, yc, yr, Xh=None):
    """ops.Session multi_head_attention and positional_encoding. Self-attention, causal, x (2, 3, 4), 2 heads,
    biases; cross-attention from a (2, 4, 6) memory (kdim 6) with key_lengths (4, 2): row 1's last two keys masked;
    positional_encoding on (2, 3, 4). Parts: each forward, the gradients of the inputs and every weight; then a
    Program of the self-attention run on a new input."""
    E, H = 4, 2
    x0 = _hw((2, 3, E), "sor-att:x", -1.0, 1.0)
    m0 = _hw((2, 4, 6), "sor-att:m", -1.0, 1.0)
    w_in0 = _hw((E, 3 * E), "sor-att:win", -0.5, 0.5)
    b_in0 = _hw((3 * E,), "sor-att:bin", -0.1, 0.1)
    wq0 = _hw((E, E), "sor-att:wq", -0.5, 0.5)
    wkv0 = _hw((6, 2 * E), "sor-att:wkv", -0.5, 0.5)
    bq0 = _hw((E,), "sor-att:bq", -0.1, 0.1)
    bkv0 = _hw((2 * E,), "sor-att:bkv", -0.1, 0.1)
    wo0 = _hw((E, E), "sor-att:wo", -0.5, 0.5)
    bo0 = _hw((E,), "sor-att:bo", -0.1, 0.1)
    parts = {}
    with ml.ops.Session(record=True, numeric_mode="identical") as s:
        x = s.tensor(x0, requires_grad=True)
        m = s.tensor(m0, requires_grad=True)
        ws = [s.tensor(a, requires_grad=True) for a in (w_in0, b_in0, wq0, wkv0, bq0, bkv0, wo0, bo0)]  # glue
        w_in, b_in, wq, wkv, bq, bkv, wo, bo = ws
        lens = s.tensor(np.asarray([4.0, 2.0], np.float32))
        outs = {
            "self_causal": s.multi_head_attention(x, x, x, H, w_in, b_in, wo, bo, is_causal=True),
            "cross_lengths": s.multi_head_attention(x, m, m, H, (wq, wkv), (bq, bkv), wo, bo, key_lengths=lens),
            "positional_encoding": s.positional_encoding(x),
        }
        for name, o in outs.items():  # driver-loop: over the three outputs
            parts[name] = _h(np.asarray(s.numpy(o)))
        _sor_backward(s, list(outs.values()), "sor-att")
        parts["grad"] = _h(*[np.asarray(s.grad(t)) for t in [x, m] + ws])  # glue: one array per input
    with ml.ops.Session(numeric_mode="identical") as s:
        x = s.tensor(x0)
        w_in, b_in, wo, bo = s.tensor(w_in0), s.tensor(b_in0), s.tensor(wo0), s.tensor(bo0)
        y = s.multi_head_attention(x, x, x, H, w_in, b_in, wo, bo, is_causal=True)
        eager = np.asarray(s.numpy(y))
        prog = s.capture(inputs=[x], outputs=[y])
        _same_bytes("Program multi_head_attention", _sor_first(prog.run(x0)), "Session.multi_head_attention", eager)
        x2 = np.ascontiguousarray((x0[:, ::-1] * np.float32(0.5)).astype(np.float32))
        parts["program"] = _h(eager, _sor_first(prog.run(x2)))
    return _fit(parts)


@lane("ops-layer-losses")
def _(ml, X, yc, yr, Xh=None):
    """ops.Session smooth_l1_loss (beta 0.5, differences on both sides of the threshold), nll_loss (class ids with a
    repeat), kl_div (batchmean, one target probability 0) and binary_cross_entropy_with_logits (pos_weight 3) on
    (3, 4) inputs; parts per loss: the value and the input's gradient."""
    p0 = _hw((3, 4), "sor-loss:p", -2.0, 2.0)
    t0 = _hw((3, 4), "sor-loss:t", -2.0, 2.0)
    z0 = _hw((3, 4), "sor-loss:z", -1.0, 1.0)
    lp0 = (z0 - np.log(np.exp(z0).sum(axis=1, keepdims=True))).astype(np.float32)  # cpu-route: lane input
    q0 = np.asarray([[0.5, 0.25, 0.25, 0.0], [0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.25, 0.25]], np.float32)
    y0 = np.asarray([[1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 0]], np.float32)
    parts = {}
    with ml.ops.Session(record=True, numeric_mode="identical") as s:
        for name in ("smooth_l1", "nll", "kl_div", "bce_pos_weight"):  # driver-loop: over the four losses
            x = s.tensor(p0 if name in ("smooth_l1", "bce_pos_weight") else np.ascontiguousarray(lp0),
                         requires_grad=True)
            if name == "smooth_l1":
                loss = s.smooth_l1_loss(x, s.tensor(t0), beta=0.5)
            elif name == "nll":
                loss = s.nll_loss(x, s.ids(np.asarray([2, 0, 2])))
            elif name == "kl_div":
                loss = s.kl_div(x, s.tensor(q0), reduction="batchmean")
            else:
                loss = s.binary_cross_entropy_with_logits(x, s.tensor(y0), pos_weight=3.0)
            s.backward(loss)
            parts[name] = _h(np.asarray(s.numpy(loss)), np.asarray(s.grad(x)))
            s.reset()
    return _fit(parts)


@lane("ops-layer-ctc")
def _(ml, X, yc, yr, Xh=None):
    """ops.Session ctc_loss on (T 5, N 3, C 4) log-probabilities: a target with a repeated label (1, 1), a shorter
    input length, and an empty target; targets and lengths given as ids, then as float tensors (the same words);
    'mean' and 'sum'. Parts: the two losses and the gradient of the log-probabilities; then a Program run."""
    z0 = _hw((5, 3, 4), "sor-ctc:z", -1.0, 1.0)
    lp0 = np.ascontiguousarray((z0 - np.log(np.exp(z0).sum(axis=2, keepdims=True))).astype(np.float32))  # cpu-route
    tg = np.asarray([[1, 1, 2], [3, 0, 0], [0, 0, 0]], np.int64)
    il = np.asarray([5, 4, 5], np.int64)
    tl = np.asarray([3, 1, 0], np.int64)
    parts = {}
    with ml.ops.Session(record=True, numeric_mode="identical") as s:
        lp = s.tensor(lp0, requires_grad=True)
        mean = s.ctc_loss(lp, s.ids(tg), s.ids(il), s.ids(tl))
        total = s.ctc_loss(lp, s.tensor(tg.astype(np.float32)), s.tensor(il.astype(np.float32)),
                           s.tensor(tl.astype(np.float32)), reduction="sum")
        s.backward(s.add(mean, total))
        parts["loss"] = _h(np.asarray(s.numpy(mean)), np.asarray(s.numpy(total)))
        parts["grad"] = _h(np.asarray(s.grad(lp)))
    with ml.ops.Session(numeric_mode="identical") as s:
        lp = s.tensor(lp0)
        loss = s.ctc_loss(lp, s.ids(tg), s.ids(il), s.ids(tl))
        eager = np.asarray(s.numpy(loss))
        prog = s.capture(inputs=[lp], outputs=[loss])
        _same_bytes("Program ctc_loss", _sor_first(prog.run(lp0)), "Session.ctc_loss", eager)
        parts["program"] = _h(eager, _sor_first(prog.run(np.ascontiguousarray(lp0[::-1]))))
    return _fit(parts)


@lane("ops-layer-ssm")
def _(ml, X, yc, yr, Xh=None):
    """ops.Session selective_scan (batch 2, length 3, dim 2, state 4), ssd_scan (batch 1, length 3, 1 head) and
    trapezoid_scan (batch 2, length 3, 2 heads, P 2, N 2); parts per scan: the output and every input's
    gradient."""
    parts = {}
    with ml.ops.Session(record=True, numeric_mode="identical") as s:
        u = s.tensor(_hw((2, 3, 2), "sor-ssm:u", -1.0, 1.0), requires_grad=True)
        dl = s.tensor(_hw((2, 3, 2), "sor-ssm:dl", 0.05, 0.5), requires_grad=True)
        al = s.tensor(_hw((2, 4), "sor-ssm:al", -1.0, 0.5), requires_grad=True)
        bm = s.tensor(_hw((2, 3, 4), "sor-ssm:b", -1.0, 1.0), requires_grad=True)
        cm = s.tensor(_hw((2, 3, 4), "sor-ssm:c", -1.0, 1.0), requires_grad=True)
        dd = s.tensor(_hw((2,), "sor-ssm:d", -1.0, 1.0), requires_grad=True)
        y = s.selective_scan(u, dl, al, bm, cm, dd)
        parts["selective_scan"] = _h(np.asarray(s.numpy(y)))
        _sor_backward(s, [y], "sor-ssm:s")
        parts["selective_scan_grad"] = _h(*[np.asarray(s.grad(t)) for t in (u, dl, al, bm, cm, dd)])  # glue
        s.reset()
        xbc = s.tensor(_hw((1, 3, 64 + 256), "sor-ssm:xbc", -1.0, 1.0), requires_grad=True)
        dt = s.tensor(_hw((1, 3, 1), "sor-ssm:dt", 0.05, 0.5), requires_grad=True)
        al2 = s.tensor(_hw((1,), "sor-ssm:al2", -1.0, 0.5), requires_grad=True)
        d2 = s.tensor(_hw((1,), "sor-ssm:d2", -1.0, 1.0), requires_grad=True)
        y2 = s.ssd_scan(xbc, dt, al2, d2)
        parts["ssd_scan"] = _h(np.asarray(s.numpy(y2)))
        _sor_backward(s, [y2], "sor-ssm:d")
        parts["ssd_scan_grad"] = _h(*[np.asarray(s.grad(t)) for t in (xbc, dt, al2, d2)])  # glue
        s.reset()
        tv = s.tensor(_hw((2, 3, 4), "sor-ssm:tv", -1.0, 1.0), requires_grad=True)
        tk = s.tensor(_hw((2, 3, 4), "sor-ssm:tk", -1.0, 1.0), requires_grad=True)
        tq = s.tensor(_hw((2, 3, 4), "sor-ssm:tq", -1.0, 1.0), requires_grad=True)
        tdt = s.tensor(_hw((2, 3, 2), "sor-ssm:tdt", 0.05, 0.5), requires_grad=True)
        ta = s.tensor(_hw((2, 3, 2), "sor-ssm:ta", -1.0, -0.1), requires_grad=True)
        ttr = s.tensor(_hw((2, 3, 2), "sor-ssm:ttr", 0.0, 1.0), requires_grad=True)
        td = s.tensor(_hw((2,), "sor-ssm:td", -1.0, 1.0), requires_grad=True)
        y3 = s.trapezoid_scan(tv, tk, tq, tdt, ta, ttr, td, 2)
        parts["trapezoid_scan"] = _h(np.asarray(s.numpy(y3)))
        _sor_backward(s, [y3], "sor-ssm:t")
        parts["trapezoid_scan_grad"] = _h(*[np.asarray(s.grad(t)) for t in (tv, tk, tq, tdt, ta, ttr, td)])  # glue
    return _fit(parts)


_batch_decl("n/a:training-step (one seeded forward and backward through the ops session, and a captured program "
            "where the lane says; the outputs are parts, digested whole)", "ops-layer-norms",
            "ops-layer-activations", "ops-layer-attention", "ops-layer-losses", "ops-layer-ctc", "ops-layer-ssm")


@lane("ops-select-reduce")
def _(ml, X, yc, yr, Xh=None):
    """ops.Session topk, gather, logsumexp and sequence_mask (lane w3i11-endpoints, 2026-11-25: no lane called
    them), and checkpoint. x (3, 5) from the hashed stream with a repeated value in row 0 (topk's lowest-position-
    first tie rule) and a row of large scores in row 2 (logsumexp's subtracted maximum). Parts: topk largest along
    the last axis and smallest along axis 0 (values and int32 positions); gather of x along axis 1 at fixed
    positions (one position taken twice: its gradient adds both); logsumexp along axis 1 and, kept, along axis 0;
    the gradients of x through the sum of those outputs with fixed weights; sequence_mask of lengths (1, 3, 0) over
    4 positions. Then a two-tanh-layer segment captured from an inference session and run through checkpoint in a
    recording session: the weights' gradients with recompute=True must equal recompute=False byte for byte (the
    documented promise; a difference REFUSES), and the part is the output and those gradients."""
    x0 = np.array(_hw((3, 5), "sor-sel:x", -2.0, 2.0), dtype=np.float32)  # a writable copy
    x0[0, 3] = x0[0, 1]
    x0[2] = x0[2] + np.float32(900.0)
    index0 = np.asarray([[4, 0, 4], [1, 1, 3], [2, 0, 1]], np.int32)
    parts = {}
    with ml.ops.Session(record=True, numeric_mode="identical") as s:
        x = s.tensor(x0, requires_grad=True)
        top_v, top_i = s.topk(x, 2)
        low_v, low_i = s.topk(x, 2, axis=0, largest=False)
        picked = s.gather(x, 1, s.ids(index0))
        lse = s.logsumexp(x, 1)
        lse0 = s.logsumexp(x, 0, keepdims=True)
        parts["topk"] = _h(np.asarray(s.numpy(top_v)), np.asarray(s.numpy(top_i)).view(np.int32),
                           np.asarray(s.numpy(low_v)), np.asarray(s.numpy(low_i)).view(np.int32))
        parts["gather"] = _h(np.asarray(s.numpy(picked)))
        parts["logsumexp"] = _h(np.asarray(s.numpy(lse)), np.asarray(s.numpy(lse0)))
        _sor_backward(s, [top_v, low_v, picked, lse, lse0], "sor-sel")
        parts["grad"] = _h(np.asarray(s.grad(x)))
        lengths = s.tensor(np.asarray([[1.0], [3.0], [0.0]], np.float32))
        parts["sequence_mask"] = _h(np.asarray(s.numpy(s.sequence_mask(lengths, 4))))
    w1 = _hw((4, 6), "sor-sel:w1", -0.5, 0.5)
    w2 = _hw((6, 4), "sor-sel:w2", -0.5, 0.5)
    h0 = _hw((8, 4), "sor-sel:h", -1.0, 1.0)
    runs = {}
    with ml.ops.Session(numeric_mode="identical") as c:
        xin = c.tensor(np.zeros((8, 4), np.float32))
        cw1, cw2 = c.tensor(w1), c.tensor(w2)
        block = c.capture(inputs=[xin], outputs=[c.tanh(c.matmul(c.tanh(c.matmul(xin, cw1)), cw2))])
        for recompute in (True, False):  # driver-loop: over the two checkpoint modes
            with ml.ops.Session(record=True, numeric_mode="identical") as s:
                v1 = s.tensor(w1, requires_grad=True)
                v2 = s.tensor(w2, requires_grad=True)
                out = s.checkpoint(block, s.tensor(h0), params=[v1, v2], recompute=recompute)
                s.backward(s.mean(s.mul(out, out)))
                runs[recompute] = [np.asarray(s.numpy(out)), np.asarray(s.grad(v1)), np.asarray(s.grad(v2))]
    for k, what in enumerate(("output", "grad w1", "grad w2")):  # driver-loop: over the three compared arrays
        _same_bytes("Session.checkpoint recompute=True " + what, runs[True][k],
                    "Session.checkpoint recompute=False " + what, runs[False][k])
    parts["checkpoint"] = _h(*runs[True])
    return _fit(parts)
