# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE COMPOSED TAGGER (lane nn-blocks-composed, 2026-11-25): the nn-blocks pieces in one model.
#
# A variable-length token tagger on `ops.Session`: an embedding (a padding id) concatenated with per-step float
# features, a captured tanh cell with a dropout inside it run by `Session.scan` with lengths, a per-step
# cross-entropy weighted by the mask built from the lengths (the padding label 3 is a class no real token has, so a
# mask bug moves the loss and the gradients), and `Session.param_group_optimizer` with the embedding, the cell and
# the head in three groups (different rates and weight decay, one global clip). It trains two steps, saves (the
# parameters, `optimizer_state_dict` and the step counter that keys the dropout), resumes in a fresh session, and
# the next update must equal uninterrupted training bit for bit; then the same tagger captured as a Program and
# trained with `Program.fit` (the scan's key advanced per minibatch by the replay), saved, loaded and resumed.
#
# A fragment of tools/identity_break.py (loaded by `_load_lane_fragments` into its namespace; np, os, tempfile, _h,
# _fit, _same_bytes, _hw, lane and _batch_decl are that namespace's). Device-agnostic
# (docs/plans/VERIFY_THROUGH_DEVICE.md contract 1): no `device=`; the harness's view gives `ops.Session` its side.
# Helpers here are prefixed `_nn_composed_`. Every reference is OWED. The ops route exposes no `digest_` or
# `training_digest_` (sessions and programs carry none), so every state is digested here as a part.
#
# Sabotage arms (tools/identity_lanes/coverage_sabotage.tsv coverage-autograd-nn_composed_*; autograd/nn_composed.mojo):
# scan_key (the cell's dropout ignores the scan's key: the key checks REFUSE, the train and program parts move) and
# group_state (a resumed grouped optimizer drops the later groups' state: the resume checks REFUSE), each on one side.

_NN_COMPOSED_B, _NN_COMPOSED_T, _NN_COMPOSED_E, _NN_COMPOSED_F, _NN_COMPOSED_H, _NN_COMPOSED_C = 3, 4, 3, 2, 3, 4
_NN_COMPOSED_PAD_ID = 5      # vocabulary 0..4, the padding id 5 (the embedding's padding_idx)
_NN_COMPOSED_PAD_TAG = 3     # tags 0..2, the padding label 3: no real token has it
_NN_COMPOSED_NAMES = ("table", "w", "u", "b", "head_w", "head_b")
_NN_COMPOSED_STATE = ("exp_avg", "exp_avg_sq")


def _nn_composed_data():
    """Three sequences of lengths 1, 3 and 4 (batch first): token ids (the padding id past each length), two float
    features per step, the lengths as a (3, 1) column and the tag codes (the padding label past each length)."""
    B, T, F = _NN_COMPOSED_B, _NN_COMPOSED_T, _NN_COMPOSED_F
    p, q = _NN_COMPOSED_PAD_ID, _NN_COMPOSED_PAD_TAG
    ids = np.array([[2, p, p, p], [0, 4, 1, p], [3, 1, 2, 0]], dtype=np.int32)
    codes = np.array([[1, q, q, q], [0, 2, 1, q], [2, 2, 0, 1]], dtype=np.float32)
    feats = _hw((B, T, F), "ops-composed:feats", -1.0, 1.0)
    lens = np.array([[1.0], [3.0], [4.0]], dtype=np.float32)
    return ids, feats, lens, codes


def _nn_composed_init():
    E, F, H, C = _NN_COMPOSED_E, _NN_COMPOSED_F, _NN_COMPOSED_H, _NN_COMPOSED_C
    return {"table": _hw((_NN_COMPOSED_PAD_ID + 1, E), "ops-composed:table", -0.5, 0.5),
            "w": _hw((E + F, H), "ops-composed:w", -0.5, 0.5), "u": _hw((H, H), "ops-composed:u", -0.5, 0.5),
            "b": _hw((1, H), "ops-composed:b", -0.1, 0.1), "head_w": _hw((H, C), "ops-composed:head_w", -0.5, 0.5),
            "head_b": np.zeros((1, C), np.float32)}


def _nn_composed_cell(c, w, u, b):
    """The step on inference session c: h' = tanh(dropout(x) W + h U + b), output h'. Parameters W, U, b."""
    B, H = _NN_COMPOSED_B, _NN_COMPOSED_H
    h = c.tensor(np.zeros((B, H), np.float32))
    x = c.tensor(np.zeros((B, _NN_COMPOSED_E + _NN_COMPOSED_F), np.float32))
    d = c.dropout(x, 0.25, seed=13, counter=0)
    h2 = c.tanh(c.add(c.add(c.matmul(d, w), c.matmul(h, u)), b))
    return c.capture(inputs=[h, x], outputs=[h2, h2])


def _nn_composed_forward(s, cell, ids, feats, lens, codes, p, cell_params, counter):
    """The tagger on session s; returns (loss, logits). Inputs batch first; the scan runs steps first."""
    B, T, E, H = _NN_COMPOSED_B, _NN_COMPOSED_T, _NN_COMPOSED_E, _NN_COMPOSED_H
    emb = s.reshape(s.embedding(p["table"], ids, padding_idx=_NN_COMPOSED_PAD_ID), (B, T, E))
    x = s.permute(s.concat([emb, feats], axis=2), (1, 0, 2))
    h0 = s.tensor(np.zeros((B, H), np.float32))
    if cell_params is None:
        ys, _ = s.scan(cell, x, h0, lengths=lens, counter=counter)
    else:
        ys, _ = s.scan(cell, x, h0, params=cell_params, lengths=lens, counter=counter)
    logits = s.linear(s.reshape(ys, (T * B, H)), p["head_w"], p["head_b"])
    code = s.reshape(s.transpose(codes), (T * B, 1))
    steps = s.tensor(np.arange(T, dtype=np.float32).reshape(T, 1))
    mask = s.reshape(s.less(steps, s.reshape(lens, (1, B))), (T * B, 1))
    return s.cross_entropy(logits, code, weight=mask), logits


def _nn_composed_optimizer(s, p):
    """AdamW with three groups: the embedding (low rate, decay), the cell, the head (high rate, no decay)."""
    return s.param_group_optimizer([{"params": [p["table"]], "lr": 0.01, "weight_decay": 0.01},
                                    {"params": [p["w"], p["u"], p["b"]], "lr": 0.02},
                                    {"params": [p["head_w"], p["head_b"]], "lr": 0.05, "weight_decay": 0.0}],
                                   "adamw", weight_decay=0.001, clip_grad_norm=1.0)


def _nn_composed_save_state(s, opt):
    """optimizer_state_dict with every state tensor read back (the saved form)."""
    sd = s.optimizer_state_dict(opt)
    state = {k: {n: (np.asarray(s.numpy(v)) if n in _NN_COMPOSED_STATE else v) for n, v in st.items()}
             for k, st in sd["state"].items()}
    return {"epoch": sd["epoch"], "param_groups": sd["param_groups"], "state": state}


def _nn_composed_load_state(s, opt, saved):
    s.load_optimizer_state_dict(opt, {"epoch": saved["epoch"], "param_groups": saved["param_groups"],
                                      "state": {k: {n: (s.tensor(v) if n in _NN_COMPOSED_STATE else v)
                                                    for n, v in st.items()} for k, st in saved["state"].items()}})


def _nn_composed_moments(saved):
    return [saved["state"][k][n] for k in sorted(saved["state"]) for n in _NN_COMPOSED_STATE]


def _nn_composed_saved(tmp, name, arrays):
    """Write arrays to an .npz file in tmp and read them back (the checkpoint's bytes)."""
    path = os.path.join(tmp, name + ".npz")
    np.savez(path, *arrays)
    with np.load(path) as z:
        return [np.array(z[f"arr_{k}"]) for k in range(len(arrays))]  # driver-loop: over the saved arrays


def _nn_composed_eager_step(ml, s, cell, p, data, counter):
    """One training step on recording session s; returns (loss, logits, gradients by name, feats gradient)."""
    ids, feats, lens, codes = data
    fh = s.tensor(feats, requires_grad=True)
    loss, logits = _nn_composed_forward(s, cell, s.ids(ids), fh, s.tensor(lens), s.tensor(codes), p,
                                        [p["w"], p["u"], p["b"]], counter)
    lv, lg = np.asarray(s.numpy(loss)), np.asarray(s.numpy(logits))
    s.backward(loss)
    grads = {k: np.asarray(s.grad(p[k])) for k in _NN_COMPOSED_NAMES}  # glue: one array per parameter
    return lv, lg, grads, np.asarray(s.grad(fh))


def _nn_composed_check_mask(lv, lg, grads, gfeats, lens, codes):
    """The mask contract: padded steps contribute nothing. The feature gradient past each length and the padding
    row of the embedding gradient are exactly zero, and the loss is the mean over the real positions only (a
    float64 oracle from the step's logits; the padding label must not enter it)."""
    B, T = _NN_COMPOSED_B, _NN_COMPOSED_T
    for b in range(B):  # driver-loop: over the three sequences
        n = int(lens[b, 0])
        if np.any(gfeats[b, n:] != 0):
            raise ValueError(f"ops-composed-tagger: sequence {b} (length {n}) has a nonzero feature gradient on a "
                             "padded step")
    if np.any(grads["table"][_NN_COMPOSED_PAD_ID] != 0):
        raise ValueError("ops-composed-tagger: the padding id's embedding row has a nonzero gradient")
    z = lg.astype(np.float64).reshape(T, B, -1)
    terms = []
    for t in range(T):  # driver-loop: over steps (oracle)
        for b in range(B):  # driver-loop: over sequences (oracle)
            if t < int(lens[b, 0]):
                row = z[t, b]
                top = row.max()
                terms.append(top + np.log(np.exp(row - top).sum()) - row[int(codes[b, t])])
    want = float(np.mean(terms))
    got = float(np.asarray(lv, np.float64).reshape(-1)[0])
    if not abs(got - want) <= 1e-5 * max(1.0, abs(want)):
        raise ValueError(f"ops-composed-tagger: the masked loss is {got!r}; the mean over the real positions is "
                         f"{want!r} (the padding label {_NN_COMPOSED_PAD_TAG} entered the loss?)")


def _nn_composed_key_masks(ml):
    """The scan's key, eagerly: a dropout cell scanned with counter 0 and 1 must draw different masks."""
    with ml.ops.Session(numeric_mode="identical") as c:
        hd = c.tensor(np.zeros((2, 8), np.float32))
        xd = c.tensor(np.zeros((2, 8), np.float32))
        yd = c.dropout(xd, 0.5, seed=17, counter=0)
        dcell = c.capture(inputs=[hd, xd], outputs=[c.add(hd, yd), yd])
        ones = np.ones((2, 2, 8), np.float32)
        masks = [np.asarray(c.numpy(c.scan(dcell, c.tensor(ones), c.tensor(np.zeros((2, 8), np.float32)),
                                           counter=k)[0])) for k in (0, 1)]  # glue: two calls
    if np.array_equal(masks[0], masks[1]):
        raise ValueError("ops-composed-tagger: Session.scan with counter 0 and counter 1 drew the same dropout "
                         "masks (the scan's key is not reaching the cell's dropout)")
    return masks


def _nn_composed_program_key(ml):
    """The scan's key under Program.fit: a dropout-only scan whose loss is sum(ys * w), w trained by SGD at rate 1
    from 0, so after each minibatch w has moved by exactly minus that minibatch's mask values (0 or 2). Two
    consecutive minibatches (two fit calls; the optimizer's step count keys the replay) must differ."""
    with ml.ops.Session(numeric_mode="identical") as c:
        hd = c.tensor(np.zeros((2, 8), np.float32))
        xd = c.tensor(np.zeros((2, 8), np.float32))
        yd = c.dropout(xd, 0.5, seed=17, counter=0)
        dcell = c.capture(inputs=[hd, xd], outputs=[c.add(hd, yd), yd])
        xin = c.tensor(np.zeros((2, 2, 8), np.float32))
        w = c.tensor(np.zeros((2, 2, 8), np.float32))
        ys, _ = c.scan(dcell, c.permute(xin, (1, 0, 2)), c.tensor(np.zeros((2, 8), np.float32)))
        prog = c.capture(inputs=[xin], outputs=[c.sum(c.mul(ys, w))])
        opt = c.optimizer("sgd", lr=1.0)
        ones = np.ones((2, 2, 8), np.float32)
        prog.fit(ones, optimizer=opt, parameters=[w], loss="output", epochs=1, batch_size=2, shuffle=False, seed=0)
        w1 = np.asarray(c.numpy(w))
        prog.fit(ones, optimizer=opt, parameters=[w], loss="output", epochs=1, batch_size=2, shuffle=False, seed=0)
        w2 = np.asarray(c.numpy(w))
    first, second = -w1, w1 - w2
    if np.array_equal(first, second):
        raise ValueError("ops-composed-tagger: Program.fit drew the same scan dropout masks in two consecutive "
                         "minibatches (the replay did not advance the scan's key)")
    return first, second


@lane("ops-composed-tagger")
def _(ml, X, yc, yr, Xh=None):
    """The composed tagger (lane nn-blocks-composed, 2026-11-25): embedding + per-step features, a captured cell with
    a dropout inside it under Session.scan with lengths, the weighted per-step masked cross-entropy (padding label
    3), and AdamW with the embedding, cell and head in three parameter groups. Eager: two steps (scan counter = the
    step), each checked against the mask contract (zero gradient on padded steps and the padding row, the loss the
    mean over the real positions); then the parameters, optimizer_state_dict and step counter are saved to a file,
    one more step runs uninterrupted, and a fresh session loads the save and runs that step: parameters and
    optimizer state must be the same bytes. Program: the tagger captured and trained by Program.fit for two
    minibatches, saved with its optimizer state, one more uninterrupted minibatch against a fresh session's
    load_program + load_optimizer_state_dict + one minibatch: the same bytes. Keys: masks of counter 0 and 1 must
    differ, and Program.fit must draw new masks in consecutive minibatches. Parts: train (losses, logits, every
    gradient, parameters after each step), resume (parameters and moments after the resumed step), program (loss
    curve, parameters, resumed parameters and moments), keys (the masks)."""
    B, T = _NN_COMPOSED_B, _NN_COMPOSED_T
    names = _NN_COMPOSED_NAMES
    init = _nn_composed_init()
    data = _nn_composed_data()
    ids, feats, lens, codes = data
    parts = {}
    train = []
    with tempfile.TemporaryDirectory(prefix="ib-ops-composed-") as tmp:
        # ---- eager: two steps, save, one uninterrupted step
        with ml.ops.Session(numeric_mode="identical") as c:
            cw = {k: c.tensor(init[k]) for k in ("w", "u", "b")}
            cell = _nn_composed_cell(c, cw["w"], cw["u"], cw["b"])
            with ml.ops.Session(record=True, numeric_mode="identical") as s:
                p = {k: s.tensor(init[k], requires_grad=True) for k in names}  # glue: six parameters
                opt = _nn_composed_optimizer(s, p)
                for step in range(2):  # driver-loop: over the two training steps
                    lv, lg, grads, gfeats = _nn_composed_eager_step(ml, s, cell, p, data, step)
                    _nn_composed_check_mask(lv, lg, grads, gfeats, lens, codes)
                    s.step(opt, [p[k] for k in names])
                    train += [lv, lg] + [grads[k] for k in names] + [np.asarray(s.numpy(p[k])) for k in names]
                    s.reset()
                saved_params = _nn_composed_saved(tmp, "eager_params", [np.asarray(s.numpy(p[k])) for k in names])
                saved = _nn_composed_save_state(s, opt)
                moments = _nn_composed_saved(tmp, "eager_moments", _nn_composed_moments(saved))
                saved_step = 2
                _nn_composed_eager_step(ml, s, cell, p, data, saved_step)
                s.step(opt, [p[k] for k in names])
                straight = [np.asarray(s.numpy(p[k])) for k in names]
                straight_state = _nn_composed_moments(_nn_composed_save_state(s, opt))
        # ---- eager resume in fresh sessions
        k = 0
        for key in sorted(saved["state"]):  # driver-loop: over the saved parameters' states
            for n in _NN_COMPOSED_STATE:  # small-loop(two moments)
                saved["state"][key][n] = moments[k]
                k += 1
        with ml.ops.Session(numeric_mode="identical") as c:
            cw = {k: c.tensor(init[k]) for k in ("w", "u", "b")}
            cell = _nn_composed_cell(c, cw["w"], cw["u"], cw["b"])
            with ml.ops.Session(record=True, numeric_mode="identical") as r:
                rp = {k: r.tensor(a, requires_grad=True) for k, a in zip(names, saved_params)}
                ropt = _nn_composed_optimizer(r, rp)
                _nn_composed_load_state(r, ropt, saved)
                _nn_composed_eager_step(ml, r, cell, rp, data, saved_step)
                r.step(ropt, [rp[k] for k in names])
                resumed = [np.asarray(r.numpy(rp[k])) for k in names]
                resumed_state = _nn_composed_moments(_nn_composed_save_state(r, ropt))
        for i, name in enumerate(names):  # driver-loop: over the six parameters
            _same_bytes("uninterrupted step 3 " + name, straight[i], "resumed step 3 " + name, resumed[i])
        for i in range(len(straight_state)):  # driver-loop: over the moments
            _same_bytes(f"uninterrupted moment {i}", straight_state[i], f"resumed moment {i}", resumed_state[i])
        parts["train"] = _h(*train)
        parts["resume"] = _h(*resumed, *resumed_state)
        # ---- Program.fit: two minibatches, save, one uninterrupted minibatch
        fit_args = dict(loss="output", batch_size=B, shuffle=False, seed=0)
        with ml.ops.Session(numeric_mode="identical") as c:
            cp = {k: c.tensor(init[k]) for k in names}
            cell = _nn_composed_cell(c, cp["w"], cp["u"], cp["b"])
            xi = c.ids(np.zeros((B, T), np.int32))
            xf = c.tensor(np.zeros((B, T, _NN_COMPOSED_F), np.float32))
            xl = c.tensor(np.zeros((B, 1), np.float32))
            xc = c.tensor(np.zeros((B, T), np.float32))
            loss, _ = _nn_composed_forward(c, cell, xi, xf, xl, xc, cp, None, 0)
            prog = c.capture(inputs=[xi, xf, xl, xc], outputs=[loss])
            index = [list(prog.params).index(cp[k]) for k in names]  # glue: six parameter positions
            opt = _nn_composed_optimizer(c, cp)
            trained = [cp[k] for k in names]
            prog.fit(ids, feats, lens, codes, optimizer=opt, parameters=trained, epochs=2, **fit_args)
            curve = np.asarray(prog.loss_curve_, dtype=np.float32)
            fitted = [np.asarray(c.numpy(cp[k])) for k in names]
            psaved = _nn_composed_save_state(c, opt)
            pmoments = _nn_composed_saved(tmp, "program_moments", _nn_composed_moments(psaved))
            path = os.path.join(tmp, "tagger.mlp")
            prog.save(path)
            prog.fit(ids, feats, lens, codes, optimizer=opt, parameters=trained, epochs=1, **fit_args)
            pstraight = [np.asarray(c.numpy(cp[k])) for k in names]
            pstraight_state = _nn_composed_moments(_nn_composed_save_state(c, opt))
        k = 0
        for key in sorted(psaved["state"]):  # driver-loop: over the saved parameters' states
            for n in _NN_COMPOSED_STATE:  # small-loop(two moments)
                psaved["state"][key][n] = pmoments[k]
                k += 1
        with ml.ops.Session(numeric_mode="identical") as d:
            loaded = d.load_program(path)
            lp = {k: loaded.params[index[i]] for i, k in enumerate(names)}  # glue: six handles
            for i, name in enumerate(names):  # driver-loop: over the six parameters
                _same_bytes("Program.fit saved " + name, fitted[i], "load_program " + name,
                            np.asarray(d.numpy(lp[name])))
            dopt = _nn_composed_optimizer(d, lp)
            _nn_composed_load_state(d, dopt, psaved)
            loaded.fit(ids, feats, lens, codes, optimizer=dopt, parameters=[lp[k] for k in names], epochs=1,
                       **fit_args)
            presumed = [np.asarray(d.numpy(lp[k])) for k in names]
            presumed_state = _nn_composed_moments(_nn_composed_save_state(d, dopt))
        for i, name in enumerate(names):  # driver-loop: over the six parameters
            _same_bytes("Program.fit uninterrupted " + name, pstraight[i], "Program.fit resumed " + name, presumed[i])
        for i in range(len(pstraight_state)):  # driver-loop: over the moments
            _same_bytes(f"Program.fit uninterrupted moment {i}", pstraight_state[i],
                        f"Program.fit resumed moment {i}", presumed_state[i])
        parts["program"] = _h(curve, *fitted, *presumed, *presumed_state)
    parts["keys"] = _h(*_nn_composed_key_masks(ml), *_nn_composed_program_key(ml))
    return _fit(parts)


_batch_decl("n/a:training-step (seeded training steps and a resume through the ops session and Program.fit; "
            "digested whole)", "ops-composed-tagger")
