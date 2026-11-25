# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE PROGRAMS', OPS SESSION'S, CONTAINERS' AND MODELS' ENDPOINT LANES (lane vdv-cov-nn, 2026-11-24).
#
# Verify through `device=` (docs/plans/VERIFY_THROUGH_DEVICE.md, item 3): the `ops.Session` block,
# recurrent, convolution and language-model operations no lane called, `Program.decode`, an
# imported ONNX program's run / run_dynamic / profile / inspect / save, `Array`'s numeric methods,
# `models.CausalLM`'s call, parameters, state dict and state reset with a generation stream's
# info and state, and the BPE tokenizer's decode and decode_batch.
#
# Every lane is device-agnostic (contract 1): no `device=`; the harness's view gives `ops.Session`,
# `import_model`, the model loaders and the tokenizer the side. Objects a loader returns (an
# imported program, a CausalLM, a tokenizer from its token bytes) are not typed by the coverage
# audit's receiver walk, so their methods are called through the class (`IP.run(prog, x)`,
# `CLM.parameters(m)`): the same call, spelled so the audit sees the endpoint by name.
#
# Loaded by tools/identity_break.py `_load_lane_fragments` in the harness namespace: np, _h, _fit,
# _same_bytes, _block_weights, _seq, _ids, _hw, _synthetic_bpe, _coverage_onnx,
# _NN_ENDPOINTS_MAMBA_SHAPES, _one_sided and `lane` are the harness's or an earlier fragment's,
# read when a lane runs; helpers here are prefixed `_nn_programs_`. Each lane builds its own small
# inputs and objects and shares nothing mutable with another lane. Every reference is OWED.


def _nn_programs_f32(a):
    return np.ascontiguousarray(a, dtype=np.float32)


# ---------------------------------------------------------------- ops.Session

@lane("nn-pg-session-blocks")
def _(ml, X, yc, yr, Xh=None):
    """ops.Session's model operations on fixture data: `mamba_block` + `mamba_step` (version 1, two
    sequences, the `mamba1` lane's shapes) held equal to Mamba1Block.forward on the same weights
    (the documented claim); `attention_block` + `attention_step` (2 heads, 1 kv head); `small_mlp`;
    `lm_head_loss`; `lstm`, `gru`, `rnn` over (4, 8, 6) time-first sequences; `conv_block` (padding
    1, pool 2) then `conv_linear`; `lm_embedding`; `lm_rows` / `lm_pick` / `lm_alive` (greedy, eos
    3); `state` (read back unchanged). Parts: every output read back."""
    shapes1 = _NN_ENDPOINTS_MAMBA_SHAPES["mamba1"]
    w1 = _block_weights("nn-pg-mamba1", shapes1[0], ones=shapes1[1])
    tshapes = {"input_layernorm.weight": (32,), "post_attention_layernorm.weight": (32,),
               "q_proj.weight": (32, 32), "k_proj.weight": (16, 32), "v_proj.weight": (16, 32),
               "o_proj.weight": (32, 32), "gate_proj.weight": (64, 32), "up_proj.weight": (64, 32),
               "down_proj.weight": (32, 64)}
    wt = _block_weights("nn-pg-transformer", tshapes,
                        near_one=("input_layernorm.weight", "post_attention_layernorm.weight"))
    x = _seq(X, 2, 16, 32)
    reference = np.asarray(ml.Mamba1Block(w1).forward(x))
    parts = {}
    with ml.ops.Session() as s:
        mb = s.mamba_block([w1[k] for k in shapes1[0]], version=1, batch_size=2)
        ym = s.numpy(s.mamba_step(s.tensor(x), mb))
        _same_bytes("Session.mamba_step", ym, "Mamba1Block.forward", reference)
        ab = s.attention_block([wt[k] for k in tshapes], n_heads=2, n_kv_heads=1, max_positions=32, batch_size=2)
        ya = s.numpy(s.attention_step(s.tensor(x), ab))
        rows = _nn_programs_f32(X[:32, :8])
        mlp = s.numpy(s.small_mlp(s.tensor(rows), s.tensor(_hw((16, 8), "nn-pg:w1", -0.5, 0.5)),
                                  s.tensor(_hw((16,), "nn-pg:b1", -0.1, 0.1)),
                                  s.tensor(_hw((3, 16), "nn-pg:w2", -0.5, 0.5)),
                                  s.tensor(_hw((3,), "nn-pg:b2", -0.1, 0.1))))
        codes = (_ids(X, 32, 1) % 16).astype(np.int32)
        head = s.numpy(s.lm_head_loss(s.tensor(rows), s.tensor(_hw((16, 8), "nn-pg:head", -0.5, 0.5)),
                                      s.tensor(codes.astype(np.float32))))  # class values (lm_head_v2 refuses int words)
        seqx = s.tensor(_seq(X, 4, 8, 6, skip=2048))
        rec = []
        for gates, op in ((4, s.lstm), (3, s.gru), (1, s.rnn)):
            h = 5
            rec.append(s.numpy(op(seqx, s.tensor(_hw((gates * h, 6), f"nn-pg:ih{gates}", -0.4, 0.4)),
                                  s.tensor(_hw((gates * h, h), f"nn-pg:hh{gates}", -0.4, 0.4)),
                                  s.tensor(_hw((1, gates * h), f"nn-pg:bi{gates}", -0.1, 0.1)),
                                  s.tensor(_hw((1, gates * h), f"nn-pg:bh{gates}", -0.1, 0.1)))))
        img = s.tensor(_nn_programs_f32(X[:8, :16]).reshape(8, 1, 4, 4))
        block = s.conv_block(img, s.tensor(_hw((3, 1, 3, 3), "nn-pg:cw", -0.5, 0.5)),
                             s.tensor(_hw((3,), "nn-pg:cb", -0.1, 0.1)), padding=1, pool_size=2)
        conv = s.numpy(block)
        dense = s.numpy(s.conv_linear(block, s.tensor(_hw((2, 12), "nn-pg:lw", -0.5, 0.5)),
                                      s.tensor(_hw((2,), "nn-pg:lb", -0.1, 0.1))))
        table = s.tensor(_hw((16, 8), "nn-pg:table", -0.25, 0.25))
        emb = s.numpy(s.lm_embedding(table, s.ids((_ids(X, 2, 4) % 16).astype(np.int32)), padding_idx=3))
        gen_rows = s.lm_rows(2, eos_token_id=3)
        picked = s.lm_pick(s.tensor(_nn_programs_f32(X[:2, :16])), gen_rows)
        alive = s.numpy(s.lm_alive(picked, gen_rows))
        carried = _nn_programs_f32(X[40:42, :4])
        st = s.numpy(s.state(carried))
        _same_bytes("Session.state read back", np.asarray(st), "the array given", carried)
        parts.update(mamba=_h(ym), attention=_h(ya), small_mlp=_h(mlp), lm_head_loss=_h(head),
                     recurrent=_h(*rec), conv=_h(conv, dense), lm_embedding=_h(emb),
                     lm_pick=_h(np.asarray(s.numpy(picked)), np.asarray(alive)))
    return _fit(parts)


@lane("nn-pg-program-decode")
def _(ml, X, yc, yr, Xh=None):
    """ops.Program.decode: a captured step (the ids' rows of a hashed (8, 8) table, `argmax` over the
    features as the next ids) run 12 steps on the device from two fixture-derived first tokens; the
    same step run 12 times with Program.run must give the same tokens. Parts: the decoded tokens."""
    table = _hw((8, 8), "nn-pg:decode-table", -1.0, 1.0)
    first = (_ids(X, 2, 1) % 8).astype(np.int32)
    with ml.ops.Session() as s:
        ids = s.ids(first)
        # keepdims=True: `first` is (2, 1) ids and the embedding has one row per id, so the argmax must keep
        # its axis for the state to keep its rank (the while_loop contract, python/mojolearn/tests/
        # test_program_decode.py; without it decode refuses "must preserve each state tensor's rank").
        nxt = s.argmax(s.embedding(s.tensor(table), ids), axis=-1, keepdims=True, int32=True)
        step = s.capture([ids], [nxt])
        out = step.decode(first, n=12)
        tokens = np.asarray(out["tokens"])
        loop, cur = [], first
        for _ in range(12):  # driver-loop: over decode steps (the independent replay of decode)
            # Program.run returns float32 words; int32 outputs are read with .view (ops.Session.argmax docs)
            cur = np.ascontiguousarray(np.asarray(step.run(cur), dtype=np.float32).view(np.int32)).reshape(first.shape)
            loop.append(cur.reshape(-1))
        _same_bytes("Program.decode tokens", tokens.reshape(-1), "Program.run x12 tokens", np.concatenate(loop))
    return _fit(dict(tokens=_h(tokens)))


# ---------------------------------------------------------------- an imported program

@lane("nn-pg-imported-program")
def _(ml, X, yc, yr, Xh=None):
    """An ImportedProgram (the `sequence-onnx-import` lane's two-layer ONNX file, batch 64):
    `run`, `run_dynamic` and `profile` (its outputs; the timings are not numbers of the model) must
    give the same output; `inspect` (called: graph metadata); `save` and Session.load_program on the
    same side, whose `run` must equal. Parts: the output and the held-out output.

    The class is imported from its module (FIX w3-fix-13, 2026-11-25): `ml.importers._program` read a private
    submodule as an attribute of the package, which exists only once something has imported it; from the installed
    wheel nothing had, and `mojolearn.importers.__getattr__` (it resolves only `import_model`) raised AttributeError,
    so the lane was REFUSED. Harness defect."""
    from mojolearn.importers._program import ImportedProgram as IP
    x = _nn_programs_f32(X[:64, :6])
    xh = _nn_programs_f32((X if Xh is None else Xh)[:64, :6])
    with tempfile.TemporaryDirectory(prefix="ib-nn-pg-import-") as tmp:
        path = _coverage_onnx(os.path.join(tmp, "model.onnx"), X)
        prog = ml.import_model(path, shapes={"x": (64, 6)}, numeric_mode="identical")
        if not isinstance(prog, IP):
            raise AssertionError(f"import_model returned {type(prog).__name__}, not ImportedProgram")
        y = np.asarray(IP.run(prog, x))
        held = np.asarray(IP.run(prog, xh))
        dyn = IP.run_dynamic(prog, x)
        _same_bytes("ImportedProgram.run", y, "ImportedProgram.run_dynamic",
                    np.asarray(dyn[0] if isinstance(dyn, (list, tuple)) else dyn))
        IP.profile(prog, x)
        IP.inspect(prog)
        saved = os.path.join(tmp, "program.mlp")
        IP.save(prog, saved)
        with ml.ops.Session() as s:
            again = s.load_program(saved)
            out = again.run(x)
            _same_bytes("ImportedProgram.run", y, "Session.load_program(save).run",
                        np.asarray(out[0] if isinstance(out, (list, tuple)) else out))
    return _fit(dict(out=_h(y), held=_h(held)))


# ---------------------------------------------------------------- Array

@lane("nn-pg-array")
def _(ml, X, yc, yr, Xh=None):
    """mojolearn.Array's numeric methods on a (4, 3) float32 array filled from the fixture:
    `copy`, `min`, `argmax`, `__getitem__` (a row and a scalar), `__len__`, `__iter__`, `__eq__` /
    `__ne__` / `__bool__`, `__array__`, `__array_interface__`, the buffer protocol, `flags`, and
    `from_list` of the same values (must equal). Parts: the values read every way, the reductions,
    the comparisons."""
    a = ml.Array((4, 3))
    np.asarray(a)[...] = _nn_programs_f32(X[:4, :3])
    values = np.asarray(a).copy()
    b = a.copy()
    rows = [np.asarray(a.__getitem__(i)) for i in range(a.__len__())]
    walked = [np.asarray(r) for r in a.__iter__()]
    eq = a.__eq__(b)
    ne = a.__ne__(b)
    one = ml.Array.from_list([[float(values[0, 0])]], "<f4")
    flags = a.flags
    listed = ml.Array.from_list(values.tolist(), "<f4")
    _same_bytes("Array", values, "Array.from_list of its values", np.asarray(listed))
    interface = a.__array_interface__
    buf = a.__buffer__(0) if hasattr(a, "__buffer__") else memoryview(a)
    raw = np.frombuffer(bytes(buf), dtype=np.float32)
    _same_bytes("Array bytes", values.reshape(-1), "Array buffer", raw)
    reductions = np.asarray([a.min(), a.argmax(), float(a.__getitem__((1, 2)))], dtype=np.float64)
    flags_v = np.asarray([int(bool(flags["C_CONTIGUOUS"])), int(bool(flags["F_CONTIGUOUS"])),
                          int(bool(flags["WRITEABLE"])), int(tuple(interface["shape"]) == (4, 3))], dtype=np.int64)
    compare = np.asarray([int(bool(np.all(np.asarray(eq)))), int(bool(np.any(np.asarray(ne)))),
                          int(one.__bool__())], dtype=np.int64)
    return _fit(dict(values=_h(np.asarray(b), np.asarray(a.__array__()), *rows, *walked),
                     reductions=_h(reductions), flags=_h(flags_v, compare)))


_one_sided("nn-pg-array", missing="gpu",
           reason="Array is the host container type (NON_COMPUTING in the device view); its reductions and "
                  "comparisons run on the host only")


# ---------------------------------------------------------------- models.CausalLM

@lane("nn-pg-causal-lm")
def _(ml, X, yc, yr, Xh=None):
    """models.CausalLM on the package's synthetic Llama checkpoint (fp32_v1): `__call__` on (2, 8)
    fixture ids; `parameters` and `state_dict`; `load_state_dict` of its
    state into the model of another seed's checkpoint (its call must equal); `reset_state` (a
    forward into a state, a reset, the same forward again must equal the first); `stream` of four
    greedy tokens: the GenerationStream's `info` and `state_dict` numbers. Parts: the logits, the
    parameters, the stream."""
    from mojolearn import _causal_lm_fixtures as fx
    CLM = ml.models.CausalLM
    GS = ml.models.GenerationStream
    cfg = fx._llama_config()
    ids = np.ascontiguousarray(_ids(X, 2, 8) % 64, dtype=np.int32)
    with tempfile.TemporaryDirectory(prefix="ib-nn-pg-causal-lm-") as tmp:
        root = fx._write_checkpoint(os.path.join(tmp, "a"), cfg, fx._llama_tensors(cfg, seed=1))
        other = fx._write_checkpoint(os.path.join(tmp, "b"), cfg, fx._llama_tensors(cfg, seed=7))
        m = CLM.load(root, numeric_profile="fp32_v1")
        m2 = CLM.load(other, numeric_profile="fp32_v1")
        A = ml.Array.from_buffer
        # `m(...)`, never `CLM.__call__(m, ...)`: CLM is the device view's BoundClass, whose own `__call__`
        # CONSTRUCTS a CausalLM (python/mojolearn/_verify_device.py BoundClass.__call__), so the old spelling built a
        # model from (m, ids) and the constructor refused it ("model operation requires its connected record
        # class"). Harness defect (FIX w3-fix-13, 2026-11-25).
        y = np.asarray(m(A(ids)))
        sd = m.state_dict()
        flat = [np.ascontiguousarray(np.asarray(p)) for p in CLM.parameters(m)]
        CLM.load_state_dict(m2, sd)
        _same_bytes("CausalLM call", y, "CausalLM call after load_state_dict", np.asarray(m2(A(ids))))
        state = m.allocate_state(2, 8)
        first = np.asarray(m.forward(A(ids), state))
        CLM.reset_state(m, state)
        _same_bytes("forward into a state", first, "forward after reset_state", np.asarray(m.forward(A(ids), state)))
        stream = m.stream(A(ids), 4)
        tokens = np.asarray(stream.read(4))
        info = GS.info(stream)
        snapshot = GS.state_dict(stream)
    numbers = []
    for part in (info, snapshot):
        for k in sorted(part, key=str):
            v = part[k]
            if isinstance(v, (bool, int, float, np.integer, np.floating)):
                numbers.append(float(v))
    return _fit(dict(logits=_h(y), parameters=_h(*flat), state=_h(*[np.asarray(sd[k]) for k in sorted(sd)]),
                     stream=_h(tokens, np.asarray(numbers, dtype=np.float64))))


# ---------------------------------------------------------------- the BPE tokenizer

@lane("nn-pg-bpe-decode")
def _(ml, X, yc, yr, Xh=None):
    """BpeTokenizer (the synthetic 512-rank vocabulary) `decode` and `decode_batch` of the ids of
    printable text cut from the fixture bytes: the decoded text must be the text encoded, and the
    batch must equal the single decodes. Parts: the ids and the decoded bytes."""
    BT = ml.BpeTokenizer
    tok = _synthetic_bpe(ml)
    raw = np.frombuffer(np.ascontiguousarray(X).tobytes()[:192], dtype=np.uint8)
    texts = ["".join(chr(32 + int(b) % 95) for b in raw[64 * k:64 * (k + 1)]) for k in range(3)]
    ids = [tok.encode(t) for t in texts]
    single = [BT.decode(tok, i) for i in ids]
    batch = BT.decode_batch(tok, ids)
    for k, (t, d, b) in enumerate(zip(texts, single, batch)):
        _same_bytes(f"text {k}", np.frombuffer(t.encode(), dtype=np.uint8), f"decode(encode(text {k}))",
                    np.frombuffer(d.encode(), dtype=np.uint8))
        _same_bytes(f"decode {k}", np.frombuffer(d.encode(), dtype=np.uint8), f"decode_batch[{k}]",
                    np.frombuffer(b.encode(), dtype=np.uint8))
    return _fit(dict(ids=_h(*[np.asarray(i, dtype=np.int64) for i in ids]),
                     decoded=_h(np.frombuffer("".join(single).encode(), dtype=np.uint8))))


# ---------------------------------------------------------------- the whole-sequence per-step model (lane rnn-parity-B)

@lane("ops-per-step-weighted-embedding")
def _(ml, X, yc, yr, Xh=None):
    """ops.Session's per-step weighted model (lane rnn-parity-B, 2026-11-25), the user's whole-flight GRU: an
    embedding of each sequence's category (4 sequences, 3 categories, width 2) reshaped to (1, batch, 2),
    expanded over the 8 steps and concatenated on the last axis with 3 fixture features a step, a GRU (hidden 5)
    with lengths 8 5 3 6 (padded steps zeroed and carried), a linear head on every step, bce_with_logits at every
    step weighted 0 on padded steps and 30 on positives, backward and one Adam step. Parts: the loss, every
    parameter's gradient and every updated parameter. Plus the recurrent options one seeded forward each:
    a bidirectional GRU from a given h_0 (the forward and reverse outputs joined), and an LSTM run one step at a
    time carrying (h_n, c_n) with return_state, digested with the whole-sequence call (inputs 5 <= 64: the same
    words, checked here). Sabotage arms coverage-autograd-rnn_parity_loss_weight_gpu / _cpu drop the weight
    from the loss numerator on one side: the loss, gradient and parameter parts move."""
    T, B, F, E, H, K = 8, 4, 3, 2, 5, 3
    x = _seq(X, T, B, F)
    lengths = np.array([8, 5, 3, 6], dtype=np.float32)
    category = (_ids(X, 1, B) % K).astype(np.int64).reshape(B)
    y = (_seq(X, T, B, 1, skip=4096) > 0.6).astype(np.float32)
    valid = (np.arange(T)[:, None] < lengths[None, :]).astype(np.float32)[:, :, None]
    w = (valid * np.where(y > 0, 30.0, 1.0)).astype(np.float32)
    init = {"table": _hw((K, E), "ops-psw:table", -0.3, 0.3), "w_ih": _hw((3 * H, F + E), "ops-psw:ih", -0.4, 0.4),
            "w_hh": _hw((3 * H, H), "ops-psw:hh", -0.4, 0.4), "b_ih": _hw((1, 3 * H), "ops-psw:bi", -0.1, 0.1),
            "b_hh": _hw((1, 3 * H), "ops-psw:bh", -0.1, 0.1), "w_out": _hw((H, 1), "ops-psw:wo", -0.5, 0.5),
            "b_out": _hw((1, 1), "ops-psw:bo", -0.1, 0.1)}
    parts = {}
    with ml.ops.Session(record=True, numeric_mode="identical") as s:
        p = {k: s.tensor(v, requires_grad=True) for k, v in init.items()}  # glue: seven parameters
        e = s.expand(s.reshape(s.embedding(p["table"], category), (1, B, E)), (T, B, E))
        h = s.gru(s.concat([s.tensor(x), e], axis=2), p["w_ih"], p["w_hh"], p["b_ih"], p["b_hh"],
                  lengths=s.tensor(lengths))
        logits = s.linear(h, p["w_out"], p["b_out"])
        loss = s.bce_with_logits(logits, s.tensor(y), weight=s.tensor(w))
        parts["loss"] = _h(np.asarray(s.numpy(loss)), np.asarray(s.numpy(logits)))
        s.backward(loss)
        parts["grad"] = _h(*[np.asarray(s.grad(p[k])) for k in init])  # glue: one array per parameter
        s.step(s.optimizer("adam", lr=0.01), [p[k] for k in init])
        parts["param"] = _h(*[np.asarray(s.numpy(p[k])) for k in init])  # glue: one array per parameter
    with ml.ops.Session(numeric_mode="identical") as s:
        xs = s.tensor(_seq(X, T, B, H, skip=8192))
        h0 = s.tensor(_hw((B, H), "ops-psw:h0", -0.5, 0.5))
        gw = [s.tensor(_hw(sh, f"ops-psw:bi{k}", -0.4, 0.4))
              for k, sh in enumerate(((3 * H, H), (3 * H, H), (1, 3 * H), (1, 3 * H),
                                      (3 * H, H), (3 * H, H), (1, 3 * H), (1, 3 * H)))]  # glue: eight weights
        fwd = s.gru(xs, *gw[:4], h_0=h0, lengths=s.tensor(lengths))
        rev = s.gru(xs, *gw[4:], h_0=h0, lengths=s.tensor(lengths), reverse=True)
        parts["bidirectional"] = _h(np.asarray(s.numpy(s.concat([fwd, rev], axis=2))))
        lw = [s.tensor(_hw(sh, f"ops-psw:l{k}", -0.4, 0.4))
              for k, sh in enumerate(((4 * H, H), (4 * H, H), (1, 4 * H), (1, 4 * H)))]  # glue: four weights
        whole, hn, cn = s.lstm(xs, *lw, return_state=True)
        steps, hc, cc = [], None, None
        for t in range(T):  # driver-loop: over time steps (the streaming form under test)
            xt = s.tensor(_seq(X, T, B, H, skip=8192)[t:t + 1])
            out, hc, cc = s.lstm(xt, *lw, h_0=hc, c_0=cc, return_state=True)
            steps.append(np.asarray(s.numpy(out)))
        streamed = np.concatenate(steps, axis=0)
        _same_bytes("Session.lstm one step at a time", streamed, "Session.lstm whole sequence",
                    np.asarray(s.numpy(whole)))
        _same_bytes("Session.lstm streamed c_n", np.asarray(s.numpy(cc)), "Session.lstm c_n", np.asarray(s.numpy(cn)))
        parts["stream"] = _h(streamed, np.asarray(s.numpy(hn)), np.asarray(s.numpy(cn)))
    return _fit(parts)


_batch_decl("n/a:training-step (one seeded training step through the ops session; its forward outputs are "
            "train parts, digested whole)", "ops-per-step-weighted-embedding")


# ---------------------------------------------------------------- the loss fold (lane rnn-parity-c, 2026-11-25)

def _nn_programs_close(name, got, want, rtol=2e-5, atol=1e-6):
    """Hold a float32 result to an independent float64 oracle; raises naming the claim (the cell reads REFUSED)."""
    got = np.asarray(got, dtype=np.float64)
    if got.shape != np.shape(want) or not np.allclose(got, want, rtol=rtol, atol=atol):
        raise AssertionError(f"{name}: {got.ravel()[:4]} differs from the oracle {np.ravel(want)[:4]}")


@lane("ops-loss-fold-num-den")
def _(ml, X, yc, yr, Xh=None):
    """The loss fold (autograd/loss_fold.mojo): a minibatch split into logical shards combines as the sum of
    the numerators over the sum of the denominators, never an average of the shards' means. Case 1, the 2-vs-20
    claim: a captured program with two outputs (the masked loss sum, sum m (x W - y)^2, and its denominator,
    sum m) trains with loss="output" over 4 rows of 11 steps in 2 logical shards, shard A with 2 valid steps and
    shard B with 20; one SGD step must equal W - lr 2 x^T (m (x W - y)) / 22 (float64 oracle; the average of
    the shards' means, 0.5 (g_A / 2 + g_B / 20), is far from it) and the epoch loss (num_A + num_B) / 22. Case 2,
    a built-in loss: mse over 5 rows in shards of 2 and 3 rows must equal the whole-batch mse step (the shards
    weigh 2/5 and 3/5, not 1/2 each). Parts: both trained parameters and both loss curves. The sabotage arms
    coverage-rnn-parity-mean_of_means-gpu / -cpu make one side average the means: this lane reads REFUSED there."""
    T = 11
    x = _hw((4, T), "ops-fold:x", -1.0, 1.0)
    y = _hw((4, T), "ops-fold:y", -1.0, 1.0)
    m = np.zeros((4, T), np.float32)
    m[0, 0] = m[1, 0] = 1.0           # shard A (rows 0-1): 2 valid steps
    m[2, :10] = m[3, :10] = 1.0       # shard B (rows 2-3): 20 valid steps
    w0 = _hw((T, T), "ops-fold:w", -0.3, 0.3)
    lr = 0.05
    parts = {}
    with ml.ops.Session(numeric_mode="identical") as s:
        xi = s.tensor(np.zeros((4, T), np.float32))
        yi = s.tensor(np.zeros((4, T), np.float32))
        mi = s.tensor(np.zeros((4, T), np.float32))
        w = s.tensor(w0)
        err = s.sub(s.matmul(xi, w), yi)
        num = s.sum(s.mul(s.mul(err, err), mi))
        den = s.sum(mi)
        p = s.capture([xi, yi, mi], [num, den])
        opt = s.optimizer("sgd", lr=lr)
        with p.distribute((0,), logical_shards=2, optimizer=opt, seed=0) as d:
            curve = np.asarray(d.fit(x, y, m, loss="output", epochs=1, batch_size=4, shuffle=False))
            trained = np.asarray(d.state()["parameters"][0])
    x64, y64, m64, w64 = (a.astype(np.float64) for a in (x, y, m, w0))
    r = m64 * (x64 @ w64 - y64)
    _nn_programs_close("num/den loss (num_A + num_B) / (den_A + den_B)", curve,
                       [float(np.sum(r * (x64 @ w64 - y64)) / m64.sum())])
    _nn_programs_close("num/den step W - lr dL/dW", trained, w64 - lr * 2.0 * (x64.T @ r) / m64.sum())
    parts.update(num_den_params=_h(trained), num_den_curve=_h(curve))

    x5 = _hw((5, 4), "ops-fold:x5", -1.0, 1.0)
    y5 = _hw((5, 2), "ops-fold:y5", -1.0, 1.0)
    v0 = _hw((4, 2), "ops-fold:v", -0.5, 0.5)
    with ml.ops.Session(numeric_mode="identical") as s:
        xi = s.tensor(np.zeros((5, 4), np.float32))
        v = s.tensor(v0)
        p = s.capture([xi], [s.matmul(xi, v)])
        opt = s.optimizer("sgd", lr=lr)
        with p.distribute((0,), logical_shards=2, optimizer=opt, seed=0) as d:
            curve5 = np.asarray(d.fit(x5, targets=y5, loss="mse", epochs=1, batch_size=5, shuffle=False))
            trained5 = np.asarray(d.state()["parameters"][0])
    xa, ya, va = (a.astype(np.float64) for a in (x5, y5, v0))
    e5 = xa @ va - ya
    _nn_programs_close("mse over shards of 2 and 3 rows", curve5, [float(np.mean(e5 * e5))])
    _nn_programs_close("mse step over shards of 2 and 3 rows", trained5, va - lr * 2.0 * (xa.T @ e5) / e5.size)
    parts.update(mse_params=_h(trained5), mse_curve=_h(curve5))
    return _fit(parts)


_batch_decl("n/a:program-endpoints (session operations, captured and imported programs, containers, model "
            "and tokenizer round trips; outputs digested whole)",
            "nn-pg-session-blocks", "nn-pg-program-decode", "nn-pg-imported-program", "nn-pg-array",
            "nn-pg-causal-lm", "nn-pg-bpe-decode", "ops-loss-fold-num-den")
