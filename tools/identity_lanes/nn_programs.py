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
        nxt = s.argmax(s.embedding(s.tensor(table), ids), axis=-1, int32=True)
        step = s.capture([ids], [nxt])
        out = step.decode(first, n=12)
        tokens = np.asarray(out["tokens"])
        loop, cur = [], first
        for _ in range(12):  # driver-loop: over decode steps (the independent replay of decode)
            cur = np.ascontiguousarray(np.asarray(step.run(cur)), dtype=np.int32).reshape(first.shape)
            loop.append(cur.reshape(-1))
        _same_bytes("Program.decode tokens", tokens.reshape(-1), "Program.run x12 tokens", np.concatenate(loop))
    return _fit(dict(tokens=_h(tokens)))


# ---------------------------------------------------------------- an imported program

@lane("nn-pg-imported-program")
def _(ml, X, yc, yr, Xh=None):
    """An ImportedProgram (the `sequence-onnx-import` lane's two-layer ONNX file, batch 64):
    `run`, `run_dynamic` and `profile` (its outputs; the timings are not numbers of the model) must
    give the same output; `inspect` (called: graph metadata); `save` and Session.load_program on the
    same side, whose `run` must equal. Parts: the output and the held-out output."""
    IP = ml.importers._program.ImportedProgram
    x = _nn_programs_f32(X[:64, :6])
    xh = _nn_programs_f32((X if Xh is None else Xh)[:64, :6])
    with tempfile.TemporaryDirectory(prefix="ib-nn-pg-import-") as tmp:
        path = _coverage_onnx(os.path.join(tmp, "model.onnx"), X)
        prog = ml.import_model(path, shapes={"x": (64, 6)}, numeric_mode="identical")
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
        y = np.asarray(CLM.__call__(m, A(ids)))
        sd = m.state_dict()
        flat = [np.ascontiguousarray(np.asarray(p)) for p in CLM.parameters(m)]
        CLM.load_state_dict(m2, sd)
        _same_bytes("CausalLM call", y, "CausalLM call after load_state_dict", np.asarray(CLM.__call__(m2, A(ids))))
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


_batch_decl("n/a:program-endpoints (session operations, captured and imported programs, containers, model "
            "and tokenizer round trips; outputs digested whole)",
            "nn-pg-session-blocks", "nn-pg-program-decode", "nn-pg-imported-program", "nn-pg-array",
            "nn-pg-causal-lm", "nn-pg-bpe-decode")
