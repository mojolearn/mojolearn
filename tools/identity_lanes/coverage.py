# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE VERIFIER COVERAGE AUDIT'S LANES (lane verify-coverage-audit, 2026-10-21).
#
# docs/plans/VERIFY_COVERAGE_20261021.md lists every native entry and public
# method that is new or changed since the last published release, with the
# verifier lanes that reach it (tools/verify_coverage_audit.py). Each lane in
# this file reaches one group the audit found no lane for, through the public
# door a user calls. They read OWED until a release record admits them.
#
# Every part is the FNV-1a64 identity digest of one returned host array,
# computed in Mojo (autograd/model_verify.mojo verify_digest through the
# x_sequence binding, the digest the sequence-tape lanes and `verify-model`
# use). Python only passes buffers. Inputs are the fixture's own rows, so a
# lane is deterministic on every fixture. Sizes are correctness-check sizes:
# each lane runs in seconds.
#
# tools/identity_break.py executes this file in ITS OWN namespace after every
# helper and registry exists (`_load_lane_fragments`): spell every lane name
# literally, add only these lanes, rebind no existing name, and prefix every
# helper with `_coverage_`.


def _coverage_binding(ml):
    return ml._backend.binding("_mojolearn_x_sequence", "identical")


def _coverage_digest(B, a):
    a = np.ascontiguousarray(a)
    # cpu-route: verification digest of a returned host array, after the run
    return str(B.verify_digest(a.ctypes.data, a.nbytes))


def _coverage_digester(ml):
    B = _coverage_binding(ml)
    return lambda a: _coverage_digest(B, a)


# ---------------------------------------------------------------- Gaussian process: the likelihood at a caller's theta

@lane("gp-lml-theta")
def _(ml, X, yc, yr, Xh=None):
    """GaussianProcessRegressor.log_marginal_likelihood(theta, eval_gradient)
    at a theta the caller passes: the `gp` lane's kernel and rows, fitted, then
    the likelihood and its gradient at the fitted theta and at two shifted
    ones, and the likelihood alone at a third. This is the path
    `gp_theta_params` (the kernel's free values from theta) and `gpr_lml_grad`
    (the factorization and the gradient) take; no other lane asks for a
    likelihood at a theta of its own."""
    d = _coverage_digester(ml)
    k = ml.ConstantKernel(1.0) * ml.RBF(1.0) + ml.WhiteKernel(0.1)
    m = ml.GaussianProcessRegressor(kernel=k).fit(X[:256, :4], yr[:256])
    theta = np.asarray(m.kernel_.theta, dtype=np.float64)
    parts = dict(theta=d(theta))
    for i, shift in enumerate((0.0, 0.25, -0.5)):  # glue: three probe points
        lml, grad = m.log_marginal_likelihood((theta + shift).tolist(), eval_gradient=True)
        parts[f"lml{i}"] = d(np.asarray([lml], dtype=np.float64))
        parts[f"grad{i}"] = d(np.asarray(grad, dtype=np.float64))
    parts["lml_only"] = d(np.asarray([m.log_marginal_likelihood((theta - 0.125).tolist())], dtype=np.float64))
    return _fit(parts)


# ---------------------------------------------------------------- the ordered fold of replicated training

@lane("cross-vendor-fold")
def _(ml, X, yc, yr, Xh=None):
    """mojolearn.cross_vendor's fold: five float32 shard buffers of 2048 x 8
    fixture values folded in logical order (`ordered_fold`), the same fold
    resumed after a two-shard prefix, and one `fold_pair`. The resumed fold
    is held byte for byte to the whole one in the cell, which is the
    function's own claim. The fold runs in the host binding on every install
    (cross_vendor_call, with core's fold_pair_f32)."""
    import mojolearn.cross_vendor as cv
    d = _coverage_digester(ml)
    flat = np.ascontiguousarray(X[:10240, :8], dtype=np.float32).reshape(5, -1)
    raw = [flat[k].tobytes() for k in range(5)]  # glue: five shard buffers
    whole = np.frombuffer(cv.ordered_fold(raw), dtype=np.float32)
    prefix = cv.ordered_fold(raw[:2])
    resumed = np.frombuffer(cv.ordered_fold(raw[2:], prefix=prefix), dtype=np.float32)
    _same_bytes("ordered_fold(all)", whole, "ordered_fold(rest, prefix=first two)", resumed)
    pair = np.frombuffer(cv.fold_pair(raw[0], raw[1]), dtype=np.float32)
    return _fit(dict(fold=d(whole), prefix=d(np.frombuffer(prefix, dtype=np.float32)), pair=d(pair)))


# ---------------------------------------------------------------- the MoE block's backward pass

@lane("sequence-moe-backward")
def _(ml, X, yc, yr, Xh=None):
    """MoEBlock.backward on the `sequence-moe` lane's two blocks (top-2 of 6
    experts; top-3 of 5 without renormalization, its router tied on two
    experts) for 256 fixture rows and an upstream gradient from the next 256:
    the input gradient and the router, gate/up and down projection gradients."""
    d = _coverage_digester(ml)
    x = np.ascontiguousarray(X[:256, :16], dtype=np.float32)
    dy = np.ascontiguousarray(X[256:512, :16], dtype=np.float32)
    a = ml.MoEBlock(16, 24, num_experts=6, top_k=2, random_state=1)
    dxa = a.backward(x, dy)
    b = ml.MoEBlock(16, 12, num_experts=5, top_k=3, norm_topk_prob=False, random_state=2)
    router = b.router.copy()
    router[3] = router[1]
    b.router = router
    dxb = b.backward(x, dy)
    return _fit(dict(dxa=d(dxa), ra=d(a.router_grad_), gua=d(a.gate_up_proj_grad_), dna=d(a.down_proj_grad_),
                     dxb=d(dxb), rb=d(b.router_grad_), gub=d(b.gate_up_proj_grad_), dnb=d(b.down_proj_grad_)))


# ---------------------------------------------------------------- Sequential: export, load, predict

@lane("sequence-export-roundtrip")
def _(ml, X, yc, yr, Xh=None):
    """A Sequential classifier (6 -> 8 relu -> 2, cross-entropy, Adam) fitted
    two unshuffled epochs on 64 fixture rows, exported for inference
    (`export`), loaded back (`Sequential.load`), and asked `predict` and
    `predict_proba` on the training and held-out rows. The loaded model's
    answers are held byte for byte to the fitted model's in the cell, which
    is `export`'s documented claim. Parts: the exported file's bytes and the
    loaded model's answers."""
    d = _coverage_digester(ml)
    x = np.ascontiguousarray(X[:64, :6], dtype=np.float32)
    y = np.ascontiguousarray(yc[:64], dtype=np.int64)
    xh = np.ascontiguousarray((X if Xh is None else Xh)[:64, :6], dtype=np.float32)
    m = ml.Sequential(numeric_mode="identical").linear(8).relu().linear(2)
    m = m.compile(loss="cross_entropy", optimizer="adam", lr=0.01)
    m = m.fit(x, y, epochs=2, batch_size=32, shuffle=False, seed=0)
    with tempfile.TemporaryDirectory(prefix="ib-sequence-export-") as tmp:
        path = os.path.join(tmp, "model.ckpt")
        m.export(path)
        saved = np.fromfile(path, dtype=np.uint8)
        back = ml.Sequential.load(path, numeric_mode="identical")
        pred, proba = np.asarray(back.predict(xh)), np.asarray(back.predict_proba(xh))
        train_proba = np.asarray(back.predict_proba(x))
    _same_bytes("loaded predict_proba", proba, "fitted predict_proba", np.asarray(m.predict_proba(xh)))
    _same_bytes("loaded predict", pred, "fitted predict", np.asarray(m.predict(xh)))
    return _fit(dict(file=d(saved), predict=d(pred), proba=d(proba), train_proba=d(train_proba)))


# ---------------------------------------------------------------- import_model on an ONNX file

def _coverage_varint(v):
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def _coverage_field(num, wire, payload):
    """One protobuf field: wire 0 an unsigned varint, wire 2 length-delimited bytes."""
    key = _coverage_varint((num << 3) | wire)
    if wire == 0:
        return key + _coverage_varint(int(payload))
    payload = payload.encode("utf-8") if isinstance(payload, str) else bytes(payload)
    return key + _coverage_varint(len(payload)) + payload


def _coverage_onnx(path, X):
    """Write x [batch, 6] -> Gemm(transB) [batch, 8] -> Relu -> Gemm(transB)
    [batch, 3] as an ONNX file (opset 17), its weights the fixture's own
    float32 values. Field numbers are onnx.proto3's: ModelProto ir_version 1,
    graph 7, opset_import 8; GraphProto node 1, name 2, initializer 5,
    input 11, output 12; NodeProto input 1, output 2, name 3, op_type 4,
    attribute 5; AttributeProto name 1, i 3, type 20 (INT = 2); TensorProto
    dims 1, data_type 2 (FLOAT = 1), name 8, raw_data 9; ValueInfoProto
    name 1, type 2; TypeProto tensor_type 1; Tensor elem_type 1, shape 2;
    TensorShapeProto dim 1; Dimension dim_value 1, dim_param 2."""
    F = _coverage_field
    flat = np.ascontiguousarray(X, dtype=np.float32).reshape(-1)
    weights = (("w1", (8, 6), 4096), ("b1", (8,), 4196), ("w2", (3, 8), 4296), ("b2", (3,), 4396))

    def tensor(name, shape, start):
        n = int(np.prod(shape))
        dims = F(1, 2, b"".join(_coverage_varint(s) for s in shape))  # glue: packed dims
        raw = (flat[start:start + n] * np.float32(0.25)).astype(np.float32).tobytes()
        return dims + F(2, 0, 1) + F(8, 2, name) + F(9, 2, raw)

    def value_info(name, dims):
        shape = b"".join(F(1, 2, F(2, 2, v) if isinstance(v, str) else F(1, 0, v)) for v in dims)  # glue: dims
        return F(1, 2, name) + F(2, 2, F(1, 2, F(1, 0, 1) + F(2, 2, shape)))

    def node(op, ins, outs, name, trans_b=False):
        out = b"".join(F(1, 2, i) for i in ins) + b"".join(F(2, 2, o) for o in outs)  # glue: names
        out += F(3, 2, name) + F(4, 2, op)
        if trans_b:
            out += F(5, 2, F(1, 2, "transB") + F(3, 0, 1) + F(20, 0, 2))
        return out

    graph = F(1, 2, node("Gemm", ["x", "w1", "b1"], ["h"], "fc1", True))
    graph += F(1, 2, node("Relu", ["h"], ["hr"], "relu"))
    graph += F(1, 2, node("Gemm", ["hr", "w2", "b2"], ["y"], "fc2", True))
    graph += F(2, 2, "coverage")
    graph += b"".join(F(5, 2, tensor(*w)) for w in weights)  # glue: four initializers
    graph += F(11, 2, value_info("x", ("batch", 6))) + F(12, 2, value_info("y", ("batch", 3)))
    model = F(1, 0, 8) + F(2, 2, "mojolearn-verify") + F(8, 2, F(1, 2, "") + F(2, 0, 17)) + F(7, 2, graph)
    with open(path, "wb") as fh:
        fh.write(model)
    return path


@lane("sequence-onnx-import")
def _(ml, X, yc, yr, Xh=None):
    """mojolearn.import_model on a two-layer ONNX file the lane writes from
    the fixture's own values (no onnx package): the model read and lowered
    in Mojo (onnx_import), run on 64 training rows and 64 held-out rows, the
    batch fixed by `shapes`. Parts: the file's bytes and both outputs."""
    d = _coverage_digester(ml)
    x = np.ascontiguousarray(X[:64, :6], dtype=np.float32)
    xh = np.ascontiguousarray((X if Xh is None else Xh)[:64, :6], dtype=np.float32)
    with tempfile.TemporaryDirectory(prefix="ib-onnx-import-") as tmp:
        path = _coverage_onnx(os.path.join(tmp, "model.onnx"), X)
        model = np.fromfile(path, dtype=np.uint8)
        prog = ml.import_model(path, shapes={"x": (64, 6)}, numeric_mode="identical")
        out = np.asarray(prog.run(x))
        held = np.asarray(prog.run(xh))
    return _fit(dict(file=d(model), out=d(out), held=d(held)))


# ---------------------------------------------------------------- the Transformer decode session's load_state

@lane("transformer-decode-load-state")
def _(ml, X, yc, yr, Xh=None):
    """TransformerDecodeSession.load_state on the `transformer-decode-session`
    lane's block: a session steps four tokens of two rows, `sync_state`
    copies its cache into the state, four more tokens are stepped, the
    state's `cached_tokens` is set back to four and `load_state` rewinds the
    session to the four-token cache, and four other tokens are stepped. Those last outputs are held byte for byte in the cell
    to a fresh session that stepped the first four tokens and then the same
    four, which is load_state's documented claim."""
    d = _coverage_digester(ml)
    dm, nh, nkv, hd, it = 32, 2, 1, 16, 64
    w = _block_weights("transformer", {
        "input_layernorm.weight": (dm,), "post_attention_layernorm.weight": (dm,),
        "q_proj.weight": (nh * hd, dm), "k_proj.weight": (nkv * hd, dm), "v_proj.weight": (nkv * hd, dm),
        "o_proj.weight": (dm, nh * hd), "gate_proj.weight": (it, dm), "up_proj.weight": (it, dm),
        "down_proj.weight": (dm, it)},
        near_one=("input_layernorm.weight", "post_attention_layernorm.weight"))
    blk = ml.TransformerBlock(w, n_heads=nh, n_kv_heads=nkv)
    x = _seq(X, 2, 12, dm, skip=2048)

    def tok(t):
        return np.ascontiguousarray(x[:, t:t + 1])

    st = blk.allocate_state(2, max_tokens=32)
    with blk.decode_session(st) as sess:
        first = [np.asarray(sess.step(tok(t))) for t in range(4)]  # glue: four decode steps
        sess.sync_state()
        for t in range(4, 8):  # glue: four steps the reload discards
            sess.step(tok(t))
        # the edit load_state is for: rewind the state's length to the synced
        # four tokens (each step advanced it)
        st.cached_tokens = 4
        sess.load_state()
        after = np.concatenate([np.asarray(sess.step(tok(t))) for t in range(8, 12)], axis=1)  # glue: four steps
        sess.sync_state()
        resident = np.concatenate(_decode_state_arrays(st))
    ref = blk.allocate_state(2, max_tokens=32)
    with blk.decode_session(ref) as sess:
        for t in range(4):  # glue: the same first four steps
            sess.step(tok(t))
        want = np.concatenate([np.asarray(sess.step(tok(t))) for t in range(8, 12)], axis=1)  # glue: four steps
    _same_bytes("steps after load_state", after, "steps on a session that never left the four-token cache", want)
    return _fit(dict(first=d(np.concatenate(first, axis=1)), after=d(after), state=d(resident)))


# ---------------------------------------------------------------- float16 checkpoint weights

@lane("safetensors-widen-f16")
def _(ml, X, yc, yr, Xh=None):
    """mojolearn.models.safetensors.widen_f16, the widening a float16
    checkpoint's weights take on load (the linalg binding's from_f16), over
    every one of the 65536 float16 bit patterns (zeros, subnormals, normals,
    both infinities and every NaN payload) and over the fixture's own values
    rounded to float16. Exact by construction, so every column has the same
    bits."""
    from mojolearn.models.safetensors import widen_f16
    d = _coverage_digester(ml)
    every = np.arange(65536, dtype=np.uint16)
    fixture = np.ascontiguousarray(X[:256, :16], dtype=np.float32).astype(np.float16).view(np.uint16)
    return _fit(dict(every=d(np.asarray(widen_f16(every), dtype=np.float32)),
                     fixture=d(np.asarray(widen_f16(np.ascontiguousarray(fixture)), dtype=np.float32))))


# ---------------------------------------------------------------- ops.Program: save and load_program

@lane("sequence-program-save-load")
def _(ml, X, yc, yr, Xh=None):
    """An ops.Program (the `sequence-tape-program` lane's 6 -> 8 tanh -> 1
    network on seeded weights) captured, run on 64 fixture rows, saved
    (`Program.save`), loaded into a second session (`Session.load_program`)
    and run on the same rows and on 64 held-out rows. The loaded program's
    output is held byte for byte to the captured one's in the cell, which is
    load_program's documented claim. Parts: both outputs and the program's
    held-out output."""
    B = _coverage_binding(ml)
    d = lambda a: _coverage_digest(B, a)
    ops = ml.ops
    x = np.ascontiguousarray(X[:64, :6], dtype=np.float32)
    xh = np.ascontiguousarray((X if Xh is None else Xh)[:64, :6], dtype=np.float32)
    init = []
    for k, shape in enumerate(((6, 8), (1, 8), (8, 1), (1, 1))):  # glue: four seeded weight buffers
        a = np.empty(shape, dtype=np.float32)
        # cpu-route: verification inputs made in Mojo before the run
        B.verify_fill(a.ctypes.data, a.size, 21, k)
        init.append(a)
    with tempfile.TemporaryDirectory(prefix="ib-program-save-") as tmp:
        path = os.path.join(tmp, "net.mlprog")
        with ops.Session(numeric_mode="identical") as s:
            w = [s.tensor(a) for a in init]  # glue: four parameters
            xin = s.tensor(np.zeros((64, 6), dtype=np.float32))
            program = s.capture([xin], [s.linear(s.linear(xin, w[0], w[1], activation="tanh"), w[2], w[3])])
            out = np.asarray(program.run(x))
            program.save(path)
        with ops.Session(numeric_mode="identical") as s:
            loaded = s.load_program(path)
            if not isinstance(loaded, ml.ops.Program):
                raise RuntimeError(f"Session.load_program returned {type(loaded).__name__}, not ops.Program")
            again = np.asarray(loaded.run(x))
            held = np.asarray(loaded.run(xh))
    _same_bytes("loaded program run", again, "captured program run", out)
    return _fit(dict(out=d(out), held=d(held)))


# ---------------------------------------------------------------- CausalLM.stream: GenerationStream

@lane("hf-causal-lm-stream")
def _(ml, X, yc, yr, Xh=None):
    """CausalLM.stream on the `hf-causal-lm` lane's Llama fixture and prompt:
    a greedy GenerationStream read two tokens at a time, and a sampled one
    (seed 7, temperature 0.8, top-k 8) read three at a time. Each stream's
    ids are held byte for byte in the cell to `generate`'s continuation with
    the same options, which is the stream's documented claim (read size does
    not change the ids)."""
    from mojolearn import _causal_lm_fixtures as fx
    d = _coverage_digester(ml)
    A = ml.Array.from_buffer
    ids = _ids(X, HF_LM_BATCH, HF_LM_LEN, HF_LM_VOCAB)
    cfg, tensors = fx.family_fixture("llama", False)
    sampling = dict(greedy=False, seed=7, temperature=0.8, top_k=8)

    def read_all(stream, step):
        if not isinstance(stream, ml.models.GenerationStream):
            raise RuntimeError(f"CausalLM.stream returned {type(stream).__name__}, not models.GenerationStream")
        pieces = [np.asarray(stream.read(step), dtype=np.int32)
                  for _ in range(-(-HF_LM_NEW // step))]  # glue: one read per chunk of new tokens
        return np.ascontiguousarray(np.concatenate(pieces, axis=1))

    with tempfile.TemporaryDirectory(prefix="ib-causal-lm-stream-") as tmp:
        root = fx._write_checkpoint(os.path.join(tmp, "llama"), cfg, tensors)
        m = ml.models.causal_lm.CausalLM.load(root, numeric_profile="fp32_v1")
        greedy = np.asarray(m.generate(A(ids), HF_LM_NEW), dtype=np.int32)[:, HF_LM_LEN:]
        sampled = np.asarray(m.generate(A(ids), HF_LM_NEW, **sampling), dtype=np.int32)[:, HF_LM_LEN:]
        streamed = read_all(m.stream(A(ids), HF_LM_NEW), 2)
        streamed_sampled = read_all(m.stream(A(ids), HF_LM_NEW, **sampling), 3)
    _same_bytes("greedy stream", streamed, "greedy generate continuation", np.ascontiguousarray(greedy))
    _same_bytes("sampled stream", streamed_sampled, "sampled generate continuation", np.ascontiguousarray(sampled))
    return _fit(dict(greedy=d(streamed), sampled=d(streamed_sampled)))


# ---------------------------------------------------------------- IVF: a query with fewer candidates than k

@lane("ivf-short-fill")
def _(ml, X, yc, yr, Xh=None):
    """IVFIndex on 128 fixture rows in 16 lists, each of 16 queries probing
    one list (about eight candidates) for 24 neighbours: every query has
    fewer candidates than k, so its answer is its real neighbours and then
    the short fill (+inf distances, id -1), on the device search and the host
    column alike. Parts: the distances and ids, squared and rooted metrics."""
    d = _coverage_digester(ml)
    q = np.ascontiguousarray(X[4096:4112])
    m = ml.IVFIndex(n_lists=16, n_probes=1, n_neighbors=24, random_state=5).fit(X[:128])
    dist, idx = m.search(q)
    e = ml.IVFIndex(metric="euclidean", n_lists=16, n_probes=1, n_neighbors=24, random_state=5).fit(X[:128])
    edist, eidx = e.search(q)
    return _fit(dict(dist=d(np.asarray(dist, dtype=np.float32)), idx=d(np.asarray(idx, dtype=np.int64)),
                     root_dist=d(np.asarray(edist, dtype=np.float32)), root_idx=d(np.asarray(eidx, dtype=np.int64))))


# ---------------------------------------------------------------- resident sequence tensors

@lane("sequence-device-tensor")
def _(ml, X, yc, yr, Xh=None):
    """RMSprop over params and gradients handed to it as resident tensors
    (`to_device`, sequence/seq_tensor.mojo on a GPU binding; the float32
    array itself on a binding without resident tensors), six steps on the
    `sequence-rmsprop` lane's fixture tensors, held byte for byte in the cell
    to the same steps on host arrays, which is the resident path's claim (the
    same launches on the same values). Then a tensor's round trip: upload,
    copy and download."""
    from mojolearn._x_sequence_device import to_device
    d = _coverage_digester(ml)

    def run(wrap):
        p1 = wrap(np.ascontiguousarray(X[:32, :8], dtype=np.float32).copy())
        p2 = wrap(np.ascontiguousarray(X[32:40, 0], dtype=np.float32).copy())
        opt = ml.RMSprop([p1, p2], lr=1e-2, momentum=0.5)
        for k in range(6):  # glue: six optimizer steps
            base = 40 + 48 * k
            g1 = np.ascontiguousarray(X[base:base + 32, 8:16], dtype=np.float32) * np.float32(0.5)
            g2 = np.ascontiguousarray(X[base + 32:base + 40, 1], dtype=np.float32)
            opt.step([wrap(g1), wrap(g2)])
        return np.asarray(p1, dtype=np.float32), np.asarray(p2, dtype=np.float32)

    host = run(lambda a: a)
    resident = run(to_device)
    _same_bytes("resident-tensor RMSprop params", np.concatenate([a.reshape(-1) for a in resident]),
                "host-array RMSprop params", np.concatenate([a.reshape(-1) for a in host]))
    t = to_device(np.ascontiguousarray(X[:16, :4], dtype=np.float32))
    back = np.asarray(t, dtype=np.float32)
    twin = np.asarray(t.copy(), dtype=np.float32)
    _same_bytes("downloaded tensor", back, "uploaded array", np.ascontiguousarray(X[:16, :4], dtype=np.float32))
    return _fit(dict(p1=d(resident[0]), p2=d(resident[1]), back=d(back), copy=d(twin)))


# ---------------------------------------------------------------- randomized SVD of a rank-deficient matrix

@lane("x-decomp-rsvd-rank-deficient")
def _(ml, X, yc, yr, Xh=None):
    """randomized_svd, PCA(svd_solver="randomized") and
    TruncatedSVD(algorithm="randomized") of a 256 x 8 matrix of rank 4 (four
    fixture columns and the same four doubled, exactly): the sketch is eight
    columns wide, so the orthonormalized range has four dead columns, and the
    live ones are compacted by the stable order of the 0 (live) / 1 (dead)
    keys (`x_decomp_order_small` on the host column, `x_decomp_dev_order_small`
    on a GPU binding, x_decomp/select_ops.mojo and select_dev.mojo) before the
    small SVD. No other lane's sketch is wider than its matrix's rank. Parts:
    U, s and Vt, the PCA components, singular values and transform, and the
    TruncatedSVD components and transform."""
    d = _coverage_digester(ml)
    B = np.ascontiguousarray(X[:256, :4], dtype=np.float32)
    M = np.ascontiguousarray(np.concatenate([B, B * np.float32(2)], axis=1))  # rank 4: a power-of-two copy is exact
    U, s, Vt = ml.randomized_svd(M, 2, random_state=0)
    p = ml.PCA(n_components=2, svd_solver="randomized", random_state=1).fit(M)
    t = ml.TruncatedSVD(n_components=3, algorithm="randomized", random_state=2).fit(M)
    f32 = lambda a: np.asarray(a, dtype=np.float32)
    return _fit(dict(U=d(f32(U)), s=d(f32(s)), Vt=d(f32(Vt)), pc=d(f32(p.components_)),
                     psv=d(f32(p.singular_values_)), pT=d(f32(p.transform(M[:64]))),
                     tc=d(f32(t.components_)), tT=d(f32(t.transform(M[:64])))))


_batch_decl("n/a:coverage-probe (the lane checks one public call path the release audit found no lane for; "
            "its outputs are train parts, digested whole)",
            "gp-lml-theta", "cross-vendor-fold", "sequence-moe-backward", "sequence-export-roundtrip",
            "sequence-onnx-import", "transformer-decode-load-state", "safetensors-widen-f16",
            "sequence-program-save-load", "hf-causal-lm-stream", "ivf-short-fill", "sequence-device-tensor",
            "x-decomp-rsvd-rank-deficient")
