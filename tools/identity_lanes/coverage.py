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
    # local names are unique: lane_applicability walks bare-name calls across
    # the harness, and a nested `run`/`d` elsewhere reaches _par_devices
    _sdt_digest = _coverage_digester(ml)

    def _rmsprop_six_steps(_as_lane_tensor):
        p1 = _as_lane_tensor(np.ascontiguousarray(X[:32, :8], dtype=np.float32).copy())
        p2 = _as_lane_tensor(np.ascontiguousarray(X[32:40, 0], dtype=np.float32).copy())
        opt = ml.RMSprop([p1, p2], lr=1e-2, momentum=0.5)
        for k in range(6):  # glue: six optimizer steps
            base = 40 + 48 * k
            g1 = np.ascontiguousarray(X[base:base + 32, 8:16], dtype=np.float32) * np.float32(0.5)
            g2 = np.ascontiguousarray(X[base + 32:base + 40, 1], dtype=np.float32)
            opt.step([_as_lane_tensor(g1), _as_lane_tensor(g2)])
        return np.asarray(p1, dtype=np.float32), np.asarray(p2, dtype=np.float32)

    host = _rmsprop_six_steps(lambda a: a)
    resident = _rmsprop_six_steps(to_device)
    _same_bytes("resident-tensor RMSprop params", np.concatenate([a.reshape(-1) for a in resident]),
                "host-array RMSprop params", np.concatenate([a.reshape(-1) for a in host]))
    t = to_device(np.ascontiguousarray(X[:16, :4], dtype=np.float32))
    back = np.asarray(t, dtype=np.float32)
    twin = np.asarray(t.copy(), dtype=np.float32)
    _same_bytes("downloaded tensor", back, "uploaded array", np.ascontiguousarray(X[:16, :4], dtype=np.float32))
    return _fit(dict(p1=_sdt_digest(resident[0]), p2=_sdt_digest(resident[1]), back=_sdt_digest(back), copy=_sdt_digest(twin)))


def _coverage_device_optimizer(ml, cls, X, configurations, factored=False):
    """Compare public host-array and resident-tensor steps and consumed state.

    The CPU column exercises to_device's host fallback; only a GPU column
    qualifies the typed device-parameter step. All configurations use identical
    mode explicitly, and no device-specific metadata enters the digest.
    """
    from mojolearn._x_sequence_device import SequenceDeviceTensor, to_device
    d = _coverage_digester(ml)
    parts = {}

    def consume(value):
        # Test-controller dispatch: invoke the tensor's native download method.
        # Its separate dtype-changing __array__ conversion remains unqualified.
        return value.numpy() if isinstance(value, SequenceDeviceTensor) else value

    def run(resident, settings):
        def wrap(value):
            return to_device(value, numeric_mode="identical") if resident else value
        params = [wrap(np.ascontiguousarray(X[:32, :8], dtype=np.float32).copy()),
                  wrap(np.ascontiguousarray(X[32:40, 0], dtype=np.float32).copy())]
        opt = cls(params, numeric_mode="identical", **settings)
        for k in range(6):  # glue: six public optimizer calls over fixture gradients
            base = 40 + 48 * k
            g1 = np.ascontiguousarray(X[base:base + 32, 8:16], dtype=np.float32) * np.float32(0.5)
            g2 = np.ascontiguousarray(X[base + 32:base + 40, 1], dtype=np.float32)
            opt.step([wrap(g1), wrap(g2)])
        # State synchronizes through the public native property while the owner
        # is alive; this checks reductions/state as well as updated parameters.
        state = opt.state
        values = {"p1": consume(params[0]), "p2": consume(params[1]),
                  "step": np.asarray([opt.t], dtype=np.int64)}
        if factored:
            values.update(row_var=state[0]["row_var"], col_var=state[0]["col_var"],
                          variance=state[1]["variance"])
        else:
            values.update(first_moment=state[0], second_moment=state[1])
        assert opt.t == 6, "optimizer did not consume all six gradient pairs"
        return values

    for name, settings in configurations:  # glue: the lane's two numerical configurations
        host = run(False, settings)
        resident = run(True, settings)
        for part, value in resident.items():  # glue: compare and digest consumed parameter/state outputs
            _same_bytes(f"resident {cls.__name__} {name} {part}", value,
                        f"host-array {cls.__name__} {name} {part}", host[part])
            parts[name + "_" + part] = d(value)
    return _fit(parts)


@lane("sequence-adafactor-device")
def _(ml, X, yc, yr, Xh=None):
    """Adafactor's resident parameter and gradient path, six steps over a
    32 x 8 factored matrix and an eight-element vector, compared byte for
    byte with host-array steps. Defaults and changed beta2_decay, clipping d
    and weight decay; consume both parameters, step count and every state
    buffer through the public native tensor/state methods."""
    return _coverage_device_optimizer(ml, ml.Adafactor, X,
                                      (("plain", dict(lr=1e-2, beta2_decay=-0.8, eps=(None, 1e-3),
                                                      d=1.0, weight_decay=0.0, maximize=False)),
                                       ("moved", dict(lr=3e-2, beta2_decay=-0.6, eps=(None, 1e-3),
                                                      d=2.0, weight_decay=0.1, maximize=False))),
                                      factored=True)


@lane("sequence-lamb-device")
def _(ml, X, yc, yr, Xh=None):
    """LAMB's resident parameter and gradient path on the same matrix and
    vector as sequence-lamb, six steps compared byte for byte with host-array
    steps. Global clipping/decay defaults and always_adapt/trust_clip without
    decay or global clipping; consume parameters, moments and step count."""
    return _coverage_device_optimizer(ml, ml.LAMB, X,
                                      (("plain", dict(lr=1e-2, bias_correction=True, betas=(0.9, 0.999),
                                                      eps=1e-6, weight_decay=0.01, grad_averaging=True, max_grad_norm=1.0,
                                                      always_adapt=False, trust_clip=False)),
                                       ("adapt", dict(lr=1e-2, bias_correction=True, betas=(0.9, 0.999),
                                                      eps=1e-6, weight_decay=0.0, grad_averaging=True, max_grad_norm=None,
                                                      always_adapt=True, trust_clip=True))))


# ================================================================ lane verify-coverage-owed (2026-10-21)
# The rows the audit left owed to the migration lanes, closed after they
# landed (docs/plans/VERIFY_COVERAGE_20261021.md, "Owed").

# ---------------------------------------------------------------- Mamba-2 and Mamba-3 resident decode sessions

def _coverage_state_pieces(st):
    """A Mamba-2 or Mamba-3 decode state as (every float buffer in the
    verifier's fixed piece order, the cursors), so two states compare byte
    for byte."""
    pieces, cursors = _RL_ROW_PIECES[type(st).__name__]
    flat = np.concatenate([np.ascontiguousarray(np.asarray(getattr(st, n)), dtype=np.float32).reshape(-1)
                           for n in pieces])  # glue: the state's named pieces
    words = np.asarray([int(getattr(st, n)) for n in cursors], dtype=np.int64)  # glue: the state's cursors
    return flat, words


def _coverage_decode_session(ml, blk, X, dm):
    """`blk._decode_session(state)` (bindings/mamba_public_native.mojo, the
    binding's `<kind>_session_open/step/export_state/load_state`) on two rows
    of the fixture slab: four steps, `sync_state`, four steps the reload
    discards, `load_state` (the session rewinds to the synced four-token
    state, which the caller's buffers still hold), four more steps and a
    final `sync_state`. Each output and both synced states are held byte for
    byte IN THE CELL to the per-call `block.step` on a second state that
    stepped the first four tokens and then the same last four, which is the
    session's contract (the same L = 1 block entry on the same state)."""
    d = _coverage_digester(ml)
    x = _seq(X, 2, 12, dm, skip=1024)

    def tok(t):
        return np.ascontiguousarray(x[:, t:t + 1])

    st = blk.allocate_state(2)
    with blk._decode_session(st) as sess:
        first = np.concatenate([np.asarray(sess.step(tok(t))) for t in range(4)], axis=1)  # glue: four steps
        sess.sync_state()
        synced = _coverage_state_pieces(st)
        for t in range(4, 8):  # glue: four steps the reload discards
            sess.step(tok(t))
        sess.load_state()
        after = np.concatenate([np.asarray(sess.step(tok(t))) for t in range(8, 12)], axis=1)  # glue: four steps
        sess.sync_state()
        resident = _coverage_state_pieces(st)
    plain = blk.allocate_state(2)
    want_first = np.concatenate([np.asarray(blk.step(tok(t), plain)) for t in range(4)], axis=1)  # glue: four steps
    plain_synced = _coverage_state_pieces(plain)
    want = np.concatenate([np.asarray(blk.step(tok(t), plain)) for t in range(8, 12)], axis=1)  # glue: four steps
    plain_final = _coverage_state_pieces(plain)
    _same_bytes("session steps", first, "per-call block.step", want_first)
    _same_bytes("session sync_state", synced[0], "per-call state", plain_synced[0])
    _same_bytes("session sync_state cursors", synced[1], "per-call state cursors", plain_synced[1])
    _same_bytes("steps after load_state", after, "per-call steps after the same four tokens", want)
    _same_bytes("final sync_state", resident[0], "per-call final state", plain_final[0])
    _same_bytes("final sync_state cursors", resident[1], "per-call final cursors", plain_final[1])
    return _fit(dict(first=d(first), after=d(after), state=d(resident[0]), cursors=d(resident[1])))


@lane("mamba2-decode-session")
def _(ml, X, yc, yr, Xh=None):
    """Mamba2Block's resident decode session on the `mamba2` lane's weights:
    steps, sync_state and load_state (`mamba2_session_open`,
    `mamba2_session_export_state`, `mamba2_session_load_state`), held in the
    cell to the per-call step. The state is conv_window, h, buffer_xbc,
    buffer_dtraw and buffered_tokens."""
    dm, di, nh = 32, 64, 1
    cd, dip = di + 256, 2 * di + 256 + nh
    w = _block_weights("mamba2", {
        "block_norm.weight": (dm,), "in_proj.weight": (dip, dm), "conv1d.weight": (cd, 1, 4),
        "conv1d.bias": (cd,), "dt_bias": (nh,), "A_log": (nh,), "D": (nh,), "norm.weight": (di,),
        "out_proj.weight": (dm, di)},
        ones=("block_norm.weight", "norm.weight"))
    return _coverage_decode_session(ml, ml.Mamba2Block(w), X, dm)


@lane("mamba3-decode-session")
def _(ml, X, yc, yr, Xh=None):
    """Mamba3Block's resident decode session on the `mamba3` lane's weights:
    steps, sync_state and load_state (`mamba3_session_open`,
    `mamba3_session_step`, `mamba3_session_export_state`,
    `mamba3_session_load_state`), held in the cell to the per-call step. The
    state is the ten `_STATE_NAMES` pieces, buffered_tokens and pending."""
    dm, di, nh = 32, 64, 1
    dip = 2 * di + 256 + 3 * nh + 32
    w = _block_weights("mamba3", {
        "block_norm.weight": (dm,), "in_proj.weight": (dip, dm), "dt_bias": (nh,),
        "B_norm.weight": (128,), "C_norm.weight": (128,), "B_bias": (nh, 128), "C_bias": (nh, 128),
        "D": (nh,), "out_proj.weight": (dm, di)},
        ones=("block_norm.weight",), near_one=("B_norm.weight", "C_norm.weight"))
    return _coverage_decode_session(ml, ml.Mamba3Block(w), X, dm)


# ---------------------------------------------------------------- training.chunked_lm_head_loss

@lane("training-chunked-lm-head")
def _(ml, X, yc, yr, Xh=None):
    """training.chunked_lm_head_loss, the language-model head that never
    holds the logits (`chunked_lm_head_v2_loss` without gradients,
    `chunked_lm_head_v2_train` with them), at two shapes: 64 rows of width 16
    over a vocabulary of 40, and 32 rows of width 8 over 300. Hidden states
    and the head are the fixture's own values (the head scaled by 1/4),
    targets its bytes. The loss without gradients is held byte for byte to
    the loss with them in the cell (the same chain over the vocabulary).
    Parts: each shape's loss and both gradients."""
    T = ml.training
    d = _coverage_digester(ml)
    parts = {}
    for tag, rows, vocab, width, skip in (("small", 64, 40, 16, 0), ("wide", 32, 300, 8, 4096)):  # glue: two shapes
        h = np.ascontiguousarray(_seq(X, 1, rows, width, skip=skip)[0])
        w = np.ascontiguousarray(_seq(X, 1, vocab, width, skip=skip + rows * width)[0] * np.float32(0.25))
        t = np.ascontiguousarray(_ids(X, 1, rows, vocab).reshape(-1))
        alone = np.asarray([T.chunked_lm_head_loss(h, w, t)], dtype=np.float64)
        loss, dh, dw = T.chunked_lm_head_loss(h, w, t, return_grad=True)
        loss = np.asarray([loss], dtype=np.float64)
        _same_bytes(f"{tag} loss", alone, f"{tag} loss with return_grad=True", loss)
        parts[f"{tag}_loss"] = d(loss)
        parts[f"{tag}_dhidden"] = d(np.asarray(dh, dtype=np.float32))
        parts[f"{tag}_dweight"] = d(np.asarray(dw, dtype=np.float32))
    return _fit(parts)


# ---------------------------------------------------------------- training.mlp_inference_sessions

@lane("training-mlp-sessions")
def _(ml, X, yc, yr, Xh=None):
    """training.mlp_inference_sessions (`mlp_sessions_rows`, `mlp_sessions`):
    a seeded 8 -> 24 relu -> 3 model over three sessions of 5, 1 and 64
    fixture rows in one call, and over the held-out rows. The one call is
    held byte for byte in the cell to each session run alone and
    concatenated, which is the function's claim (independent sessions, in
    the order given). Parts: the training-row and held-out logits."""
    T = ml.training
    d = _coverage_digester(ml)
    w1 = _hw((24, 8), "training-mlp-sessions:w1", -0.5, 0.5)
    b1 = _hw((24,), "training-mlp-sessions:b1", -0.1, 0.1)
    w2 = _hw((3, 24), "training-mlp-sessions:w2", -0.5, 0.5)
    b2 = _hw((3,), "training-mlp-sessions:b2", -0.1, 0.1)
    spans = ((0, 5), (5, 6), (6, 70))

    def sessions(A):
        return [np.ascontiguousarray(A[a:b, :8], dtype=np.float32) for a, b in spans]  # glue: three sessions

    out = np.asarray(T.mlp_inference_sessions(sessions(X), w1, b1, w2, b2))
    alone = np.concatenate([np.asarray(T.mlp_inference_sessions([s], w1, b1, w2, b2))
                            for s in sessions(X)])  # glue: one call per session
    _same_bytes("three sessions in one call", out, "each session alone, concatenated", alone)
    held = np.asarray(T.mlp_inference_sessions(sessions(X if Xh is None else Xh), w1, b1, w2, b2))
    return _fit(dict(logits=d(out), held=d(held)))


# ---------------------------------------------------------------- SambaStack: optimizer state and maximize

@lane("samba-optimizer-state")
def _(ml, X, yc, yr, Xh=None):
    """The Samba stack's resident optimizer state: a stack takes one step,
    `state_dict()` (the slot's moments downloaded, `optimizer_resident_download`)
    and a second step; a second stack whose slot is already open (one step on
    other rows) `load_state_dict`s the saved state (`optimizer_resident_upload`)
    and takes the same second step. Its parameters and logits are held byte for
    byte in the cell to the uninterrupted stack's, which is load_state_dict's
    claim. Then `optimizer.maximize = True` on the `samba` lane's stack with
    dropout 0.1, two steps (on a GPU column the resident step with the
    maximize word, `optimizer_resident_step_io`). Parts: the saved moments, the
    resumed parameters and logits, the maximized parameters."""
    d = _coverage_digester(ml)
    T = ml.training
    cfg = ml.SambaConfig(vocab=256, d_model=32, layers=("mamba3", "attention"), n_heads=2, intermediate=64)
    ids = _ids(X, 8, 17)
    win = [(ids[2 * k:2 * k + 2, :-1], ids[2 * k:2 * k + 2, 1:]) for k in range(4)]  # glue: four (2, 16) windows

    def flat(m):
        p = m.parameters()
        return np.concatenate([np.asarray(p[k], dtype=np.float32).reshape(-1) for k in sorted(p)])  # glue: names

    a = ml.SambaStack(cfg, generator=T.Generator(1), lr=1e-3)
    a.train_step(*win[0])
    saved = a.state_dict()
    a.train_step(*win[1])
    b = ml.SambaStack(cfg, generator=T.Generator(1), lr=1e-3)
    b.train_step(*win[2])
    b.load_state_dict(saved)
    b.train_step(*win[1])
    _same_bytes("resumed stack parameters", flat(b), "uninterrupted stack parameters", flat(a))
    logits = np.asarray(b.forward(ids[:2, :-1]))
    _same_bytes("resumed stack logits", logits, "uninterrupted stack logits", np.asarray(a.forward(ids[:2, :-1])))
    moments = np.concatenate([np.asarray(saved[k], dtype=np.float32).reshape(-1) for k in ("exp_avg", "exp_avg_sq")])
    c = ml.SambaStack(ml.SambaConfig(vocab=256, d_model=32, layers=("mamba3", "attention"), n_heads=2,
                                     intermediate=64, dropout=0.1),
                      generator=T.Generator(1), lr=1e-3)
    c.optimizer.maximize = True
    for k in range(2):  # glue: two maximized steps
        c.train_step(*win[k])
    return _fit(dict(moments=d(moments), resumed=d(flat(b)), logits=d(logits), maximize=d(flat(c))))


# ---------------------------------------------------------------- sequence optimizers: state_dict and load_state_dict

@lane("sequence-optimizer-state")
def _(ml, X, yc, yr, Xh=None):
    """NAdam and centered RMSprop with momentum (three state slots, NAdam's
    carried scalars) on two fixture tensors: one step, `state_dict()`
    (`seq_optimizer_export`), two more steps; a fresh optimizer on copies of
    the one-step parameters `load_state_dict`s the saved state
    (`seq_optimizer_import`) and takes the same two steps. Its parameters are
    held byte for byte in the cell to the uninterrupted optimizer's, which is
    the resume claim. Parts: the exported state and the resumed parameters."""
    d = _coverage_digester(ml)

    def grads(k):
        base = 40 + 48 * k
        return [np.ascontiguousarray(X[base:base + 32, 8:16], dtype=np.float32) * np.float32(0.5),
                np.ascontiguousarray(X[base + 32:base + 40, 1], dtype=np.float32)]

    parts = {}
    for tag, make in (("nadam", lambda ps: ml.NAdam(ps, lr=1e-2)),
                      ("rmsprop", lambda ps: ml.RMSprop(ps, lr=1e-2, momentum=0.5, centered=True))):  # glue: two
        ps = [np.ascontiguousarray(X[:32, :8], dtype=np.float32).copy(),
              np.ascontiguousarray(X[32:40, 0], dtype=np.float32).copy()]
        opt = make(ps)
        opt.step(grads(0))
        saved = opt.state_dict()
        start = [p.copy() for p in ps]  # glue: the one-step parameters
        for k in (1, 2):  # glue: two steps
            opt.step(grads(k))
        again = make(start)
        again.load_state_dict(saved)
        for k in (1, 2):  # glue: the same two steps
            again.step(grads(k))
        got = np.concatenate([p.reshape(-1) for p in start])
        _same_bytes(f"{tag} resumed parameters", got, f"{tag} uninterrupted parameters",
                    np.concatenate([p.reshape(-1) for p in ps]))
        parts[f"{tag}_state"] = d(np.concatenate([np.asarray(s, dtype=np.float32).reshape(-1)
                                                  for s in saved["state"]]))  # glue: the state slots
        parts[f"{tag}_resumed"] = d(got)
    return _fit(parts)


# ---------------------------------------------------------------- linear models: the per-row Mojo glue (py2mojo_rows)

@lane("x-huber-outliers")
def _(ml, X, yc, yr, Xh=None):
    """HuberRegressor.outliers_ (`py2mojo_rows` ROWS_HUBER_OUT: |y - pred| >
    scale_ * epsilon in binary64) on the `x-huber` lane's planted outliers,
    at the default epsilon and at 1.1. Parts: both masks and the fitted
    scale."""
    d = _coverage_digester(ml)
    y = yr[:2000].copy()
    y[::17] = y[::17] + np.float32(25.0)  # planted outliers
    m = ml.HuberRegressor(max_iter=30).fit(X[:2000], y)
    t = ml.HuberRegressor(max_iter=30, epsilon=1.1).fit(X[:2000], y)
    return _fit(dict(outliers=d(np.asarray(m.outliers_, dtype=np.uint8)),
                     tight=d(np.asarray(t.outliers_, dtype=np.uint8)),
                     scale=d(np.asarray([m.scale_, t.scale_], dtype=np.float64))))


@lane("x-sgd-clf-proba")
def _(ml, X, yc, yr, Xh=None):
    """SGDClassifier.predict_proba (`py2mojo_rows` ROWS_SGD_PROBA, log_loss
    only): the binary target (one sigmoid column to [1 - p, p]) and three
    classes (the one-vs-rest sigmoids divided by their row sum), on 256
    training and 256 held-out rows. Parts: the four probability tables."""
    d = _coverage_digester(ml)
    xh = (X if Xh is None else Xh)[:256]
    b = ml.SGDClassifier(loss="log_loss", max_iter=5, tol=None, random_state=3).fit(X[:2000], yc[:2000])
    m = ml.SGDClassifier(loss="log_loss", max_iter=5, tol=None, random_state=3).fit(
        X[:2000], _linear_y3(X[:2000]))
    return _fit(dict(binary=d(np.asarray(b.predict_proba(X[:256]), dtype=np.float32)),
                     binary_held=d(np.asarray(b.predict_proba(xh), dtype=np.float32)),
                     multi=d(np.asarray(m.predict_proba(X[:256]), dtype=np.float32)),
                     multi_held=d(np.asarray(m.predict_proba(xh), dtype=np.float32))))


@lane("x-logistic-cv-proba")
def _(ml, X, yc, yr, Xh=None):
    """LogisticRegressionCV.predict_proba (`py2mojo_rows` ROWS_LRCV_PROBA): the
    two-branch sigmoid on the binary target and the softmax on three classes,
    on 256 training and 256 held-out rows. Parts: the four probability
    tables."""
    d = _coverage_digester(ml)
    xh = (X if Xh is None else Xh)[:256]
    b = ml.LogisticRegressionCV(Cs=4, cv=3, max_iter=15).fit(X[:1200], yc[:1200])
    m = ml.LogisticRegressionCV(Cs=4, cv=3, max_iter=15).fit(X[:1200], _linear_y3(X[:1200]))
    return _fit(dict(binary=d(np.asarray(b.predict_proba(X[:256]), dtype=np.float32)),
                     binary_held=d(np.asarray(b.predict_proba(xh), dtype=np.float32)),
                     multi=d(np.asarray(m.predict_proba(X[:256]), dtype=np.float32)),
                     multi_held=d(np.asarray(m.predict_proba(xh), dtype=np.float32))))


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



# ---------------------------------------------------------------- lane verify-gaps-20261022: the audit's GAP rows
#
# docs/plans/VERIFY_COVERAGE_20261022.md: native entries and public methods
# new or changed since the last release that no lane reached. Each lane below
# reaches one group through the public door a user calls; they read OWED until
# a release record admits them.

@lane("saved-model-host-infer-trees")
def _(ml, X, yc, yr, Xh=None):
    """Saved Extra Trees (classifier and regressor) and a RandomForestRegressor
    on its default engine, written with `save` and predicted through
    `HostForest.from_file` and `host_predict`: the forest host binding's
    `forest_host_et_predict` and `forest_host_rf_predict_reg`, which the
    `saved-model-host-infer` lane (a random forest classifier, a GBDT and a
    groves regressor) never takes. Each host answer is held byte for byte in
    the cell to the estimator that saved it, and `host_predict` on the file to
    the loaded object. Parts: the host predictions on training rows and the
    flags."""
    d = _coverage_digester(ml)
    rows = np.ascontiguousarray(X[:4096])
    held = np.ascontiguousarray((X if Xh is None else Xh)[:512])
    etc = ml.ExtraTreesClassifier(n_estimators=16, max_depth=8, random_state=7).fit(X, yc)
    etr = ml.ExtraTreesRegressor(n_estimators=16, max_depth=8, random_state=7).fit(X, yr)
    rfr = ml.RandomForestRegressor(n_estimators=16, max_depth=8, random_state=7).fit(X, yr)
    with tempfile.TemporaryDirectory(prefix="ib-saved-model-host-trees-") as tmp:
        paths = {}
        for tag, est in (("etc", etc), ("etr", etr), ("rfr", rfr)):  # glue: three saved models
            paths[tag] = os.path.join(tmp, f"{tag}.npz")
            est.save(paths[tag])
        hetc = ml.HostForest.from_file(paths["etc"])
        hetr = ml.HostForest.from_file(paths["etr"])
        hrfr = ml.HostForest.from_file(paths["rfr"])
        one = dict(etr=np.asarray(ml.host_predict(paths["etr"], held)),
                   rfr=np.asarray(ml.host_predict(paths["rfr"], held)))
    _same_bytes("HostForest(ExtraTreesClassifier).predict_proba", np.asarray(hetc.predict_proba(held)),
                "ExtraTreesClassifier.predict_proba", np.asarray(etc.predict_proba(held)))
    _same_bytes("HostForest(ExtraTreesRegressor).predict", np.asarray(hetr.predict(held)),
                "ExtraTreesRegressor.predict", np.asarray(etr.predict(held)))
    _same_bytes("HostForest(RandomForestRegressor).predict", np.asarray(hrfr.predict(held)),
                "RandomForestRegressor.predict", np.asarray(rfr.predict(held)))
    flags = np.asarray([
        one["etr"].tobytes() == np.asarray(hetr.predict(held)).tobytes(),
        one["rfr"].tobytes() == np.asarray(hrfr.predict(held)).tobytes(),
        hetc.inference_engine == "sequential", hetr.inference_engine == "sequential",
        hrfr.inference_engine == "sequential",
        hetc.n_trees == 16, hetr.n_trees == 16, hrfr.n_trees == 16,
        list(hetc.classes_) == list(etc.classes_),
    ], dtype=np.int64)
    return _fit(dict(etc=d(np.asarray(hetc.predict_proba(rows), dtype=np.float32)),
                     etc_label=d(np.asarray(hetc.predict(rows), dtype=np.int64)),
                     etr=d(np.asarray(hetr.predict(rows), dtype=np.float32)),
                     rfr=d(np.asarray(hrfr.predict(rows), dtype=np.float32)),
                     flags=d(flags)))


#: The 64 byte tokens the `hf-causal-lm-text` vocabulary puts at ids 0..63:
#: the fixture model's vocabulary is 64 (HF_LM_VOCAB), and the prompt is
#: spelled in these bytes only, so the tokenizer never hands the model an id at
#: or above 64. The other 192 bytes follow at ids 64..255 because byte-level
#: BPE needs every byte.
_COVERAGE_TEXT_ALPHABET = b"abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 ."


def _coverage_text_tokenizers(ml):
    """The same 256-token byte-level vocabulary three ways: `Tokenizer(...)`,
    `Tokenizer._from_json` on its `tokenizer.json` object (`vocab_json_public`)
    and `Tokenizer._from_spelled` on its GPT-2 spelled vocabulary
    (`vocab_spelled_public`). Byte-level BPE needs every one of the 256 bytes
    (the vocabulary build refuses one that lacks any), so ids 0..63 are the
    64-byte alphabet, which is the fixture model's vocabulary, and ids 64..255
    carry the other 192 bytes. The prompt is spelled in the alphabet and the
    model emits ids below 64 only, so ids 64..255 are never encoded or
    decoded."""
    tk = ml.models.tokenizer
    rest = bytes(b for b in range(256) if b not in _COVERAGE_TEXT_ALPHABET)  # glue: the 192 other bytes
    tokens = [bytes([c]) for c in _COVERAGE_TEXT_ALPHABET + rest]  # glue: the 256 one-byte tokens
    spelled = ml.tokenizer._byte_to_char()
    vocab = {spelled[t[0]]: i for i, t in enumerate(tokens)}  # glue: GPT-2 spelling of each token
    data = dict(version="1.0", truncation=None, padding=None, added_tokens=[], normalizer=None,
                pre_tokenizer={"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True,
                               "use_regex": True},
                post_processor=None,
                decoder={"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True,
                         "use_regex": True},
                model=dict(type="BPE", dropout=None, unk_token=None, continuing_subword_prefix=None,
                           end_of_word_suffix=None, fuse_unk=False, byte_fallback=False,
                           ignore_merges=False, vocab=dict(vocab), merges=[]))
    direct = tk.Tokenizer(tokens, [], pattern="gpt2")
    from_json = tk.Tokenizer._from_json(data, "coverage-tokenizer.json")
    from_spelled = tk.Tokenizer._from_spelled(dict(vocab), [], pattern="gpt2", added={},
                                              normalizer=None, bos_ids=(), eos_ids=(),
                                              ignore_merges=False, source="coverage-spelled")
    return direct, from_json, from_spelled


@lane("hf-causal-lm-text")
def _(ml, X, yc, yr, Xh=None):
    """CausalLM.generate_text and stream_text (`causal_lm_generate_text`,
    `causal_lm_text_stream`) on the `hf-causal-lm` lane's Llama fixture with a
    256-token byte-level tokenizer (ids 0..63 the fixture's 64-byte alphabet)
    built three ways (`Tokenizer(...)`,
    `Tokenizer._from_json`, `Tokenizer._from_spelled`); the prompt is the
    fixture's first 24 bytes spelled in that alphabet. The text continuation is
    held byte for byte in the cell to the stream read two tokens at a time, to
    the decoded ids of `generate` on the encoded prompt, and across the three
    tokenizers; on a build with the resident decode session,
    `CausalLM._generate_resident` (`causal_lm_resident`) is held to `generate`.
    Parts: the greedy and sampled texts, the prompt ids and the greedy
    continuation ids."""
    from mojolearn import _causal_lm_fixtures as fx
    d = _coverage_digester(ml)
    A = ml.Array.from_buffer
    if len(_COVERAGE_TEXT_ALPHABET) != HF_LM_VOCAB:
        raise RuntimeError("hf-causal-lm-text: the tokenizer alphabet must be the fixture vocabulary")
    direct, from_json, from_spelled = _coverage_text_tokenizers(ml)
    raw = np.frombuffer(np.ascontiguousarray(X).tobytes()[:24], dtype=np.uint8)
    text = bytes(_COVERAGE_TEXT_ALPHABET[int(b) % 64] for b in raw).decode("ascii")  # glue: 24 prompt bytes
    prompt = np.asarray([direct.encode(text)], dtype=np.int32)
    for name, other in (("_from_json", from_json), ("_from_spelled", from_spelled)):  # glue: two loaders
        _same_bytes(f"Tokenizer.{name} encode", np.asarray([other.encode(text)], dtype=np.int32),
                    "Tokenizer(...) encode", prompt)
    cfg, tensors = fx.family_fixture("llama", False)
    sampling = dict(greedy=False, seed=7, temperature=0.8, top_k=8)
    with tempfile.TemporaryDirectory(prefix="ib-causal-lm-text-") as tmp:
        root = fx._write_checkpoint(os.path.join(tmp, "llama"), cfg, tensors)
        m = ml.models.causal_lm.CausalLM.load(root, numeric_profile="fp32_v1")
        greedy = m.generate_text(text, HF_LM_NEW, tokenizer=direct)
        sampled = m.generate_text(text, HF_LM_NEW, tokenizer=direct, **sampling)
        spelled = m.generate_text(text, HF_LM_NEW, tokenizer=from_spelled)
        stream = m.stream_text(text, HF_LM_NEW, tokenizer=from_json)
        pieces = [stream.read(2) for _ in range(-(-HF_LM_NEW // 2) + 1)]  # glue: bounded reads
        full = np.asarray(m.generate(A(np.ascontiguousarray(prompt)), HF_LM_NEW), dtype=np.int32)
        resident = m._generate_resident(A(np.ascontiguousarray(prompt)), HF_LM_NEW,
                                        prompt.shape[1] + HF_LM_NEW)
    cont = np.ascontiguousarray(full[:, prompt.shape[1]:])
    _same_bytes("stream_text read two at a time", _hf_text("".join(pieces)),
                "generate_text", _hf_text(greedy))
    _same_bytes("generate_text through _from_spelled", _hf_text(spelled), "generate_text", _hf_text(greedy))
    _same_bytes("decoded generate continuation", _hf_text(direct.decode(cont[0].tolist())),
                "generate_text", _hf_text(greedy))
    if resident is not None:
        _same_bytes("CausalLM._generate_resident", np.asarray(resident, dtype=np.int32),
                    "CausalLM.generate", full)
    return _fit(dict(greedy=d(_hf_text(greedy)), sampled=d(_hf_text(sampled)),
                     prompt=d(prompt), continuation=d(cont)))


@lane("hf-model-surface")
def _(ml, X, yc, yr, Xh=None):
    """The `mojolearn.models` reader surface no other lane calls: `HFConfig`
    built from a dict and `HFConfig.from_json` on a written `config.json` with
    every accessor (`model_config_init`, `model_config_load`,
    `model_config_access`), `ModelPlan(**fields)` rebuilt from `plan_for`'s
    plan with `layer_weights` and `checkpoint_names` (`model_plan_init`,
    `model_layer_weights`, `model_checkpoint_names`), `TensorInfo` built by
    hand from each header entry with `nbytes` and `size` (`tensor_info_init`,
    `tensor_info_size`), `SafetensorsFile._map` (`safetensors_map`),
    `Checkpoint._file` on a shard (`checkpoint_file`),
    `CausalLM._read_weights` (`causal_model_read_weights`) and `CausalLM(plan,
    weights)` with a column-major head (`causal_model_init`,
    `causal_lm_gather_f32`). The tensors are the fixture's own bytes (F32
    values, F16 bit patterns). Parts: the config as JSON text, the plan
    table, the header table, the mapped file bytes, the shard reads, the
    top-level weights read, both models' logits and the flags."""
    from mojolearn import _causal_lm_fixtures as fx
    d = _coverage_digester(ml)
    M = ml.models
    st = M.safetensors
    raw = np.ascontiguousarray(X).tobytes()
    bits = np.frombuffer(raw[:512], dtype="<u2")
    f32 = np.frombuffer(raw[512:512 + 256 * 4], dtype="<f4")
    tensors = {"t.f32": ("F32", (16, 16), f32.tobytes()), "t.f16": ("F16", (16, 16), bits.tobytes()),
               "t.f32s": ("F32", (), f32[:1].tobytes())}
    cfg0 = fx._llama_config()
    with tempfile.TemporaryDirectory(prefix="ib-hf-model-surface-") as tmp:
        one = fx._write_checkpoint(os.path.join(tmp, "one"), cfg0, tensors, shards=1)
        many = fx._write_checkpoint(os.path.join(tmp, "many"), cfg0, tensors, shards=2)
        built = M.HFConfig(dict(cfg0), os.path.join(one, "config.json"))
        loaded = M.HFConfig.from_json(one)
        config_text = json.dumps(loaded.to_dict(), sort_keys=True)
        plan = M.plan_for(loaded)
        copy = M.ModelPlan(**dict(vars(plan)))
        plan_text = json.dumps(_hf_plan_row(copy), sort_keys=True)
        sf = st.SafetensorsFile(os.path.join(one, "model.safetensors"))
        ck = st.Checkpoint.open(many)
        try:
            names = sorted(sf.names())
            infos = [sf.info(n) for n in names]  # glue: one header entry per tensor
            rebuilt = [st.TensorInfo(t.name, t.dtype, t.shape, t.begin, t.end, t.file) for t in infos]  # glue: by hand
            header = "|".join(f"{t.name},{t.dtype},{t.shape},{t.begin},{t.end},{t.nbytes},{t.size}"
                              for t in rebuilt)
            mapped = np.frombuffer(bytes(sf._map()), dtype=np.uint8).copy()
            shard_bytes = np.concatenate([_hf_bytes(ck._file(ck.info(n).file).read(n)) for n in names])
            whole_bytes = np.concatenate([_hf_bytes(ck.read(n)) for n in names])
            flags = np.asarray([
                built.model_type == loaded.model_type == "llama",
                built.get("hidden_size") == loaded["hidden_size"] == cfg0["hidden_size"],
                built.get("no_such_field", 7) == 7,
                "vocab_size" in loaded and "no_such_field" not in loaded,
                sorted(loaded.keys()) == sorted(cfg0),
                built.to_dict() == loaded.to_dict(),
                _hf_refused(lambda: M.HFConfig([1, 2], None), "JSON object"),
                _hf_refused(lambda: loaded["no_such_field"], "no_such_field"),
                copy.checkpoint_names() == plan.checkpoint_names(),
                all(copy.layer_weights(i) == plan.layer_weights(i) for i in range(plan.n_layers)),
                all((a.nbytes, a.size) == (b.nbytes, b.size) for a, b in zip(infos, rebuilt)),
                _hf_refused(lambda: st.TensorInfo("t", "F32", (4,), 0, 15, "f"), "bytes differ"),
                ck._file(ck.info(names[0]).file) is ck._file(ck.info(names[0]).file),
                shard_bytes.tobytes() == whole_bytes.tobytes(),
            ], dtype=np.int64)
        finally:
            sf.close()
            ck.close()
        lcfg, ltensors = fx.family_fixture("llama", False)
        root = fx._write_checkpoint(os.path.join(tmp, "llama"), lcfg, ltensors)
        lplan = M.plan_for(M.HFConfig.from_json(root))
        lck = st.Checkpoint.open(root)
        try:
            weights = M.causal_lm.CausalLM._read_weights(lck, lplan, "float32")
        finally:
            lck.close()
    top = np.concatenate([np.asarray(weights[k], dtype=np.float32).reshape(-1)
                          for k in ("embed", "norm", "head") if weights.get(k) is not None])
    # The read weights assembled into a model twice: as read, and with the
    # untied head handed over as a column-major Array, which the model
    # gathers into C order (`causal_lm_gather_f32`). The two forwards are
    # separate parts, so a gather that moves a value moves the cell.
    A = ml.Array.from_buffer
    ids = _ids(X, HF_LM_BATCH, HF_LM_LEN, HF_LM_VOCAB)
    fortran = dict(weights)
    fortran["head"] = A(np.asfortranarray(np.asarray(weights["head"], dtype=np.float32)))
    logits = np.asarray(M.causal_lm.CausalLM(lplan, weights, numeric_profile="fp32_v1").forward(A(ids)),
                        dtype=np.float32)
    gathered = np.asarray(M.causal_lm.CausalLM(lplan, fortran, numeric_profile="fp32_v1").forward(A(ids)),
                          dtype=np.float32)
    return _fit(dict(config=d(_hf_text(config_text)), plan=d(_hf_text(plan_text)),
                     header=d(_hf_text(header)), mapped=d(mapped), shards=d(shard_bytes),
                     weights=d(np.ascontiguousarray(top)), logits=d(logits), gathered=d(gathered),
                     flags=d(flags)))


@lane("training-device-cross-entropy")
def _(ml, X, yc, yr, Xh=None):
    """training.cross_entropy on device-resident logits (`ce_loss_dev`): a
    64 x 16 activation and a 40 x 16 weight uploaded once
    (`train_to_device`), the logits made on the device by `linear_forward`
    (`linear_forward_dev`), the loss and its gradient by `cross_entropy` with
    `return_grad=True`, and the input and weight gradients by
    `linear_backward` (`linear_backward_dev`). Every device result is held
    byte for byte in the cell to the same calls on host arrays, which is the
    device path's claim (the host entries' own arithmetic). GPU columns only:
    the CPU training binding has no device pool. Parts: the loss, the logits
    gradient and both linear gradients."""
    from mojolearn._training_impl import train_to_device
    T = ml.training
    d = _coverage_digester(ml)
    a = np.ascontiguousarray(X[:64, :16], dtype=np.float32)
    w = np.ascontiguousarray(X[64:104, :16], dtype=np.float32) * np.float32(0.25)
    t = np.ascontiguousarray(_ids(X, 1, 64, 40).reshape(-1)).astype(np.int64)
    logits = np.asarray(T.linear_forward(a, w), dtype=np.float32)
    loss, grad = T.cross_entropy(logits, t, return_grad=True, label_smoothing=0.1)
    grad = np.ascontiguousarray(np.asarray(grad, dtype=np.float32))
    da, dw = T.linear_backward(grad, a, w)
    a_dev, w_dev = train_to_device(a), train_to_device(w)
    logits_dev = T.linear_forward(a_dev, w_dev)
    loss_dev, grad_dev = T.cross_entropy(logits_dev, t, return_grad=True, label_smoothing=0.1)
    da_dev, dw_dev = T.linear_backward(grad_dev, a_dev, w_dev)
    host = lambda v: np.asarray(v.to_host(), dtype=np.float32)
    _same_bytes("device logits", host(logits_dev), "host logits", logits)
    _same_bytes("device loss", np.asarray([loss_dev], dtype=np.float64), "host loss",
                np.asarray([loss], dtype=np.float64))
    _same_bytes("device logits gradient", host(grad_dev), "host logits gradient", grad)
    _same_bytes("device input gradient", host(da_dev), "host input gradient", np.asarray(da, dtype=np.float32))
    _same_bytes("device weight gradient", host(dw_dev), "host weight gradient", np.asarray(dw, dtype=np.float32))
    return _fit(dict(loss=d(np.asarray([loss_dev], dtype=np.float64)), grad=d(host(grad_dev)),
                     da=d(host(da_dev)), dw=d(host(dw_dev))))


GPU_ONLY_LANES["training-device-cross-entropy"] = "the CPU training binding has no device array pool"


@lane("sequence-layernorm-backward")
def _(ml, X, yc, yr, Xh=None):
    """mojolearn.layer_norm_backward, the functional gradient of
    `layer_norm_forward`: 256 x 2 x 8 fixture values normalized over (2, 8)
    with weight and bias (eps 1e-3), and 512 x 16 over the last axis with
    neither; dy from later fixture rows. The weighted call's dx is held byte
    for byte in the cell to `LayerNorm.backward` on the same values, which is
    the functional form's claim. Parts: dx, dweight and dbias of the affine
    call, dx of the plain one."""
    d = _coverage_digester(ml)
    x3 = np.ascontiguousarray(X[:256, :16], dtype=np.float32).reshape(256, 2, 8)
    dy3 = np.ascontiguousarray(X[256:512, :16], dtype=np.float32).reshape(256, 2, 8)
    wgt = (np.float32(1.0) + np.float32(0.125) * np.ascontiguousarray(X[600, :16], dtype=np.float32)).reshape(2, 8)
    bias = np.ascontiguousarray(X[601, :16], dtype=np.float32).reshape(2, 8)
    dx, dwgt, dbias = ml.layer_norm_backward(dy3, x3, (2, 8), weight=wgt, bias=bias, eps=1e-3)
    x2 = np.ascontiguousarray(X[:512, :16], dtype=np.float32)
    dy2 = np.ascontiguousarray(X[512:1024, :16], dtype=np.float32)
    dx2, nw, nb = ml.layer_norm_backward(dy2, x2)
    if nw is not None or nb is not None:
        raise RuntimeError("layer_norm_backward without weight or bias returned their gradients")
    ln = ml.LayerNorm(16)
    ln.weight[:] = np.float32(1.0) + np.float32(0.125) * np.ascontiguousarray(X[600, :16], dtype=np.float32)
    ln.bias[:] = np.ascontiguousarray(X[601, :16], dtype=np.float32)
    ln(x2)
    _same_bytes("LayerNorm.backward dx", np.asarray(ln.backward(dy2), dtype=np.float32),
                "layer_norm_backward dx (weight, bias)",
                np.asarray(ml.layer_norm_backward(dy2, x2, weight=np.ascontiguousarray(ln.weight),
                                                  bias=np.ascontiguousarray(ln.bias))[0], dtype=np.float32))
    return _fit(dict(dx=d(np.asarray(dx, dtype=np.float32)), dweight=d(np.asarray(dwgt, dtype=np.float32)),
                     dbias=d(np.asarray(dbias, dtype=np.float32)), dx_plain=d(np.asarray(dx2, dtype=np.float32))))


@lane("x-metrics-classification-report")
def _(ml, X, yc, yr, Xh=None):
    """metrics.classification_report on the `x-metrics-classification` lane's
    four-class labels and binary targets: the dict form unweighted and
    weighted, and the text form at four digits with target names. Parts: the
    dict's numbers in sorted key order and the report text."""
    d = _coverage_digester(ml)
    mt = ml.metrics
    n = 3000
    yt, yp = _metrics_labels(X, n)
    w = _metrics_weights(n, "x-metrics-classification-report:w")
    bt, bp = _metrics_binary(yc, X, n)

    def numbers(report):
        out = []
        for key in sorted(report):  # glue: the report's rows in key order
            row = report[key]
            if isinstance(row, dict):
                out.extend(float(row[k]) for k in sorted(row))  # glue: one row's fields
            else:
                out.append(float(row))
        return np.asarray(out, dtype=np.float64)

    parts = dict(dict_u=d(numbers(mt.classification_report(yt, yp, output_dict=True, zero_division=0))),
                 dict_w=d(numbers(mt.classification_report(yt, yp, sample_weight=w, output_dict=True,
                                                           zero_division=0))),
                 binary=d(numbers(mt.classification_report(bt, bp, output_dict=True, zero_division=0))),
                 text=d(_hf_text(mt.classification_report(yt, yp, digits=4, zero_division=0,
                                                          target_names=["a", "b", "c", "d"]))))
    return _fit(parts)


@lane("x-metrics-permutation-test-score")
def _(ml, X, yc, yr, Xh=None):
    """model_selection.permutation_test_score: a Ridge on 600 fixture rows,
    three folds, 12 permutations from the counter generator (random_state
    3), the default score; and the same with groups (permutations within
    each of four groups). Parts: each call's score, permutation scores and
    p-value."""
    d = _coverage_digester(ml)
    ms = ml.model_selection
    x = np.ascontiguousarray(X[:600, :8], dtype=np.float32)
    y = np.ascontiguousarray(yr[:600], dtype=np.float32)
    groups = (np.arange(600) % 4).astype(np.int64)
    parts = {}
    for tag, kw in (("plain", {}), ("groups", dict(groups=groups))):  # glue: two calls
        score, perm, p = ms.permutation_test_score(ml.Ridge(alpha=1.0), x, y, cv=ms.KFold(3),
                                                   n_permutations=12, random_state=3, **kw)
        parts[tag] = d(np.concatenate([np.asarray([score, p], dtype=np.float64),
                                       np.asarray(perm, dtype=np.float64)]))
    return _fit(parts)


@lane("classical-float16-sequences")
def _(ml, X, yc, yr, Xh=None):
    """LinearRegression, Ridge, StandardScaler, MinMaxScaler and PCA fitted on
    a Python list of float16 row arrays: the classical input door widens each
    row's half words (`sequence_widen_halves`) and stores the promoted half
    matrix (`sequence_narrow_halves`), and the scalers cast that half matrix
    to float32 (`scaler_cast`). Each linear fit is held byte for byte in the
    cell to the fit on the same values as one float16 ndarray, and each
    scaler to the fit on the float32 ndarray of the same values, which is the
    door's claim (the half to float32 widening is exact). Parts: the linear
    models' coefficients, intercepts and predictions, the scalers' statistics
    and transforms, and the PCA components and transform."""
    d = _coverage_digester(ml)
    xh = np.ascontiguousarray(X[:512, :8], dtype=np.float32).astype(np.float16)
    rows = [np.ascontiguousarray(r) for r in xh]  # glue: the input as a list of half rows
    y = np.ascontiguousarray(yr[:512], dtype=np.float32)
    parts = {}
    for tag, cls, kw in (("ols", ml.LinearRegression, {}), ("ridge", ml.Ridge, dict(alpha=0.5))):  # glue: two models
        listed = cls(**kw).fit(rows, y)
        whole = cls(**kw).fit(xh, y)
        coef = np.asarray(listed.coef_, dtype=np.float64)
        _same_bytes(f"{tag} coef_ from a list of float16 rows", coef,
                    f"{tag} coef_ from a float16 ndarray", np.asarray(whole.coef_, dtype=np.float64))
        parts[f"{tag}_coef"] = d(coef)
        parts[f"{tag}_intercept"] = d(np.asarray([listed.intercept_], dtype=np.float64))
        parts[f"{tag}_predict"] = d(np.asarray(listed.predict(xh[:64].astype(np.float32)), dtype=np.float64))
    x32 = np.ascontiguousarray(xh.astype(np.float32))
    for tag, cls, attrs in (("standard", ml.StandardScaler, ("mean_", "var_", "scale_")),  # glue: two scalers
                            ("minmax", ml.MinMaxScaler, ("data_min_", "data_max_", "scale_", "min_"))):
        listed = cls().fit(rows)
        whole = cls().fit(x32)
        stats = np.concatenate([np.asarray(getattr(listed, a), dtype=np.float64).reshape(-1) for a in attrs])
        _same_bytes(f"{tag} scaler statistics from a list of float16 rows", stats,
                    f"{tag} scaler statistics from the float32 ndarray",
                    np.concatenate([np.asarray(getattr(whole, a), dtype=np.float64).reshape(-1) for a in attrs]))
        parts[f"{tag}_stats"] = d(stats)
        parts[f"{tag}_transform"] = d(np.asarray(listed.transform(x32[:64]), dtype=np.float32))
    p = ml.PCA(n_components=3, svd_solver="full").fit(rows)
    parts["pca_components"] = d(np.asarray(p.components_, dtype=np.float32))
    parts["pca_transform"] = d(np.asarray(p.transform(x32[:64]), dtype=np.float32))
    return _fit(parts)


_batch_decl("n/a:coverage-probe (the lane checks one public call path the release audit found no lane for; "
            "its outputs are train parts, digested whole)",
            "gp-lml-theta", "cross-vendor-fold", "sequence-moe-backward", "sequence-export-roundtrip",
            "sequence-onnx-import", "transformer-decode-load-state", "safetensors-widen-f16",
            "sequence-program-save-load", "hf-causal-lm-stream", "ivf-short-fill", "sequence-device-tensor",
            "mamba2-decode-session", "mamba3-decode-session", "training-chunked-lm-head",
            "training-mlp-sessions", "samba-optimizer-state", "sequence-optimizer-state",
            "sequence-adafactor-device", "sequence-lamb-device",
            "x-huber-outliers", "x-sgd-clf-proba", "x-logistic-cv-proba",
            "x-decomp-rsvd-rank-deficient",
            # lane verify-gaps-20261022
            "saved-model-host-infer-trees", "hf-causal-lm-text", "hf-model-surface",
            "training-device-cross-entropy", "sequence-layernorm-backward",
            "x-metrics-classification-report", "x-metrics-permutation-test-score",
            "classical-float16-sequences")
