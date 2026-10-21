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
    copies its cache into the state, four more tokens are stepped, then
    `load_state` rewinds the session to the four-token cache and four other
    tokens are stepped. Those last outputs are held byte for byte in the cell
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


_batch_decl("n/a:coverage-probe (the lane checks one public call path the release audit found no lane for; "
            "its outputs are train parts, digested whole)",
            "gp-lml-theta", "cross-vendor-fold", "sequence-moe-backward", "sequence-export-roundtrip",
            "sequence-onnx-import", "transformer-decode-load-state", "safetensors-widen-f16")
