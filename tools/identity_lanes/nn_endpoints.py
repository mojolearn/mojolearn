# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE NEURAL LAYERS' AND MODELS' ENDPOINT LANES (lane vdv-cov-nn, 2026-11-24).
#
# Verify through `device=` (docs/plans/VERIFY_THROUGH_DEVICE.md, item 3): a lane for every public
# neural endpoint with a numerical result that no lane called by name: `__call__`, `state_dict` /
# `load_state_dict` (a round trip: the state out, loaded into a FRESH object built from other
# weights on the same side, the answer held equal byte for byte in the lane), `parameters` /
# `named_parameters`, `to`, the state containers, the CNN layers' modes, setters and fits, the
# recurrent estimators' state dicts, the Sequential builders and checkpoints, Embedding, LayerNorm,
# MoEBlock and the Samba stack, config and CPU inference class.
#
# Every lane is device-agnostic (contract 1): no `device=`; the harness's view gives every
# constructor and loader the side. `_lane_side()` is read in one place only, as the argument of
# `.to(...)` (rebuild the object on the side it already runs on; the answer must not move). A
# `__call__` endpoint is called as `obj.__call__(x)` so the coverage audit's receiver walk
# (tools/verification_endpoint_calls.py) sees the method by name; it is the same call as `obj(x)`.
#
# Loaded by tools/identity_break.py `_load_lane_fragments` in the harness namespace: np, _h, _fit,
# _same_bytes, _block_weights, _seq, _ids, _hw, _three_class, _sequence_seq, _sequence_targets,
# _cnn_graph, _cnn_grad, _lane_side, _one_sided and `lane` are the harness's (or an earlier
# fragment's, read when a lane runs); helpers here are prefixed `_nn_endpoints_`. Each lane builds
# its own small inputs and objects and shares nothing mutable with another lane. Every reference is
# OWED (recorded once, at the release's one verification).


def _nn_endpoints_host(a):
    """A returned tensor as a host array: a device tensor's `numpy()` / `to_host()`, else the array."""
    for name in ("numpy", "to_host"):
        read = getattr(a, name, None)
        if callable(read):
            return np.asarray(read())
    return np.asarray(a)


def _nn_endpoints_named(pairs):
    """(names, arrays) of a `named_parameters()` iterator, in its order."""
    names, arrays = [], []
    for name, value in pairs:
        names.append(str(name))
        arrays.append(np.ascontiguousarray(np.asarray(value)))
    return names, arrays


def _nn_endpoints_state(sd):
    """The arrays of a {name: array} state dict in sorted-name order, and the names."""
    names = sorted(sd)
    return names, [np.ascontiguousarray(np.asarray(sd[k])) for k in names]


def _nn_endpoints_check_order(what, names, sd):
    if list(names) != list(sd):
        raise AssertionError(f"{what}: named_parameters order {names} is not the state_dict order {list(sd)}")


# ---------------------------------------------------------------- the sequence blocks

_NN_ENDPOINTS_MAMBA_SHAPES = {
    "mamba1": (dict(**{"norm.weight": (32,), "in_proj.weight": (128, 32), "conv1d.weight": (64, 1, 4),
                       "conv1d.bias": (64,), "x_proj.weight": (34, 64), "dt_proj.weight": (64, 2),
                       "dt_proj.bias": (64,), "A_log": (64, 16), "D": (64,), "out_proj.weight": (32, 64)}),
               ("norm.weight",), ()),
    "mamba2": (dict(**{"block_norm.weight": (32,), "in_proj.weight": (385, 32), "conv1d.weight": (320, 1, 4),
                       "conv1d.bias": (320,), "dt_bias": (1,), "A_log": (1,), "D": (1,), "norm.weight": (64,),
                       "out_proj.weight": (32, 64)}),
               ("block_norm.weight", "norm.weight"), ()),
    "mamba3": (dict(**{"block_norm.weight": (32,), "in_proj.weight": (419, 32), "dt_bias": (1,),
                       "B_norm.weight": (128,), "C_norm.weight": (128,), "B_bias": (1, 128), "C_bias": (1, 128),
                       "D": (1,), "out_proj.weight": (32, 64)}),
               ("block_norm.weight",), ("B_norm.weight", "C_norm.weight")),
}


def _nn_endpoints_mamba_weights(kind, label):
    shapes, ones, near = _NN_ENDPOINTS_MAMBA_SHAPES[kind]
    return _block_weights(label, shapes, ones=ones, near_one=near)


def _nn_endpoints_block_round_trip(tag, blk, fresh, x, parts):
    """`__call__`, `state_dict` into `fresh` by `load_state_dict`, `parameters`, `named_parameters`
    and `to` of one block; the reloaded and the moved block must answer `x` as `blk` does."""
    y = _nn_endpoints_host(blk.__call__(x))
    sd = blk.state_dict()
    fresh.load_state_dict(sd)
    _same_bytes(f"{tag} forward", y, f"{tag} forward after load_state_dict", _nn_endpoints_host(fresh.forward(x)))
    names, named = _nn_endpoints_named(blk.named_parameters())
    _nn_endpoints_check_order(tag, names, sd)
    flat = [np.ascontiguousarray(np.asarray(p)) for p in blk.parameters()]
    blk.to(_lane_side())
    _same_bytes(f"{tag} forward", y, f"{tag} forward after to()", _nn_endpoints_host(blk.forward(x)))
    parts[tag + "_call"] = _h(y)
    parts[tag + "_state"] = _h(*_nn_endpoints_state(sd)[1])
    parts[tag + "_parameters"] = _h(*flat)
    parts[tag + "_named"] = _h(*named)


@lane("nn-ep-mamba-blocks")
def _(ml, X, yc, yr, Xh=None):
    """Mamba1Block, Mamba2Block and Mamba3Block at the base lanes' shapes (d_model 32): `__call__`
    on a (2, 16, 32) fixture slab, the state dict round trip into a block built from other weights,
    `parameters`, `named_parameters` (order = state_dict order), `to` (the answer must not move);
    and each block's state CONSTRUCTED by the caller (`Mamba1State`, `Mamba2State`, `Mamba3State`
    from zero float32 arrays of the allocated shapes): a prefill into it must equal the prefill into
    an allocated state. Parts per block: the call, the state, the parameters, the constructed-state
    prefill."""
    x = _seq(X, 2, 16, 32)
    parts = {}
    b1 = ml.Mamba1Block(_nn_endpoints_mamba_weights("mamba1", "nn-ep-mamba1"))
    f1 = ml.Mamba1Block(_nn_endpoints_mamba_weights("mamba1", "nn-ep-mamba1-other"))
    _nn_endpoints_block_round_trip("mamba1", b1, f1, x, parts)
    s1 = b1.allocate_state(2)
    own1 = ml.Mamba1State(np.zeros(np.asarray(s1.conv_window).shape, dtype=np.float32),
                          np.zeros(np.asarray(s1.h).shape, dtype=np.float32))
    p1 = _nn_endpoints_host(b1.forward(x, own1))
    _same_bytes("mamba1 prefill (Mamba1State)", p1, "mamba1 prefill (allocate_state)", _nn_endpoints_host(b1.forward(x, s1)))
    parts["mamba1_own_state"] = _h(p1, np.asarray(own1.h))
    b2 = ml.Mamba2Block(_nn_endpoints_mamba_weights("mamba2", "nn-ep-mamba2"))
    f2 = ml.Mamba2Block(_nn_endpoints_mamba_weights("mamba2", "nn-ep-mamba2-other"))
    _nn_endpoints_block_round_trip("mamba2", b2, f2, x, parts)
    s2 = b2.allocate_state(2)
    own2 = ml.Mamba2State(*[np.zeros(np.asarray(getattr(s2, k)).shape, dtype=np.float32)
                            for k in ("conv_window", "h", "buffer_xbc", "buffer_dtraw")])
    p2 = _nn_endpoints_host(b2.forward(x, own2))
    _same_bytes("mamba2 prefill (Mamba2State)", p2, "mamba2 prefill (allocate_state)", _nn_endpoints_host(b2.forward(x, s2)))
    parts["mamba2_own_state"] = _h(p2, np.asarray(own2.h))
    b3 = ml.Mamba3Block(_nn_endpoints_mamba_weights("mamba3", "nn-ep-mamba3"))
    f3 = ml.Mamba3Block(_nn_endpoints_mamba_weights("mamba3", "nn-ep-mamba3-other"))
    _nn_endpoints_block_round_trip("mamba3", b3, f3, x, parts)
    s3 = b3.allocate_state(2)
    fields3 = ("theta", "h", "buffer_qrot", "buffer_krot", "buffer_v", "buffer_dt", "buffer_sig", "buffer_adt",
               "pending_k", "pending_v")
    own3 = ml.Mamba3State(*[np.zeros(np.asarray(getattr(s3, k)).shape, dtype=np.float32) for k in fields3])
    p3 = _nn_endpoints_host(b3.forward(x, own3))
    _same_bytes("mamba3 prefill (Mamba3State)", p3, "mamba3 prefill (allocate_state)", _nn_endpoints_host(b3.forward(x, s3)))
    # set_input_states: a reference Input_States continuation, values cut from the fixture (theta
    # folded into [0, 2 pi)); the next prefill consumes them.
    cont = ml.Mamba3State(*[np.zeros(np.asarray(getattr(s3, k)).shape, dtype=np.float32) for k in fields3])
    flat = np.abs(np.ascontiguousarray(X, dtype=np.float32).reshape(-1))
    offset = [0]

    def take(shape, scale):
        n = int(np.prod(shape))
        out = (flat[offset[0]:offset[0] + n] * np.float32(scale)).reshape(shape).astype(np.float32)
        offset[0] += n
        return np.ascontiguousarray(out)

    theta = np.ascontiguousarray(np.mod(take(np.asarray(s3.theta).shape, 1.0), np.float32(6.0)))
    cont.set_input_states(theta, take(np.asarray(s3.h).shape, 0.01), take(np.asarray(s3.pending_k).shape, 0.1),
                          take(np.asarray(s3.pending_v).shape, 0.1))
    parts["mamba3_own_state"] = _h(p3, _nn_endpoints_host(b3.forward(x, cont)), np.asarray(cont.h))
    return _fit(parts)


@lane("nn-ep-mamba3-owned")
def _(ml, X, yc, yr, Xh=None):
    """Mamba3Block's owned weight endpoints as the product offers them: the owned snapshot arm
    (MOJOLEARN_NN40_OWNED_WEIGHTS) was deleted 2026-11-15 (core/experiment_guards_neural.mojo), so
    every image, GPU and CPU, reports no MAMBA3_CAP_OWNED_WEIGHTS and `install_owned_weights`
    refuses by name (bindings/mamba_public_native_door.mojo mamba3_install_owned). On each side:
    the forward; `install_owned_weights` must refuse naming owned weights and leave the block
    working (the forward after it equal); `export_owned_weights` must refuse "no installed owned
    snapshot"; `release_owned_weights` releases nothing and the forward after it is equal again.
    An image that admitted the snapshot fails the lane (the capability would be untested here).
    Parts: the forward. (w3-fix-11, 2026-11-25: the earlier body asked for a snapshot no build
    provides, so its clean run was REFUSED on both sides.)"""
    w = _nn_endpoints_mamba_weights("mamba3", "nn-ep-mamba3-owned")
    blk = ml.Mamba3Block(w)
    x = _seq(X, 2, 16, 32)
    y = _nn_endpoints_host(blk.forward(x))
    try:
        blk.install_owned_weights()
    except RuntimeError as exc:
        if "owned weights" not in str(exc):
            raise
    else:
        blk.release_owned_weights()
        raise AssertionError("install_owned_weights was admitted: an owned-weight image needs its own lane")
    _same_bytes("forward", y, "forward after the refused install", _nn_endpoints_host(blk.forward(x)))
    try:
        blk.export_owned_weights()
    except RuntimeError as exc:
        if "no installed owned snapshot" not in str(exc):
            raise
    else:
        raise AssertionError("export_owned_weights answered without an installed snapshot")
    blk.release_owned_weights()
    _same_bytes("forward", y, "forward after release_owned_weights", _nn_endpoints_host(blk.forward(x)))
    return _fit(dict(forward=_h(y)))


@lane("nn-ep-transformer-block")
def _(ml, X, yc, yr, Xh=None):
    """TransformerBlock at the `transformer` lane's shapes (d_model 32, 2 heads, 1 kv head): `__call__`,
    the state dict round trip into a block from other weights, `parameters`, `named_parameters`, `to`;
    a TransformerState CONSTRUCTED from zero caches (its prefill must equal the allocated state's),
    and the state's `keys()` / `values()` after the prefill. Parts: the call, the state, the
    parameters, the constructed-state prefill and the caches."""
    shapes = {"input_layernorm.weight": (32,), "post_attention_layernorm.weight": (32,),
              "q_proj.weight": (32, 32), "k_proj.weight": (16, 32), "v_proj.weight": (16, 32),
              "o_proj.weight": (32, 32), "gate_proj.weight": (64, 32), "up_proj.weight": (64, 32),
              "down_proj.weight": (32, 64)}
    near = ("input_layernorm.weight", "post_attention_layernorm.weight")
    blk = ml.TransformerBlock(_block_weights("nn-ep-transformer", shapes, near_one=near), n_heads=2, n_kv_heads=1)
    fresh = ml.TransformerBlock(_block_weights("nn-ep-transformer-other", shapes, near_one=near),
                                n_heads=2, n_kv_heads=1)
    x = _seq(X, 2, 16, 32)
    parts = {}
    _nn_endpoints_block_round_trip("transformer", blk, fresh, x, parts)
    st = blk.allocate_state(2, max_tokens=32)
    own = ml.TransformerState(2, 1, 16, 32, np.zeros(np.asarray(st.k_cache).shape, dtype=np.float32),
                              np.zeros(np.asarray(st.v_cache).shape, dtype=np.float32))
    y_own = _nn_endpoints_host(blk.forward(x, own))
    _same_bytes("transformer prefill (TransformerState)", y_own, "transformer prefill (allocate_state)",
                _nn_endpoints_host(blk.forward(x, st)))
    parts["own_state"] = _h(y_own, np.asarray(own.keys()), np.asarray(own.values()))
    return _fit(parts)


# ---------------------------------------------------------------- LayerNorm, MoEBlock, Embedding

@lane("nn-ep-layernorm")
def _(ml, X, yc, yr, Xh=None):
    """LayerNorm over the 16 columns of 256 rows (fixture-derived weight and bias): `__call__`, and
    `to_device` of the input and of dy: the forward and backward from the uploaded arrays must equal
    the host forward and backward (on a side without a device path `to_device` returns the array).
    Parts: the output, dx and the parameter gradients."""
    x = np.ascontiguousarray(X[:256, :16], dtype=np.float32)
    dy = np.ascontiguousarray(X[256:512, :16], dtype=np.float32)
    ln = ml.LayerNorm(16)
    ln.weight[:] = np.float32(1.0) + np.float32(0.125) * np.ascontiguousarray(X[512, :16], dtype=np.float32)
    ln.bias[:] = np.ascontiguousarray(X[513, :16], dtype=np.float32)
    y = _nn_endpoints_host(ln.__call__(x))
    dx = _nn_endpoints_host(ln.backward(dy))
    grads = (np.asarray(ln.weight_grad).copy(), np.asarray(ln.bias_grad).copy())
    yd = _nn_endpoints_host(ln.forward(ln.to_device(x)))
    dxd = _nn_endpoints_host(ln.backward(ln.to_device(dy)))
    _same_bytes("LayerNorm forward", y, "LayerNorm forward from to_device", yd)
    _same_bytes("LayerNorm backward", dx, "LayerNorm backward from to_device", dxd)
    return _fit(dict(y=_h(y), dx=_h(dx), grads=_h(*grads)))


@lane("nn-ep-moe")
def _(ml, X, yc, yr, Xh=None):
    """MoEBlock (16 -> 24, 6 experts, top 2) over 128 fixture tokens: `__call__` and `forward`;
    `load_state_dict` of its router, gate_up_proj and down_proj into a block of another seed (the
    answer must equal); `to_device` of the input (the forward from it must equal the host forward).
    Parts: the output, the routing, the loaded block's output."""
    x = np.ascontiguousarray(X[:128, :16], dtype=np.float32)
    a = ml.MoEBlock(16, 24, num_experts=6, top_k=2, random_state=3)
    y = _nn_endpoints_host(a.__call__(x))
    routing = (np.asarray(a.selected_experts_), np.asarray(a.routing_weights_))
    b = ml.MoEBlock(16, 24, num_experts=6, top_k=2, random_state=4)
    b.load_state_dict({"router": np.asarray(a.router).copy(), "gate_up_proj": np.asarray(a.gate_up_proj).copy(),
                       "down_proj": np.asarray(a.down_proj).copy()})
    yb = _nn_endpoints_host(b.forward(x))
    _same_bytes("MoEBlock forward", y, "MoEBlock forward after load_state_dict", yb)
    yd = _nn_endpoints_host(a.forward(a.to_device(x)))
    _same_bytes("MoEBlock forward", y, "MoEBlock forward from to_device", yd)
    return _fit(dict(y=_h(y), routing=_h(*routing), loaded=_h(yb)))


@lane("nn-ep-embedding")
def _(ml, X, yc, yr, Xh=None):
    """Embedding.from_pretrained over a hashed (64, 8) table with padding_idx 3: `__call__` on 256
    fixture ids, `backward` with a host dy and with `to_device(dy)` (the dense gradients must agree).
    Parts: the lookup and the gradient."""
    w = _hw((64, 8), "nn-ep-embedding:w", -0.25, 0.25)
    ids = (_ids(X, 1, 256).reshape(256) % 64).astype(np.int32)
    dy = np.ascontiguousarray(_seq(X, 1, 256, 8).reshape(256, 8))
    e = ml.Embedding.from_pretrained(w, padding_idx=3)
    y = _nn_endpoints_host(e.__call__(ids))
    dw = _nn_endpoints_host(e.backward(ids, dy))
    dwd = _nn_endpoints_host(e.backward(ids, e.to_device(dy)))
    _same_bytes("Embedding backward", dw, "Embedding backward from to_device(dy)", dwd)
    return _fit(dict(y=_h(y), dw=_h(dw)))


# ---------------------------------------------------------------- the CNN layers

def _nn_endpoints_layer_round_trip(tag, layer, fresh, run, parts):
    """state_dict -> fresh.load_state_dict, parameters, named_parameters of a CNN layer; `run(obj)`
    is the forward both must agree on."""
    y = run(layer)
    sd = layer.state_dict()
    fresh.load_state_dict(sd)
    _same_bytes(f"{tag} forward", y, f"{tag} forward after load_state_dict", run(fresh))
    names, named = _nn_endpoints_named(layer.named_parameters())
    flat = [np.ascontiguousarray(np.asarray(p)) for p in layer.parameters()]
    parts[tag] = _h(y, *_nn_endpoints_state(sd)[1])
    parts[tag + "_parameters"] = _h(*flat, *named)


@lane("nn-ep-cnn-layers")
def _(ml, X, yc, yr, Xh=None):
    """The CNN layers' state dicts, parameters, modes, setters and fits on 64 fixture rows: Conv1d and
    Conv2d (round trip into a layer of another seed; `set_weights` of the first's weights into a
    third, which must answer equally; `pin_weights` / `unpin_weights` around a forward, which must
    not move), BatchNorm1d / BatchNorm2d (`train()` forwards that move the running statistics, the
    round trip, `train(False)` forwards compared), BasicBlock (`fit` returns the block, `train()`,
    the round trip in eval mode), GCNConv and SAGEConv on the x-cnn graph (round trip). Parts per
    layer: the forward and the state, the parameters."""
    rows = np.ascontiguousarray(X[:64, :16], dtype=np.float32)
    parts = {}
    c1 = ml.Conv1d(2, 3, 3, padding=1, random_state=5, input_shape=(2, 8))
    s1 = rows.reshape(64, 2, 8)
    _nn_endpoints_layer_round_trip("conv1d", c1, ml.Conv1d(2, 3, 3, padding=1, random_state=6, input_shape=(2, 8)),
                                   lambda m: _nn_endpoints_host(m.forward(s1)), parts)
    c1b = ml.Conv1d(2, 3, 3, padding=1, random_state=7, input_shape=(2, 8))
    c1b.set_weights(np.asarray(c1.weight_).copy(), np.asarray(c1.bias_).copy())
    _same_bytes("conv1d forward", _nn_endpoints_host(c1.forward(s1)), "conv1d forward after set_weights",
                _nn_endpoints_host(c1b.forward(s1)))
    c1.pin_weights()
    pinned1 = _nn_endpoints_host(c1.forward(s1))
    c1.unpin_weights()
    _same_bytes("conv1d forward pinned", pinned1, "conv1d forward unpinned", _nn_endpoints_host(c1.forward(s1)))
    c2 = ml.Conv2d(1, 4, 3, padding=1, random_state=3, input_shape=(1, 4, 4))
    s2 = rows.reshape(64, 1, 4, 4)
    _nn_endpoints_layer_round_trip("conv2d", c2, ml.Conv2d(1, 4, 3, padding=1, random_state=4, input_shape=(1, 4, 4)),
                                   lambda m: _nn_endpoints_host(m.forward(s2)), parts)
    c2b = ml.Conv2d(1, 4, 3, padding=1, random_state=8, input_shape=(1, 4, 4))
    c2b.set_weights(np.asarray(c2.weight_).copy(), np.asarray(c2.bias_).copy())
    _same_bytes("conv2d forward", _nn_endpoints_host(c2.forward(s2)), "conv2d forward after set_weights",
                _nn_endpoints_host(c2b.forward(s2)))
    c2.pin_weights()
    pinned2 = _nn_endpoints_host(c2.forward(s2))
    c2.unpin_weights()
    _same_bytes("conv2d forward pinned", pinned2, "conv2d forward unpinned", _nn_endpoints_host(c2.forward(s2)))
    parts["pinned"] = _h(pinned1, pinned2)
    bn1 = ml.BatchNorm1d(16, momentum=0.25)
    bn1.train()
    bn1.forward(rows[:32])
    bn1.forward(rows[32:])
    bn1.train(False)
    bn1b = ml.BatchNorm1d(16, momentum=0.25)
    bn1b.train(False)
    _nn_endpoints_layer_round_trip("batchnorm1d", bn1, bn1b, lambda m: _nn_endpoints_host(m.forward(rows)), parts)
    parts["batchnorm1d_running"] = _h(np.asarray(bn1.running_mean_), np.asarray(bn1.running_var_))
    m2 = rows.reshape(64, 4, 2, 2)
    bn2 = ml.BatchNorm2d(4, momentum=0.3, input_shape=(4, 2, 2))
    bn2.train(True)
    bn2.forward(m2[:32])
    bn2.forward(m2[32:])
    bn2.train(False)
    bn2b = ml.BatchNorm2d(4, momentum=0.3, input_shape=(4, 2, 2))
    bn2b.train(False)
    _nn_endpoints_layer_round_trip("batchnorm2d", bn2, bn2b, lambda m: _nn_endpoints_host(m.forward(m2)), parts)
    parts["batchnorm2d_running"] = _h(np.asarray(bn2.running_mean_), np.asarray(bn2.running_var_))
    blk = ml.BasicBlock(1, 3, stride=2, random_state=21, input_shape=(1, 4, 4))
    if blk.fit(rows) is not blk:
        raise AssertionError("BasicBlock.fit must return its block")
    blk.train()
    trained = _nn_endpoints_host(blk.forward(s2))
    blk.train(False)
    other = ml.BasicBlock(1, 3, stride=2, random_state=22, input_shape=(1, 4, 4))
    other.train(False)
    _nn_endpoints_layer_round_trip("basicblock", blk, other, lambda m: _nn_endpoints_host(m.forward(s2)), parts)
    parts["basicblock_train"] = _h(trained)
    nodes = np.ascontiguousarray(X[:64, :16], dtype=np.float32)
    ei = _cnn_graph(64)
    _nn_endpoints_layer_round_trip("gcnconv", ml.GCNConv(16, 5, random_state=31), ml.GCNConv(16, 5, random_state=33),
                                   lambda m: _nn_endpoints_host(m.forward(nodes, ei)), parts)
    _nn_endpoints_layer_round_trip("sageconv", ml.SAGEConv(16, 4, random_state=41), ml.SAGEConv(16, 4, random_state=43),
                                   lambda m: _nn_endpoints_host(m.forward(nodes, ei)), parts)
    return _fit(parts)


@lane("nn-ep-cnn-fits")
def _(ml, X, yc, yr, Xh=None):
    """`fit` of the parameterless CNN layers (it returns the layer; the layer then transforms the
    same rows as an unfitted twin does): MaxPool1d, AvgPool1d, MaxPool2d, AvgPool2d,
    AdaptiveAvgPool2d, AdaptiveMaxPool2d, Dropout2d (eval mode). Parts: each fitted transform."""
    rows = np.ascontiguousarray(X[:64, :16], dtype=np.float32)
    parts = {}
    for tag, layer, twin in (
            ("maxpool1d", ml.MaxPool1d(3, stride=2, padding=1, input_shape=(2, 8)),
             ml.MaxPool1d(3, stride=2, padding=1, input_shape=(2, 8))),
            ("avgpool1d", ml.AvgPool1d(2, input_shape=(2, 8)), ml.AvgPool1d(2, input_shape=(2, 8))),
            ("maxpool2d", ml.MaxPool2d(2, stride=1, input_shape=(1, 4, 4)), ml.MaxPool2d(2, stride=1, input_shape=(1, 4, 4))),
            ("avgpool2d", ml.AvgPool2d(2, input_shape=(1, 4, 4)), ml.AvgPool2d(2, input_shape=(1, 4, 4))),
            ("adaptiveavg", ml.AdaptiveAvgPool2d(2, input_shape=(1, 4, 4)), ml.AdaptiveAvgPool2d(2, input_shape=(1, 4, 4))),
            ("adaptivemax", ml.AdaptiveMaxPool2d(2, input_shape=(1, 4, 4)), ml.AdaptiveMaxPool2d(2, input_shape=(1, 4, 4)))):
        if layer.fit(rows) is not layer:
            raise AssertionError(f"{tag}.fit must return its layer")
        t = np.asarray(layer.transform(rows))
        _same_bytes(f"{tag} transform after fit", t, f"{tag} transform unfitted", np.asarray(twin.transform(rows)))
        parts[tag] = _h(t)
    drop = ml.Dropout2d(p=0.3, random_state=11, input_shape=(4, 2, 2))
    drop.eval()
    if drop.fit(rows) is not drop:
        raise AssertionError("Dropout2d.fit must return its layer")
    parts["dropout2d"] = _h(np.asarray(drop.transform(rows)))
    return _fit(parts)


# ---------------------------------------------------------------- the recurrent estimators' state dicts

@lane("nn-ep-rnn-state")
def _(ml, X, yc, yr, Xh=None):
    """GRU, LSTM and RNN, classifier and regressor (hidden 6, one epoch on 96 sequences of 4 steps):
    `state_dict` of one fit into a twin fitted from another seed by `load_state_dict`; the twin's
    predictions must equal the first's. Parts per class: the state and the predictions."""
    Xs = _sequence_seq(X)
    ycs, yrs = _sequence_targets(yc, yr)
    parts = {}
    for cls, y, tag in ((ml.GRUClassifier, ycs, "gru_c"), (ml.LSTMClassifier, ycs, "lstm_c"),
                        (ml.RNNClassifier, ycs, "rnn_c"), (ml.GRURegressor, yrs, "gru_r"),
                        (ml.LSTMRegressor, yrs, "lstm_r"), (ml.RNNRegressor, yrs, "rnn_r")):
        a = cls(hidden_size=6, batch_size=32, max_epochs=1, random_state=13).fit(Xs, y)
        b = cls(hidden_size=6, batch_size=32, max_epochs=1, random_state=14).fit(Xs, y)
        sd = a.state_dict()
        b.load_state_dict(sd)
        pa = np.asarray(a.predict(Xs))
        _same_bytes(f"{tag} predict", pa, f"{tag} predict after load_state_dict", np.asarray(b.predict(Xs)))
        parts[tag] = _h(pa, *_nn_endpoints_state(sd)[1])
    return _fit(parts)


# ---------------------------------------------------------------- Sequential

def _nn_endpoints_dense(ml):
    """(8,) samples: add("linear", 16), sigmoid, layer_norm, dropout, residual(0), branch(1),
    linear(16), multiply(4), concat([4, 7]), softmax, gated(gate, up), linear(1)."""
    gate = ml.Sequential(numeric_mode="identical")
    gate.linear(16, bias=False)
    gate.silu()
    up = ml.Sequential(numeric_mode="identical")
    up.linear(16, bias=False)
    m = ml.Sequential(numeric_mode="identical", input_shape=(8,))
    m.add("linear", 16)
    m.sigmoid()
    m.layer_norm()
    m.dropout(0.1)
    m.residual(0)
    m.branch(1)
    m.linear(16)
    m.multiply(4)
    m.concat([4, 7])
    m.softmax()
    m.gated(gate, up)
    m.linear(1)
    return m


def _nn_endpoints_conv(ml):
    """(1, 4, 4) samples: conv2d(4, 3, padding 1), batch_norm2d(4), instance_norm2d(4, affine),
    max_pool2d(2, stride 1), avg_pool2d(2, stride 1), permute((1, 2, 0)), reshape((4, 4)), rnn(4),
    flatten, linear(1)."""
    m = ml.Sequential(numeric_mode="identical", input_shape=(1, 4, 4))
    m.conv2d(4, 3, padding=1)
    m.batch_norm2d(4)
    m.instance_norm2d(4, affine=True)
    m.max_pool2d(2, stride=1)
    m.avg_pool2d(2, stride=1)
    m.permute((1, 2, 0))
    m.reshape((4, 4))
    m.rnn(4)
    m.flatten()
    m.linear(1)
    return m


def _nn_endpoints_conv1d(ml):
    """(2, 8) samples: conv1d(3, 3, padding 1), avg_pool1d(2), flatten, linear(1)."""
    m = ml.Sequential(numeric_mode="identical", input_shape=(2, 8))
    m.conv1d(3, 3, padding=1)
    m.avg_pool1d(2)
    m.flatten()
    m.linear(1)
    return m


def _nn_endpoints_seq_fit(m, x, y, xh, tag, parts, loss="mse"):
    m.compile(loss=loss, optimizer="adam", lr=0.01)
    m.fit(x, y, epochs=2, batch_size=16, shuffle=True, seed=0)
    pred = np.asarray(m.predict(x))
    parts[tag] = _h(np.asarray(m.loss_curve_, dtype=np.float32), np.asarray(m.params_), pred,
                    np.asarray(m.predict(xh)))
    return pred


def _nn_endpoints_seq_round_trip(ml, m, fresh, x, pred, tag, parts):
    """state_dict into `fresh` (the same architecture, unfitted), parameters, named_parameters, to,
    save_checkpoint + Sequential.from_checkpoint: every reloaded model predicts as `m` does."""
    sd = m.state_dict()
    fresh.load_state_dict(sd)
    _same_bytes(f"{tag} predict", pred, f"{tag} predict after load_state_dict", np.asarray(fresh.predict(x)))
    names, named = _nn_endpoints_named(m.named_parameters())
    _nn_endpoints_check_order(tag, names, sd)
    flat = [np.ascontiguousarray(np.asarray(p)) for p in m.parameters()]
    m.to(_lane_side())
    _same_bytes(f"{tag} predict", pred, f"{tag} predict after to()", np.asarray(m.predict(x)))
    with tempfile.TemporaryDirectory(prefix="ib-nn-ep-seq-") as tmp:
        path = os.path.join(tmp, "model.ckpt")
        m.save_checkpoint(path)
        loaded = ml.Sequential.from_checkpoint(path, numeric_mode="identical")
        _same_bytes(f"{tag} predict", pred, f"{tag} predict after from_checkpoint", np.asarray(loaded.predict(x)))
    parts[tag + "_state"] = _h(*_nn_endpoints_state(sd)[1], *flat, *named)


@lane("nn-ep-sequential-dense")
def _(ml, X, yc, yr, Xh=None):
    """Sequential's dense builders (add, sigmoid, layer_norm, dropout, residual, branch, multiply,
    concat, softmax, gated) fitted two Adam epochs on 64 fixture rows; then the state dict round trip
    into the same architecture unfitted, parameters, named_parameters, to, save_checkpoint and
    Sequential.from_checkpoint (each reloaded model must predict equally). Parts: the loss curve,
    parameters and predictions; the state."""
    x = np.ascontiguousarray(X[:64, :8], dtype=np.float32)
    xh = np.ascontiguousarray((X if Xh is None else Xh)[:64, :8], dtype=np.float32)
    y = np.ascontiguousarray(yr[:64], dtype=np.float32).reshape(64, 1)
    parts = {}
    m = _nn_endpoints_dense(ml)
    pred = _nn_endpoints_seq_fit(m, x, y, xh, "dense", parts)
    _nn_endpoints_seq_round_trip(ml, m, _nn_endpoints_dense(ml), x, pred, "dense", parts)
    return _fit(parts)


@lane("nn-ep-sequential-conv")
def _(ml, X, yc, yr, Xh=None):
    """Sequential's image and sequence builders (batch_norm2d, instance_norm2d, max_pool2d,
    avg_pool2d, permute, rnn; and conv1d with avg_pool1d) fitted two Adam epochs on 64 fixture rows,
    with the state dict round trip and checkpoint of the first. Parts: per model the loss curve,
    parameters and predictions; the state."""
    x = np.ascontiguousarray(X[:64, :16], dtype=np.float32)
    xh = np.ascontiguousarray((X if Xh is None else Xh)[:64, :16], dtype=np.float32)
    y = np.ascontiguousarray(yr[:64], dtype=np.float32).reshape(64, 1)
    parts = {}
    m = _nn_endpoints_conv(ml)
    pred = _nn_endpoints_seq_fit(m, x, y, xh, "conv2d", parts)
    _nn_endpoints_seq_round_trip(ml, m, _nn_endpoints_conv(ml), x, pred, "conv2d", parts)
    _nn_endpoints_seq_fit(_nn_endpoints_conv1d(ml), x, y, xh, "conv1d", parts)
    return _fit(parts)


@lane("nn-ep-sequential-lm")
def _(ml, X, yc, yr, Xh=None):
    """Sequential's language-model builders: (8,) id samples (ids as floats, vocabulary 32):
    embedding(32, 16), rotary(8), transformer_encoder(an encoder layer, 1), lm_head(32), trained two
    epochs with token_cross_entropy; and (8, 16) samples: linear(16), transformer_decoder(a decoder
    layer, 1, memory=-1), flatten, linear(1), trained with mse. Parts: per model the loss curve,
    parameters and predictions."""
    ids = (_ids(X, 64, 9) % 32).astype(np.int32)
    x = np.ascontiguousarray(ids[:, :8], dtype=np.float32)
    held = (_ids(X if Xh is None else Xh, 64, 9) % 32).astype(np.int32)
    xh = np.ascontiguousarray(held[:, :8], dtype=np.float32)
    targets = np.ascontiguousarray(ids[:, 1:], dtype=np.int32)
    parts = {}
    enc = ml.Sequential(numeric_mode="identical")
    enc.transformer_encoder_layer(16, 2, dim_feedforward=32, dropout=0.0)
    lm = ml.Sequential(numeric_mode="identical", input_shape=(8,))
    lm.embedding(32, 16)
    lm.rotary(8)
    lm.transformer_encoder(enc, 1)
    lm.lm_head(32)
    lm.compile(loss="token_cross_entropy", optimizer="adamw", lr=0.01)
    lm.fit(x, targets, epochs=2, batch_size=16, shuffle=True, seed=0)
    parts["lm"] = _h(np.asarray(lm.loss_curve_, dtype=np.float32), np.asarray(lm.params_),
                     np.asarray(lm.predict(x)), np.asarray(lm.predict(xh)))
    dec = ml.Sequential(numeric_mode="identical")
    dec.transformer_decoder_layer(16, 2, 32, dropout=0.0, tgt_is_causal=True)
    m = ml.Sequential(numeric_mode="identical", input_shape=(8, 16))
    m.linear(16)
    m.transformer_decoder(dec, 1, memory=-1)
    m.flatten()
    m.linear(1)
    xs = _seq(X, 64, 8, 16).reshape(64, 128)
    xsh = _seq(X if Xh is None else Xh, 64, 8, 16).reshape(64, 128)
    y = np.ascontiguousarray(yr[:64], dtype=np.float32).reshape(64, 1)
    _nn_endpoints_seq_fit(m, xs, y, xsh, "decoder", parts)
    return _fit(parts)


# ---------------------------------------------------------------- Samba

def _nn_endpoints_samba_config(ml):
    return ml.SambaConfig(vocab=64, d_model=32, layers=("mamba3", "attention"), n_heads=2, intermediate=64)


@lane("nn-ep-samba-stack")
def _(ml, X, yc, yr, Xh=None):
    """SambaConfig's `to_dict` / `from_dict` (the rebuilt config's `registry()` must equal), its
    `block_shapes` of both kinds and `registry`; SambaStack (generator seed 1, vocabulary 64):
    `parameters`, and `load_weights` of those weights into a stack of another seed (its forward must
    equal). Parts: the registry and block shapes' numbers, the parameters, the forward."""
    cfg = _nn_endpoints_samba_config(ml)
    again = ml.SambaConfig.from_dict(cfg.to_dict())
    reg = cfg.registry()
    if [tuple(map(str, (n,))) + tuple(s) for n, s in reg] != [tuple(map(str, (n,))) + tuple(s) for n, s in again.registry()]:
        raise AssertionError("SambaConfig.from_dict(to_dict()).registry() differs from the original's")
    shapes = [int(v) for kind in ("mamba3", "attention") for _, s in cfg.block_shapes(kind) for v in (len(s),) + tuple(s)]
    m = ml.SambaStack(cfg, generator=ml.training.Generator(1), lr=1e-3)
    names = [str(n) for n, _ in reg]
    flat = [np.ascontiguousarray(np.asarray(p)) for p in m.parameters()]
    ids = _ids(X, 2, 16, vocab=64)
    y = _nn_endpoints_host(m.forward(ids))
    other = ml.SambaStack(cfg, generator=ml.training.Generator(2), lr=1e-3)
    other.load_weights({n: p.copy() for n, p in zip(names, flat)})
    _same_bytes("SambaStack forward", y, "SambaStack forward after load_weights", _nn_endpoints_host(other.forward(ids)))
    reg_numbers = [int(v) for _, s in reg for v in (len(s),) + tuple(s)]
    return _fit(dict(registry=_h(np.asarray(reg_numbers, dtype=np.int64)),
                     block_shapes=_h(np.asarray(shapes, dtype=np.int64)), parameters=_h(*flat), forward=_h(y)))


@lane("nn-ep-samba-inference")
def _(ml, X, yc, yr, Xh=None):
    """SambaInference (the CPU inference class) built from a SambaStack's config and weights:
    `__call__` and `logits` on (2, 16) fixture ids (the same answer: `logits` is `forward`),
    `parameters` (must equal the weights given). Parts: the logits and the parameters."""
    cfg = _nn_endpoints_samba_config(ml)
    stack = ml.SambaStack(cfg, generator=ml.training.Generator(1), lr=1e-3)
    names = [str(n) for n, _ in cfg.registry()]
    weights = {n: np.ascontiguousarray(np.asarray(p)).copy() for n, p in zip(names, stack.parameters())}
    inf = ml.SambaInference(cfg, weights)
    ids = _ids(X, 2, 16, vocab=64)
    y = _nn_endpoints_host(inf.__call__(ids))
    _same_bytes("SambaInference __call__", y, "SambaInference logits", _nn_endpoints_host(inf.logits(ids)))
    flat = [np.ascontiguousarray(np.asarray(p)) for p in inf.parameters()]
    for n, p in zip(names, flat):
        _same_bytes(f"parameters[{n}]", p, f"weights[{n}]", weights[n])
    return _fit(dict(logits=_h(y), parameters=_h(*flat)))


_one_sided("nn-ep-samba-inference", missing="gpu",
           reason="SambaInference is the forward-only CPU inference class over the host neural binding; "
                  "its GPU counterpart is SambaStack (nn-ep-samba-stack)")


_batch_decl("n/a:endpoint-round-trips (state dict, parameter, mode and checkpoint round trips of neural layers "
            "and models, held equal in the lane; no separate prediction batch axis)",
            "nn-ep-mamba-blocks", "nn-ep-mamba3-owned", "nn-ep-transformer-block", "nn-ep-layernorm", "nn-ep-moe",
            "nn-ep-embedding", "nn-ep-cnn-layers", "nn-ep-cnn-fits", "nn-ep-rnn-state", "nn-ep-sequential-dense",
            "nn-ep-sequential-conv", "nn-ep-sequential-lm", "nn-ep-samba-stack", "nn-ep-samba-inference")
