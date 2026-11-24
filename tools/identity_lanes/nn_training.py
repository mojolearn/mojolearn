# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE TRAINING ENDPOINTS' IDENTITY LANES (lane vdv-cov-nn, 2026-11-24).
#
# Verify through `device=` (docs/plans/VERIFY_THROUGH_DEVICE.md, item 3): a lane for every public
# training endpoint with a numerical result that no lane called: the bare optimizers' state round
# trips, zero_grad, step_accumulated and resident IO; the sequence optimizers' state round trips; the
# learning-rate schedules' lr_at / bits_at / config / from_config; the seeded Generator's draws; the
# SmallMLPTrainer's batched and split steps; the byte LM trainer's steps, evaluation, gradients and
# checkpoints. Every lane is device-agnostic (contract 1): it calls the public API with no `device=`
# and the harness's view gives every constructor the side. A round trip is held byte for byte in the
# lane (`_same_bytes`), so a broken load reads REFUSED naming the pair, never a quiet new hash.
#
# Loaded by tools/identity_break.py `_load_lane_fragments` in the harness namespace: np, _h, _fit,
# _same_bytes, _optim_tensors, _hw, _ids, _byte_lm_params, LM_CONFIG_SHAPES, _one_sided and `lane`
# are the harness's; helpers here are prefixed `_nn_training_`. Each lane builds its own small inputs
# and objects and shares nothing mutable with another lane (the verify workers run lanes in
# parallel). Every reference is OWED (recorded once, at the release's one verification).


def _nn_training_arrays(obj):
    """The numeric leaves of a state dict (nested dicts, lists and tuples), in sorted-key order, as
    arrays for `_h`. Text (names, profiles, paths, vendor words) and None are not numbers and are
    left out; a bool or an int becomes an int64, a float a float64."""
    out = []
    if isinstance(obj, dict):
        for k in sorted(obj, key=str):
            out.extend(_nn_training_arrays(obj[k]))
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            out.extend(_nn_training_arrays(v))
    elif isinstance(obj, (bool, int, np.integer, np.bool_)):
        out.append(np.asarray([int(obj)], dtype=np.int64))
    elif isinstance(obj, (float, np.floating)):
        out.append(np.asarray([float(obj)], dtype=np.float64))
    elif obj is None or isinstance(obj, (str, bytes)):
        pass
    else:
        a = np.asarray(obj)
        if a.dtype != object and a.dtype.kind not in "OUSV":
            out.append(np.ascontiguousarray(a))
    return out


def _nn_training_digest(obj):
    arrays = _nn_training_arrays(obj)
    return _h(*arrays) if arrays else _h(np.zeros(0, dtype=np.int64))


def _nn_training_copies(tensors):
    return [np.ascontiguousarray(t, dtype=np.float32).copy() for t in tensors]


# ---------------------------------------------------------------- the bare optimizers (SGD, Adam, AdamW)

@lane("nn-tr-optim-state")
def _(ml, X, yc, yr, Xh=None):
    """SGD (momentum), Adam and AdamW: one step, `state_dict`, a FRESH optimizer over copies of the
    stepped parameters `load_state_dict`s it, both take the same second step and their parameters
    must agree byte for byte (the resume claim); `zero_grad` over a gradient list with NaN, +-inf
    and -0.0 cells (all +0.0 bytes after); and `step_accumulated` (A = 2, T = 256, the split clause
    9.2 admits) on a third optimizer of each kind. Parts per class: the saved state, the resumed
    parameters, the zeroed buffers, the accumulated step's norm and parameters."""
    parts = {}
    for cls, kw, tag in ((ml.SGD, dict(momentum=0.5, dampening=0.25), "sgd"), (ml.Adam, {}, "adam"),
                         (ml.AdamW, dict(weight_decay=0.01), "adamw")):
        params, grads = _optim_tensors(f"nn-tr-state-{tag}")
        owner = cls(params, lr=3e-3, numeric_mode="identical", **kw)
        owner.step(_nn_training_copies(grads[0]))
        saved = owner.state_dict()
        parts[tag + "_state"] = _nn_training_digest(saved)
        resumed_params = _nn_training_copies(params)
        resumed = cls(resumed_params, lr=3e-3, numeric_mode="identical", **kw)
        resumed.load_state_dict(saved)
        owner.step(_nn_training_copies(grads[1]))
        resumed.step(_nn_training_copies(grads[1]))
        for k, (a, b) in enumerate(zip(params, resumed_params)):
            _same_bytes(f"{tag} continued p{k}", a, f"{tag} resumed after load_state_dict p{k}", b)
        parts[tag + "_resumed"] = _h(*resumed_params)
        parts[tag + "_reloaded_state"] = _nn_training_digest(resumed.state_dict())
        zeroed = [np.asarray([np.nan, np.inf, -np.inf, -0.0, 1.0, -2.0], dtype=np.float32),
                  _nn_training_copies(grads[0])[0]]
        owner.zero_grad(zeroed)
        parts[tag + "_zero"] = _h(*zeroed)
        acc_params, acc_grads = _optim_tensors(f"nn-tr-acc-{tag}")
        acc = cls(acc_params, lr=1e-3, numeric_mode="identical", accumulation_steps=2, **kw)
        norm = acc.step_accumulated([_nn_training_copies(g) for g in acc_grads], 256)
        parts[tag + "_accumulated"] = _h(np.asarray([np.nan if norm is None else float(norm)], dtype=np.float64),
                                         *acc_params)
    return _fit(parts)


@lane("nn-tr-optim-resident")
def _(ml, X, yc, yr, Xh=None):
    """The resident parameter and gradient IO of SGD, Adam and AdamW: `params_resident_available`,
    `params_to_device` (the host arrays then overwritten so a missing download cannot pass),
    `params_to_host`, `grads_to_device` / `grads_to_host` (into -19-filled buffers), `param_device`
    copies read back, one `step_device` on the uploaded gradient, the parameters downloaded after
    it. The GPU side only: the CPU training binding keeps no resident buffers (declared below)."""
    parts = {}
    for cls, tag in ((ml.SGD, "sgd"), (ml.Adam, "adam"), (ml.AdamW, "adamw")):
        params, grads = _optim_tensors(f"nn-tr-resident-{tag}")
        owner = cls(params, lr=3e-3, numeric_mode="identical", resident=True)
        if not owner.params_resident_available():
            raise RuntimeError(f"nn-tr-optim-resident: {tag} reports no resident parameter buffers on the GPU side")
        owner.params_to_device()
        for p in params:
            p.fill(-17.0)
        owner.params_to_host()
        parts[tag + "_params"] = _h(*params)
        owner.grads_to_device(_nn_training_copies(grads[0]))
        downloaded = [np.full_like(np.asarray(g, dtype=np.float32), -19.0) for g in grads[0]]
        owner.grads_to_host(downloaded)
        parts[tag + "_grads"] = _h(*downloaded)
        outputs = [owner.param_device(j) for j in range(len(params))]
        parts[tag + "_param_device"] = _h(*[np.asarray(o.to_host()) for o in outputs])
        owner.step_device()
        owner.params_to_host()
        parts[tag + "_stepped"] = _h(*params)
    return _fit(parts)


_one_sided("nn-tr-optim-resident", missing="cpu",
           reason="the CPU training binding keeps no resident parameter or gradient buffers "
                  "(params_resident_available is False there by design; optim-io-capability holds that refusal)")


# ---------------------------------------------------------------- the sequence optimizers

@lane("nn-tr-seqopt-state")
def _(ml, X, yc, yr, Xh=None):
    """Adagrad, Adamax, LAMB and Lion: three steps on two fixture-cut tensors, `state_dict`, a fresh
    optimizer over copies of the stepped tensors `load_state_dict`s it, both take the same fourth
    step and must agree byte for byte. Parts per class: the saved state's numbers and the resumed
    tensors."""
    parts = {}
    for cls, tag in ((ml.Adagrad, "adagrad"), (ml.Adamax, "adamax"), (ml.LAMB, "lamb"), (ml.Lion, "lion")):
        p1 = np.ascontiguousarray(X[:32, :8], dtype=np.float32).copy()
        p2 = np.ascontiguousarray(X[32:40, 0], dtype=np.float32).copy()
        owner = cls([p1, p2], lr=1e-2)
        grads = []
        for k in range(4):
            base = 40 + 48 * k
            grads.append([np.ascontiguousarray(X[base:base + 32, 8:16], dtype=np.float32) * np.float32(0.5),
                          np.ascontiguousarray(X[base + 32:base + 40, 1], dtype=np.float32)])
        for g in grads[:3]:
            owner.step(_nn_training_copies(g))
        saved = owner.state_dict()
        parts[tag + "_state"] = _nn_training_digest(saved)
        q1, q2 = p1.copy(), p2.copy()
        resumed = cls([q1, q2], lr=1e-2)
        resumed.load_state_dict(saved)
        owner.step(_nn_training_copies(grads[3]))
        resumed.step(_nn_training_copies(grads[3]))
        _same_bytes(f"{tag} continued", np.concatenate([p1.ravel(), p2]),
                    f"{tag} resumed after load_state_dict", np.concatenate([q1.ravel(), q2]))
        parts[tag + "_resumed"] = _h(q1, q2)
    return _fit(parts)


# ---------------------------------------------------------------- the learning-rate schedules

@lane("nn-tr-schedules")
def _(ml, X, yc, yr, Xh=None):
    """`lr_at` and `bits_at` at steps 1..24 of StepLR, ExponentialLR, OneCycleLR (cosine and linear
    three-phase) and of training.ConstantLR, WarmupLinearLR and WarmupCosineLR (warmup and decay
    edges inside the range); the three training schedules' `config()` turned back by `from_config`,
    whose rates must equal the original's bit for bit. Parts: the rates as float32, the bits, the
    configs' numbers."""
    steps = range(1, 25)
    seq = [ml.StepLR(0.1, step_size=5, gamma=0.5), ml.ExponentialLR(0.05, gamma=0.9),
           ml.OneCycleLR(0.2, total_steps=24),
           ml.OneCycleLR(0.2, total_steps=24, anneal_strategy="linear", three_phase=True, pct_start=0.25)]
    rates = [np.asarray([s.lr_at(t) for t in steps], dtype=np.float32) for s in seq]
    bits = [np.asarray([s.bits_at(t) for t in steps], dtype=np.uint32) for s in seq]
    constant = ml.training.ConstantLR(3e-3, warmup_steps=4)
    linear = ml.training.WarmupLinearLR(3e-3, warmup_steps=4, total_steps=18, min_lr=1e-4)
    cosine = ml.training.WarmupCosineLR(3e-3, warmup_steps=4, total_steps=18, min_lr=1e-4)
    configs = []
    for sched, again, tag in ((constant, ml.training.ConstantLR.from_config(constant.config()), "constant"),
                              (linear, ml.training.WarmupLinearLR.from_config(linear.config()), "linear"),
                              (cosine, ml.training.WarmupCosineLR.from_config(cosine.config()), "cosine")):
        r = np.asarray([sched.lr_at(t) for t in steps], dtype=np.float32)
        b = np.asarray([sched.bits_at(t) for t in steps], dtype=np.uint32)
        _same_bytes(f"{tag} bits_at", b, f"{tag} from_config(config()) bits_at",
                    np.asarray([again.bits_at(t) for t in steps], dtype=np.uint32))
        rates.append(r)
        bits.append(b)
        configs.extend(_nn_training_arrays(sched.config()))
    return _fit(dict(rates=_h(*rates), bits=_h(*bits), configs=_h(*configs)))


# ---------------------------------------------------------------- the seeded Generator

@lane("nn-tr-generator")
def _(ml, X, yc, yr, Xh=None):
    """training.Generator(seed from the fixture's first byte): `next_stream`, `uniform`, `normal`,
    the four initializers, `dropout` and `dropout_backward` (the mask replayed from the key); then
    `state_dict`, a fresh generator `load_state_dict`s it and its next draws must equal the first
    generator's. Parts: the stream ids, every draw, the dropout output and its backward."""
    seed = int(np.frombuffer(np.ascontiguousarray(X).tobytes()[:1], dtype=np.uint8)[0])
    gen = ml.training.Generator(seed)
    streams = [gen.next_stream(), gen.next_stream()]
    draws = [np.asarray(gen.uniform((16, 8), -0.5, 0.5)), np.asarray(gen.normal((16, 8), 0.0, 0.5)),
             np.asarray(gen.kaiming_uniform((16, 8), 8)), np.asarray(gen.kaiming_normal((16, 8), 8)),
             np.asarray(gen.xavier_uniform((16, 8), 8, 16)), np.asarray(gen.xavier_normal((16, 8), 8, 16))]
    x = np.ascontiguousarray(X[:16, :8], dtype=np.float32)
    y, key = gen.dropout(x, 0.25)
    dx = np.asarray(gen.dropout_backward(np.ascontiguousarray(X[16:32, :8], dtype=np.float32), key))
    saved = gen.state_dict()
    again = ml.training.Generator(seed + 1)
    again.load_state_dict(saved)
    a, b = np.asarray(gen.uniform((8, 8))), np.asarray(again.uniform((8, 8)))
    _same_bytes("uniform after the state", a, "uniform after load_state_dict", b)
    return _fit(dict(streams=_h(np.asarray(streams, dtype=np.int64)), draws=_h(*draws),
                     dropout=_h(np.asarray(y), dx), resumed=_h(b),
                     state=_nn_training_digest(saved)))


# ---------------------------------------------------------------- SmallMLPTrainer

def _nn_training_mlp_weights():
    return [((np.arange(int(np.prod(s)), dtype=np.float32) % 7 - 3) / 32).reshape(s).astype(np.float32)
            for s in ((16, 8), (16,), (3, 16), (3,))]


@lane("nn-tr-small-mlp")
def _(ml, X, yc, yr, Xh=None):
    """SmallMLPTrainer (the fixed 8-16-3 network, the `mlp` lane's starting weights): `train_steps`
    over two 64-row minibatches in one call; `loss_and_grads` then `apply_gradients` (the split
    step); `predict_logits_sessions` over two batches; `state_dict`, a fresh trainer from other
    weights `load_state_dict`s it and its logits and its next `train_step` must equal the first
    trainer's. Parts: the batched losses and last gradients, the split loss and gradients, the
    sessions' logits, the state's numbers, the resumed weights."""
    w = _nn_training_mlp_weights()
    m = ml.SmallMLPTrainer(*w, data_schedule={"dataset": "nn-tr-small-mlp", "order": "sequential"})
    Xm = np.ascontiguousarray(X[:256, :8], dtype=np.float32)
    t = _three_class(X[:256])
    out = m.train_steps(Xm[:128], t[:128], 2)
    batched = [np.asarray(out["losses"], dtype=np.float32)] + [np.asarray(g) for g in out["gradients"]]
    loss, grads = m.loss_and_grads(Xm[128:192], t[128:192])
    split_grads = [np.asarray(g) for g in grads]
    step = m.apply_gradients(grads)
    sessions = np.asarray(m.predict_logits_sessions([Xm[:32], Xm[32:96]]))
    saved = m.state_dict()
    other = [a + np.float32(0.125) for a in w]
    fresh = ml.SmallMLPTrainer(*other, data_schedule={"dataset": "nn-tr-small-mlp", "order": "sequential"})
    fresh.load_state_dict(saved)
    _same_bytes("logits", np.asarray(m.predict_logits(Xm)), "logits after load_state_dict",
                np.asarray(fresh.predict_logits(Xm)))
    a = m.train_step(Xm[192:256], t[192:256])
    b = fresh.train_step(Xm[192:256], t[192:256])
    _same_bytes("next step loss", np.float32(a["loss"]), "next step loss after load_state_dict", np.float32(b["loss"]))
    return _fit(dict(batched=_h(*batched), split=_h(np.float64(loss), np.int64(step), *split_grads),
                     sessions=_h(sessions), state=_nn_training_digest(saved),
                     resumed=_h(*[np.asarray(fresh.weights_[k]) for k in sorted(fresh.weights_)])))


# ---------------------------------------------------------------- the byte language-model trainer

@lane("nn-tr-byte-lm")
def _(ml, X, yc, yr, Xh=None):
    """LanguageModelTrainer at the `language-model-config` lane's first shape (one block, length 5,
    d_model 16): `train_steps` over two windows of the fixture bytes in one call, `evaluate`,
    `export_gradients`, `parameter_registry`, `run_metadata` and `attention_stage_report`
    (called and type-checked: build constants and a device footprint, not digested); `state_dict` into a fresh trainer by `load_state_dict`; the
    three checkpoint round trips (`save_checkpoint` / `from_checkpoint`, `export_checkpoint_binary` /
    `from_checkpoint_binary`, `export_checkpoint` bytes / `from_checkpoint_bytes`): every restored
    trainer's `evaluate` must equal the original's. LanguageModelInference.parameters_sha256 of the
    trained parameters. Parts: the losses, the evaluation, the gradients, the registry, the state,
    the restored evaluation, the parameters' sha256."""
    cfg = ml.LanguageModelConfig(**LM_CONFIG_SHAPES[0])
    _, flat = _byte_lm_params(cfg)
    m = ml.LanguageModelTrainer(flat, shape=cfg, data_schedule={"lane": "nn-tr-byte-lm"})
    width = cfg.length + 1
    ids = np.stack([_ids(X, cfg.batch, width, vocab=cfg.vocab_size),
                    _ids(X[64:], cfg.batch, width, vocab=cfg.vocab_size)])
    held = _ids(X if Xh is None else Xh, cfg.batch, width, vocab=cfg.vocab_size)
    steps = m.train_steps(ids)
    grads = m.export_gradients(named=False)
    evaluated = np.float64(m.evaluate(held))
    registry = ml.LanguageModelTrainer.parameter_registry(cfg)
    metadata = m.run_metadata()
    if not isinstance(metadata, dict):
        raise AssertionError(f"run_metadata returned {type(metadata).__name__}")
    report = m.attention_stage_report()
    if report is not None and not isinstance(report, dict):
        raise AssertionError(f"attention_stage_report returned {type(report).__name__}")
    saved = m.state_dict()
    fresh = ml.LanguageModelTrainer(_byte_lm_params(cfg)[1], shape=cfg, data_schedule={"lane": "nn-tr-byte-lm"})
    fresh.load_state_dict(saved)
    _same_bytes("evaluate", evaluated, "evaluate after load_state_dict", np.float64(fresh.evaluate(held)))
    with tempfile.TemporaryDirectory(prefix="ib-nn-tr-byte-lm-") as tmp:
        json_path, bin_path, exp_path = (os.path.join(tmp, n) for n in ("a.json", "b.bin", "c.json"))
        m.save_checkpoint(json_path)
        m.export_checkpoint_binary(bin_path)
        m.export_checkpoint(exp_path)
        with open(exp_path, "rb") as fh:
            encoded = fh.read()
        restored = [ml.LanguageModelTrainer.from_checkpoint(json_path),
                    ml.LanguageModelTrainer.from_checkpoint_binary(bin_path),
                    ml.LanguageModelTrainer.from_checkpoint_bytes(encoded)]
        for name, r in zip(("from_checkpoint", "from_checkpoint_binary", "from_checkpoint_bytes"), restored):
            _same_bytes("evaluate", evaluated, f"evaluate after {name}", np.float64(r.evaluate(held)))
    inference = ml.LanguageModelInference(np.asarray(m.parameters_, dtype=np.float32), shape=cfg, threaded=False)
    sha = bytes.fromhex(str(inference.parameters_sha256()))
    return _fit(dict(losses=_h(np.asarray(steps["losses"], dtype=np.float32)), evaluate=_h(evaluated),
                     gradients=_nn_training_digest(grads), registry=_nn_training_digest(registry),
                     state=_nn_training_digest(saved),
                     sha256=_h(np.frombuffer(sha, dtype=np.uint8))))


_batch_decl("n/a:training-endpoints (optimizer, schedule, generator and trainer state round trips and steps; "
            "no prediction batch axis)",
            "nn-tr-optim-state", "nn-tr-optim-resident", "nn-tr-seqopt-state", "nn-tr-schedules",
            "nn-tr-generator", "nn-tr-small-mlp", "nn-tr-byte-lm")
