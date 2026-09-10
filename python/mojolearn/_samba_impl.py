# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""A Samba-shaped stack on the GPU: embedding -> [Mamba3Block |
TransformerBlock]* (each block carries its own pre-norm and residual) ->
final RMSNorm -> tied or untied LM head -> cross-entropy, with a full
backward, AdamW with clipping, a learning-rate schedule, clause 9.2 gradient
accumulation, a position-keyed RNG and a JSON+hex+sha256 checkpoint.

PRIVATE MODULE, AND IT HOLDS NO NUMERICS. Every arithmetic step is a call
into `_mojolearn_training` (embedding, RMSNorm, head GEMM, accumulate, RNG,
loss, optimizer), `_mojolearn_mamba` (Mamba3Block forward/backward) or
`_mojolearn_transformer` (TransformerBlock forward and IDENTICAL zero-state
prefill backward). Attention layers use full causal attention, and each
backward recomputes its saved layer input from a fresh cache. This file
decides the ORDER of the tensor registry (part of the clipped answer), the
order of the blocks, and what goes in a checkpoint.

The tensor registry, in order, is the clip's cross-tensor summation order:

    embed.weight
    layers.{i}.<block tensor names, in the block class's _W_NAMES order>
    norm_f.weight
    lm_head.weight            only when tie_embeddings is False
"""

import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from . import _backend
from . import _training_impl as T
from ._mamba_impl import Mamba3Block
from ._transformer_impl import TransformerBlock

__all__ = ["SambaConfig", "SambaStack"]

PROFILE = "mojolearn.samba-stack.fp32.v1"
_STATE_SCHEMA = "mojolearn.samba-stack-state.v1"
_CHECKPOINT_SCHEMA = "mojolearn.samba-stack-json-checkpoint.v1"
_CHECKPOINT_LIMIT = 256 * 1024 * 1024

_M3_HEADDIM = 64
_M3_EXPAND = 2
_M3_D_STATE = 128
_M3_NGROUPS = 1
_M3_NUM_ROPE_ANGLES = 32

_LAYER_KINDS = ("mamba3", "attention")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("ascii")


class SambaConfig(object):
    """The stack's shape. `layers` is a sequence of "mamba3" / "attention"
    in stack order. `d_model` must be a multiple of 32 for a Mamba-3 layer
    (headdim 64, expand 2) and equal `n_heads * head_dim` for an attention
    layer. `norm_eps` is the FINAL norm's eps; the blocks carry their own
    profile constants."""

    def __init__(self, vocab, d_model, layers, n_heads=None, n_kv_heads=None,
                 head_dim=None, intermediate=None, tie_embeddings=True,
                 norm_eps=1e-5, dropout=0.0):
        self.vocab = int(vocab)
        self.d_model = int(d_model)
        self.layers = tuple(str(k) for k in layers)
        self.n_heads = None if n_heads is None else int(n_heads)
        self.n_kv_heads = (None if n_kv_heads is None else int(n_kv_heads))
        self.head_dim = None if head_dim is None else int(head_dim)
        self.intermediate = None if intermediate is None else int(intermediate)
        self.tie_embeddings = bool(tie_embeddings)
        self.norm_eps = float(np.float32(norm_eps))
        self.dropout = float(np.float32(dropout))
        if self.vocab < 2 or self.d_model < 1 or not self.layers:
            raise ValueError("mojolearn.SambaConfig: vocab >= 2, d_model >= 1 "
                             "and at least one layer are required")
        for k in self.layers:
            if k not in _LAYER_KINDS:
                raise ValueError("mojolearn.SambaConfig: layer kind %r; the "
                                 "kinds are %r" % (k, _LAYER_KINDS))
        if "mamba3" in self.layers and self.d_model % (_M3_HEADDIM // _M3_EXPAND):
            raise ValueError("mojolearn.SambaConfig: a mamba3 layer needs "
                             "d_model to be a multiple of 32 (Mamba3Block)")
        if "attention" in self.layers:
            if self.n_heads is None or self.intermediate is None:
                raise ValueError("mojolearn.SambaConfig: an attention layer "
                                 "needs n_heads and intermediate")
            if self.head_dim is None:
                self.head_dim = self.d_model // self.n_heads
            if self.n_kv_heads is None:
                self.n_kv_heads = self.n_heads
            if self.n_heads * self.head_dim != self.d_model:
                raise ValueError("mojolearn.SambaConfig: d_model must equal "
                                 "n_heads * head_dim")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("mojolearn.SambaConfig: dropout must be in [0, 1)")
        if not (self.norm_eps >= 0.0):
            raise ValueError("mojolearn.SambaConfig: norm_eps must be >= 0")

    def to_dict(self):
        return {"vocab": self.vocab, "d_model": self.d_model,
                "layers": list(self.layers), "n_heads": self.n_heads,
                "n_kv_heads": self.n_kv_heads, "head_dim": self.head_dim,
                "intermediate": self.intermediate,
                "tie_embeddings": self.tie_embeddings,
                "norm_eps": self.norm_eps, "dropout": self.dropout}

    @classmethod
    def from_dict(cls, d):
        return cls(d["vocab"], d["d_model"], d["layers"], d["n_heads"],
                   d["n_kv_heads"], d["head_dim"], d["intermediate"],
                   d["tie_embeddings"], d["norm_eps"], d["dropout"])

    # -- the registry ---------------------------------------------------
    def block_shapes(self, kind):
        """`[(name, shape)]` of one block's tensors, in the block class's
        `_W_NAMES` order."""
        dm = self.d_model
        if kind == "mamba3":
            di = _M3_EXPAND * dm
            nh = di // _M3_HEADDIM
            dip = 2 * di + 2 * _M3_NGROUPS * _M3_D_STATE + 3 * nh + _M3_NUM_ROPE_ANGLES
            shapes = {
                "block_norm.weight": (dm,), "in_proj.weight": (dip, dm),
                "dt_bias": (nh,), "B_norm.weight": (_M3_D_STATE,),
                "C_norm.weight": (_M3_D_STATE,), "B_bias": (nh, _M3_D_STATE),
                "C_bias": (nh, _M3_D_STATE), "D": (nh,),
                "out_proj.weight": (dm, di),
            }
            return [(n, shapes[n]) for n in Mamba3Block._W_NAMES]
        qw = self.n_heads * self.head_dim
        kw = self.n_kv_heads * self.head_dim
        it = self.intermediate
        shapes = {
            "input_layernorm.weight": (dm,),
            "post_attention_layernorm.weight": (dm,),
            "q_proj.weight": (qw, dm), "k_proj.weight": (kw, dm),
            "v_proj.weight": (kw, dm), "o_proj.weight": (dm, qw),
            "gate_proj.weight": (it, dm), "up_proj.weight": (it, dm),
            "down_proj.weight": (dm, it),
        }
        return [(n, shapes[n]) for n in TransformerBlock._W_NAMES]

    def registry(self):
        """`[(name, shape)]` in the clip's order."""
        out = [("embed.weight", (self.vocab, self.d_model))]
        for i, kind in enumerate(self.layers):
            out.extend(("layers.%d.%s" % (i, n), s)
                       for n, s in self.block_shapes(kind))
        out.append(("norm_f.weight", (self.d_model,)))
        if not self.tie_embeddings:
            out.append(("lm_head.weight", (self.vocab, self.d_model)))
        return out


def _init_tensor(gen, name, shape):
    """Initial bits from the position-keyed stream: normal(0, 0.02) for the
    embedding and an untied head, torch's Linear default
    (kaiming-uniform, bound 1/sqrt(fan_in)) for every projection, ones for
    the norms / D / B_bias / C_bias, and uniform(-4, -2) for dt_bias
    (softplus of that is a dt near 0.02 to 0.13)."""
    last = name.split(".")[-1]
    if name in ("embed.weight", "lm_head.weight"):
        return gen.normal(shape, 0.0, 0.02)
    if last == "weight" and len(shape) == 2:
        return gen.kaiming_uniform(shape, fan_in=shape[1])
    if last == "dt_bias":
        return gen.uniform(shape, -4.0, -2.0)
    return np.ones(shape, dtype=np.float32)


class SambaStack(object):
    """The stack, its optimizer, schedule and RNG, over one flat float32
    parameter buffer whose registry views the blocks read directly.

        cfg = SambaConfig(vocab=256, d_model=64, layers=("mamba3", "mamba3"))
        stack = SambaStack(cfg, generator=Generator(7), lr=1e-3,
                           lr_schedule=WarmupCosineLR(1e-3, 8, 64),
                           max_norm=1.0, accumulation_steps=2)
        out = stack.train_step(inputs, targets)     # (B, L) int32 each
        stack.save_checkpoint("run.json")

    `weights` (a dict keyed by the registry names) or `generator` (a
    `Generator`) supplies the initial bits; exactly one of them. The
    optimizer is AdamW over the registry in order; `max_norm=None` turns
    the clip off. `accumulation_steps` splits the batch rows into that many
    microbatches and combines by the clause 9.2 tree; a split the clause
    does not admit is refused BY NAME before any gradient is computed.
    """

    def __init__(self, config, weights=None, generator=None, lr=1e-3,
                 betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01,
                 lr_schedule=None, max_norm=None, accumulation_steps=1,
                 numeric_mode=None):
        if not isinstance(config, SambaConfig):
            raise TypeError("mojolearn.SambaStack: config must be a SambaConfig")
        if (weights is None) == (generator is None):
            raise ValueError("mojolearn.SambaStack: pass exactly one of "
                             "weights= or generator=")
        self.config = config
        self.numeric_mode = numeric_mode
        self.names = [n for n, _ in config.registry()]
        self.shapes = {n: tuple(s) for n, s in config.registry()}
        self.offsets = [0]
        for n in self.names:
            self.offsets.append(self.offsets[-1] + int(np.prod(self.shapes[n])))
        self.n_total = self.offsets[-1]
        self.flat = np.zeros(self.n_total, dtype=np.float32)
        self.arrays = {}
        for j, n in enumerate(self.names):
            self.arrays[n] = self.flat[self.offsets[j]:self.offsets[j + 1]].reshape(self.shapes[n])
        self.generator = generator if generator is not None else T.Generator(0, numeric_mode)
        if weights is not None:
            self.load_weights(weights)
        else:
            for n in self.names:
                self.arrays[n][...] = _init_tensor(self.generator, n, self.shapes[n])
        self.max_norm = None if max_norm is None else float(max_norm)
        self.optimizer = T.AdamW([self.arrays[n] for n in self.names], lr=lr,
                                 betas=betas, eps=eps, weight_decay=weight_decay,
                                 lr_schedule=lr_schedule,
                                 accumulation_steps=accumulation_steps,
                                 numeric_mode=numeric_mode)
        self.last_ = None

    # -- weights ----------------------------------------------------------
    def load_weights(self, weights):
        """Copy the registry's tensors in from a dict keyed by name, exact
        key set and exact shapes, float32 only."""
        missing = [n for n in self.names if n not in weights]
        extra = [n for n in weights if n not in self.shapes]
        if missing or extra:
            raise ValueError("mojolearn.SambaStack: weight dict mismatch; "
                             "missing %r, unknown %r" % (missing, extra))
        for n in self.names:
            a = np.asarray(weights[n])
            if a.dtype != np.float32:
                raise TypeError("mojolearn.SambaStack: %s has dtype %s; float32 only"
                                % (n, a.dtype))
            if a.shape != self.shapes[n]:
                raise ValueError("mojolearn.SambaStack: %s has shape %r, want %r"
                                 % (n, a.shape, self.shapes[n]))
            if not np.isfinite(a).all():
                raise ValueError("mojolearn.SambaStack: %s is not finite" % n)
            self.arrays[n][...] = a

    def parameters(self):
        """The registry as `{name: array}` (views of the flat buffer)."""
        return {n: self.arrays[n] for n in self.names}

    def _block(self, i):
        kind = self.config.layers[i]
        w = {n: self.arrays["layers.%d.%s" % (i, n)]
             for n, _ in self.config.block_shapes(kind)}
        if kind == "mamba3":
            return Mamba3Block(w, numeric_mode=self.numeric_mode)
        c = self.config
        return TransformerBlock(w, n_heads=c.n_heads, n_kv_heads=c.n_kv_heads,
                                head_dim=c.head_dim, numeric_mode=self.numeric_mode)

    def _head_weight(self):
        return self.arrays["embed.weight" if self.config.tie_embeddings
                           else "lm_head.weight"]

    # -- forward ------------------------------------------------------------
    @staticmethod
    def _ids(x, what):
        x = np.asarray(x)
        if x.ndim != 2 or x.dtype.kind not in "iu":
            raise ValueError("mojolearn.SambaStack: %s must be (B, L) integer ids" % what)
        return np.ascontiguousarray(x, dtype=np.int32)

    def _forward(self, inputs, dropout_stream=None, token_offset=0):
        """The forward with every block input kept for the backward."""
        c = self.config
        ids = self._ids(inputs, "inputs")
        b, l = ids.shape
        if ids.min() < 0 or ids.max() >= c.vocab:
            raise ValueError("mojolearn.SambaStack: inputs must be in [0, vocab)")
        x = T.embedding_forward(self.arrays["embed.weight"], ids.reshape(-1),
                                self.numeric_mode).reshape(b, l, c.d_model)
        key = None
        if c.dropout > 0.0 and dropout_stream is not None:
            x, key = self.generator.dropout(x, c.dropout,
                                            offset=token_offset * c.d_model,
                                            stream=dropout_stream)
        xs = []
        for i in range(len(c.layers)):
            xs.append(x)
            x = self._block(i).forward(x)
        hn = T.rms_norm_forward(x, self.arrays["norm_f.weight"], c.norm_eps,
                                self.numeric_mode)
        logits = T.linear_forward(hn.reshape(b * l, c.d_model),
                                  self._head_weight(), self.numeric_mode)
        return {"ids": ids, "key": key, "xs": xs, "h": x, "hn": hn,
                "logits": logits}

    def forward(self, inputs):
        """`(B, L)` ids in, `(B, L, vocab)` float32 logits out, no dropout."""
        acts = self._forward(inputs)
        b, l = acts["ids"].shape
        return acts["logits"].reshape(b, l, self.config.vocab)

    def loss(self, inputs, targets):
        """Mean cross-entropy over the targets (no dropout, no gradient)."""
        acts = self._forward(inputs)
        y = self._ids(targets, "targets").reshape(-1)
        return float(T.cross_entropy(acts["logits"], y, numeric_mode=self.numeric_mode))

    # -- backward -----------------------------------------------------------
    def _refuse_no_backward(self):
        for i, kind in enumerate(self.config.layers):
            if kind == "attention" and not callable(getattr(TransformerBlock, "backward", None)):
                raise NotImplementedError(
                    "mojolearn.SambaStack: layer %d is an attention block and "
                    "loaded TransformerBlock wrapper has no callable backward; "
                    "install a matching wrapper and IDENTICAL transformer "
                    "extension. The layer gradient cannot be skipped "
                    "(python/mojolearn/_samba_impl.py)" % i)

    def loss_and_grads(self, inputs, targets, num_items=None,
                       dropout_stream=None, token_offset=0):
        """One forward and the full backward. Returns `(loss, grads)` with
        `grads` a list in registry order. `num_items` (an integer) makes the
        divisor of the sum-reduced loss and gradient the whole step's target
        count, which is how a microbatch's gradient equals its slice of the
        unsplit one; `None` divides by this call's own target count."""
        self._refuse_no_backward()
        c = self.config
        acts = self._forward(inputs, dropout_stream, token_offset)
        ids = acts["ids"]
        b, l = ids.shape
        y = self._ids(targets, "targets")
        if y.shape != ids.shape:
            raise ValueError("mojolearn.SambaStack: targets must match inputs' shape")
        y = y.reshape(-1)
        count = int(np.count_nonzero(y != T._IGNORE_INDEX_DEFAULT))
        items = count if num_items is None else int(num_items)
        loss, dlogits = T.cross_entropy(acts["logits"], y, reduction="sum",
                                        num_items=items, return_grad=True,
                                        numeric_mode=self.numeric_mode)
        grads = {}
        dhn, dw_head = T.linear_backward(
            dlogits, acts["hn"].reshape(b * l, c.d_model), self._head_weight(),
            self.numeric_mode)
        dh, grads["norm_f.weight"] = T.rms_norm_backward(
            dhn.reshape(b, l, c.d_model), acts["h"], self.arrays["norm_f.weight"],
            c.norm_eps, self.numeric_mode)
        for i in reversed(range(len(c.layers))):
            g = self._block(i).backward(acts["xs"][i], dh)
            dh = g.pop("x")
            for n, v in g.items():
                grads["layers.%d.%s" % (i, n)] = v
        if acts["key"] is not None:
            dh = self.generator.dropout_backward(dh, acts["key"])
        d_emb = T.embedding_backward(dh.reshape(b * l, c.d_model), ids.reshape(-1),
                                     c.vocab, self.numeric_mode)
        if c.tie_embeddings:
            # The tied gradient is ONE pair add, embedding first, no
            # alignment claim (tokens=None).
            d_emb = T.accumulate_grads([d_emb, dw_head], tokens=None,
                                       numeric_mode=self.numeric_mode)
        else:
            grads["lm_head.weight"] = dw_head
        grads["embed.weight"] = d_emb
        return float(loss), [grads[n] for n in self.names]

    def train_step(self, inputs, targets):
        """One optimizer step over `(B, L)` inputs and targets: the batch
        rows are split into `accumulation_steps` microbatches, each
        microbatch's gradient is computed with the whole step's divisor and
        the pieces are combined by the clause 9.2 tree. Returns a dict with
        `loss`, `lr`, `step`, `total_norm`."""
        ids = self._ids(inputs, "inputs")
        y = self._ids(targets, "targets")
        if y.shape != ids.shape:
            raise ValueError("mojolearn.SambaStack: targets must match inputs' shape")
        b, l = ids.shape
        a = self.optimizer.accumulation_steps
        tokens = b * l
        if b % a != 0:
            raise ValueError("mojolearn.SambaStack.train_step: batch rows B=%d "
                             "are not divisible by accumulation_steps=%d" % (b, a))
        if a > 1 and not T.accumulation_is_aligned(tokens, a, self.numeric_mode):
            raise ValueError(
                "mojolearn.SambaStack.train_step: MISALIGNED microbatch split, "
                "T = %d tokens at A = %d does not satisfy optimizer contract "
                "clause 9.2 (leaf size, T mod L, A divides P, A a power of "
                "two); refused before any gradient is computed "
                "(python/mojolearn/_samba_impl.py)" % (tokens, a))
        count = int(np.count_nonzero(y.reshape(-1) != T._IGNORE_INDEX_DEFAULT))
        stream = (self.generator.next_stream()
                  if self.config.dropout > 0.0 else None)
        rows = b // a
        losses, parts = [], []
        for k in range(a):
            sl = slice(k * rows, (k + 1) * rows)
            loss_k, g_k = self.loss_and_grads(ids[sl], y[sl], num_items=count,
                                              dropout_stream=stream,
                                              token_offset=k * rows * l)
            losses.append(np.array([loss_k], dtype=np.float32))
            parts.append(g_k)
        if a == 1:
            loss = float(losses[0][0])
            total = self.optimizer.step(parts[0], max_norm=self.max_norm)
        else:
            loss = float(T.accumulate_grads(losses, tokens=None,
                                            numeric_mode=self.numeric_mode)[0])
            total = self.optimizer.step_accumulated(parts, tokens,
                                                    max_norm=self.max_norm)
        self.last_ = {"loss": loss, "lr": self.optimizer.lr_,
                      "step": self.optimizer.t, "total_norm": total}
        return dict(self.last_)

    # -- state ----------------------------------------------------------------
    def state_dict(self):
        o = self.optimizer
        sched = None if o.lr_schedule is None else o.lr_schedule.config()
        return {
            "schema": _STATE_SCHEMA, "profile": PROFILE,
            "numeric_mode": _backend._CODE_MODE.get(
                T._load(self.numeric_mode).training_numeric_mode(), "unknown"),
            "config": self.config.to_dict(),
            "registry": [{"name": n, "shape": list(self.shapes[n]),
                          "offset": self.offsets[j],
                          "size": self.offsets[j + 1] - self.offsets[j]}
                         for j, n in enumerate(self.names)],
            "parameters": self.flat.copy(),
            "exp_avg": o.exp_avg.copy(), "exp_avg_sq": o.exp_avg_sq.copy(),
            "buf_initialized": o.buf_initialized.copy(), "t": int(o.t),
            "optimizer": {"kind": "adamw", "lr": o.lr, "beta1": o.betas[0],
                          "beta2": o.betas[1], "eps": o.eps,
                          "weight_decay": o.weight_decay,
                          "max_norm": self.max_norm,
                          "accumulation_steps": o.accumulation_steps},
            "schedule": sched,
            "rng": self.generator.state_dict(),
        }

    def load_state_dict(self, state):
        if state.get("schema") != _STATE_SCHEMA or state.get("profile") != PROFILE:
            raise ValueError("mojolearn.SambaStack: state schema/profile mismatch")
        if state["config"] != self.config.to_dict():
            raise ValueError("mojolearn.SambaStack: state config differs from this stack's")
        p = np.ascontiguousarray(state["parameters"], dtype=np.float32)
        if p.shape != (self.n_total,):
            raise ValueError("mojolearn.SambaStack: parameters hold %d floats, "
                             "the registry is %d" % (p.size, self.n_total))
        self.flat[...] = p
        self.optimizer.load_state_dict({
            "t": state["t"], "exp_avg": state["exp_avg"],
            "exp_avg_sq": state["exp_avg_sq"],
            "buf_initialized": state["buf_initialized"]})
        oc = state["optimizer"]
        self.optimizer.lr = float(oc["lr"])
        self.optimizer.betas = (float(oc["beta1"]), float(oc["beta2"]))
        self.optimizer.eps = float(oc["eps"])
        self.optimizer.weight_decay = float(oc["weight_decay"])
        self.max_norm = None if oc["max_norm"] is None else float(oc["max_norm"])
        if int(oc["accumulation_steps"]) != self.optimizer.accumulation_steps:
            raise ValueError("mojolearn.SambaStack: accumulation_steps in the "
                             "state differs from this stack's; the microbatch "
                             "count is part of the run's numerical specification")
        self.optimizer.lr_schedule = (None if state["schedule"] is None
                                      else T._Schedule.from_config(state["schedule"]))
        self.generator.load_state_dict(state["rng"])
        return self

    # -- checkpoint (the byte-LM's JSON+hex+sha256 envelope, generalized) ---
    _ARRAYS = (("parameters", "<f4"), ("exp_avg", "<f4"),
               ("exp_avg_sq", "<f4"), ("buf_initialized", "<i4"))

    def save_checkpoint(self, path):
        """Canonical JSON with the four state arrays as little-endian hex,
        a sha256 over the canonical payload, written atomically."""
        payload = self.state_dict()
        for key, dtype in self._ARRAYS:
            v = np.asarray(payload[key], dtype=dtype, order="C")
            payload[key] = {"dtype": dtype, "shape": list(v.shape),
                            "hex": v.tobytes().hex()}
        envelope = {"schema": _CHECKPOINT_SCHEMA, "payload": payload,
                    "payload_sha256": hashlib.sha256(_canonical(payload)).hexdigest()}
        encoded = _canonical(envelope) + b"\n"
        if len(encoded) > _CHECKPOINT_LIMIT:
            raise ValueError("mojolearn.SambaStack: checkpoint exceeds the size limit")
        path = Path(path)
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name + ".",
                                             delete=False) as stream:
                tmp = stream.name
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp, path)
            tmp = None
        finally:
            if tmp is not None:
                os.unlink(tmp)
        return hashlib.sha256(encoded).hexdigest()

    @classmethod
    def from_checkpoint(cls, path, numeric_mode=None):
        with Path(path).open("rb") as stream:
            encoded = stream.read(_CHECKPOINT_LIMIT + 1)
        if len(encoded) > _CHECKPOINT_LIMIT:
            raise ValueError("mojolearn.SambaStack: checkpoint exceeds the size limit")
        envelope = json.loads(encoded)
        if (not isinstance(envelope, dict)
                or set(envelope) != {"schema", "payload", "payload_sha256"}
                or envelope["schema"] != _CHECKPOINT_SCHEMA):
            raise ValueError("mojolearn.SambaStack: checkpoint schema mismatch")
        payload = envelope["payload"]
        if hashlib.sha256(_canonical(payload)).hexdigest() != envelope["payload_sha256"]:
            raise ValueError("mojolearn.SambaStack: checkpoint integrity mismatch")
        for key, dtype in cls._ARRAYS:
            d = payload[key]
            if d["dtype"] != dtype:
                raise ValueError("mojolearn.SambaStack: checkpoint tensor dtype mismatch")
            raw = bytes.fromhex(d["hex"])
            payload[key] = np.frombuffer(raw, dtype=dtype).astype(
                np.int32 if dtype == "<i4" else np.float32, copy=True).reshape(d["shape"])
        config = SambaConfig.from_dict(payload["config"])
        oc = payload["optimizer"]
        sched = (None if payload["schedule"] is None
                 else T._Schedule.from_config(payload["schedule"]))
        weights = {}
        for entry in payload["registry"]:
            weights[entry["name"]] = payload["parameters"][
                entry["offset"]:entry["offset"] + entry["size"]].reshape(entry["shape"])
        stack = cls(config, weights=weights, lr=oc["lr"],
                    betas=(oc["beta1"], oc["beta2"]), eps=oc["eps"],
                    weight_decay=oc["weight_decay"], lr_schedule=sched,
                    max_norm=oc["max_norm"],
                    accumulation_steps=oc["accumulation_steps"],
                    numeric_mode=numeric_mode)
        return stack.load_state_dict(payload)
