# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`CausalLM`: a whole decoder-only language model assembled from a Hugging
Face checkpoint (lane/model-loader, 2026-09-17).

    lm = CausalLM.load("path/to/model", weight_format="bfloat16")
    logits = lm.forward(ids)                    # (B, L) int32 in, (B, L, vocab) float32 out
    state = lm.allocate_state(B, max_tokens)
    logits = lm.forward(ids, state)             # prefill, state carried
    logits = lm.step(next_ids, state)           # one token per row, (B, vocab)
    out = lm.generate(ids, 16)                  # greedy, (B, L + 16) int32

THE STACK. Embedding gather -> N blocks in checkpoint order -> final RMSNorm
-> tied or untied head, exactly the spelling `SambaStack._forward` and
`SambaInference._run` use for the same three non-block steps: on a GPU
column the `_training_impl` primitives (`embedding_forward`,
`rms_norm_forward`, `linear_forward`; the `_mojolearn_training` binding),
on the CPU column the three entries of `_mojolearn_neural_host` at the
addresses `SambaInference._run` hands them. So those bits are the ones the
Samba lanes certified; nothing new is spelled here. The blocks are the
library's own (`TransformerBlock`, `Mamba1Block`, `Mamba2Block` on a GPU;
their `*Inference` twins on the CPU), built ONCE in `__init__` and kept, so
the device-resident session a `TransformerBlock` carries per Python object
(commit 9da5c4685) lives as long as this model does; `SambaStack` rebuilds
its blocks per call and this class deliberately does not.

WHICH OPTIONS REACH THE BLOCK. `config.py` plans every interface option
with its value; `_block_kwargs` reads the LIVE block signature and passes a
keyword only when the value differs from the interface default, and refuses
by name any option the live block does not accept whose value differs from
what today's block fixes. That is how a Llama 3 config is refused at
`norm_eps=1e-5` today (the block fixes 1e-6) and honored the day lane B1's
signature lands, without a line changing here.

WEIGHT FORMATS. `weight_format="float32"` keeps every tensor float32.
`"bfloat16"` hands the blocks their projection matrices as
`lowbit.BF16Weight`: bits copied straight from a BF16 checkpoint (no
conversion at all), or `lowbit.pack_one` of a float32/float16 checkpoint;
`"int8"` is `lowbit.pack_one(..., "int8")` of the materialized float32.
Which tensors count as projections is BY NAME (`_PROJECTIONS`): the
attention and MLP matrices, Mamba's `in_proj`/`x_proj`/`dt_proj`/`out_proj`;
`A_log` stays float32 and so do the embedding table and the head, which
never pass through a block (the head GEMM takes float32). Every block then
reports `weight_format` equal to what was asked.

POSITIONS. A transformer block refuses absolute positions at or above 8192
in Mojo (DEVIATION 812). `max_positions` defaults to the smaller of the
config's `max_position_embeddings` and that ceiling, is refused by name
above either, and every `forward`, `step` and `generate` checks the
sequence length against it BEFORE any kernel runs. Mamba models are
recurrent and carry no position; their `max_positions` is None.

GREEDY DECODING. `generate` is greedy only (`greedy=True`; anything else is
refused by name): the next id is the argmax of the last position's logits,
ties to the LOWEST index, spelled as a sequential scan with a strict `>`
over the row (`_byte_lm_host._greedy_next_bytes`'s rule), which no fold
order can move. The logits are the certified bits; the argmax is a host
selection over them, part of no identity claim.

NO REAL CHECKPOINT HAS BEEN LOADED THROUGH THIS CLASS YET (lane C runs
that). This lane loaded the synthetic checkpoints of
`tests/test_models_loader.py` only, and it moves no arithmetic bit, so it
takes no DEVIATION number.
"""
import inspect
import os

from .. import _backend
from .. import lowbit as _lowbit
from .._array import Array
from .._buffer import addr, addr_ro, all_finite, as_i32_c, empty
from .._bufcheck import flat_view, is_integer, probe
from .config import (FIXED_TODAY, INTERFACE_DEFAULTS, POSITION_CEILING, HFConfig,
                     UnsupportedModel, plan_for)
from .safetensors import Checkpoint

__all__ = ["CausalLM", "CausalLMState"]

#: The block-dict keys that are projection matrices, per kind: the tensors a
#: low-bit format packs. Everything else in a block dict stays float32.
_PROJECTIONS = {
    "transformer": ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                    "gate_proj.weight", "up_proj.weight", "down_proj.weight"),
    "mamba1": ("in_proj.weight", "x_proj.weight", "dt_proj.weight", "out_proj.weight"),
    "mamba2": ("in_proj.weight", "out_proj.weight"),
}

#: Interface option -> the config field it came from, for the refusals.
_OPTION_FIELD = {
    "n_kv_heads": "num_key_value_heads", "head_dim": "head_dim", "window": "sliding_window",
    "rope_theta": "rope_theta", "rope_scaling": "rope_scaling", "rope_dim": "partial_rotary_factor",
    "max_positions": "max_position_embeddings", "qkv_bias": "attention_bias", "o_bias": "attention_bias",
    "norm": "(the family's norm)", "norm_eps": "rms_norm_eps", "norm_bias": "(the family's norm)",
    "mlp": "hidden_act", "mlp_bias": "mlp_bias", "qk_norm": "(qwen3 q_norm/k_norm)",
    "attn_softcap": "attn_logit_softcapping",
}


def _accepted_options(cls):
    """The keyword names the live block constructor accepts, read off its
    signature (`NumericModeMixin` wraps `__init__` with `functools.wraps`,
    and `inspect.signature` follows `__wrapped__`). `None` means the
    constructor takes `**kwargs` and every option must be tried."""
    sig = inspect.signature(cls.__init__)
    names = set()
    for p in sig.parameters.values():
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            return None
        names.add(p.name)
    return names - {"self", "weights"}


def _block_kwargs(plan, cls):
    """The keywords to pass to `cls` for `plan`, or `UnsupportedModel` by
    name for an option the live block cannot honor at the plan's value."""
    accepted = _accepted_options(cls)
    kwargs = {}
    for opt, value in plan.block_options.items():
        default = INTERFACE_DEFAULTS[opt]
        if accepted is None or opt in accepted:
            if value != default:
                kwargs[opt] = value
            continue
        fixed = FIXED_TODAY.get(opt, default)
        if value != fixed:
            raise UnsupportedModel(
                f"mojolearn.models: model_type {plan.model_type!r}: option {opt}={value!r} "
                f"(config field {_OPTION_FIELD.get(opt, opt)}) is not accepted by {cls.__name__} on this "
                f"build, which fixes {opt}={fixed!r}; the accepted options are {sorted(accepted)}")
    return kwargs


def _route(device):
    if device == "auto":
        try:
            vendor = _backend.vendor()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"mojolearn.models.CausalLM: cannot read the selected backend's vendor: {exc}") from exc
        return "cpu" if vendor in (None, "cpu") else "gpu"
    if device in ("cpu", "gpu"):
        return device
    raise ValueError(f"mojolearn.models.CausalLM: device must be 'auto', 'cpu' or 'gpu', got {device!r}")


def _block_classes(route):
    if route == "gpu":
        from .._mamba_impl import Mamba1Block, Mamba2Block
        from .._transformer_impl import TransformerBlock
        return {"transformer": TransformerBlock, "mamba1": Mamba1Block, "mamba2": Mamba2Block}
    from ..neural_inference import Mamba1BlockInference, Mamba2BlockInference, TransformerBlockInference
    return {"transformer": TransformerBlockInference, "mamba1": Mamba1BlockInference,
            "mamba2": Mamba2BlockInference}


class _GpuPrimitives:
    """The three non-block steps as `SambaStack._forward` spells them."""

    name = "_mojolearn_training"

    def __init__(self):
        from .. import _training_impl as T
        self._T = T

    def embedding(self, table, ids_flat):
        return self._T.embedding_forward(table, ids_flat)

    def rms_norm(self, x2d, weight, eps):
        return self._T.rms_norm_forward(x2d, weight, eps)

    def linear(self, x2d, weight):
        return self._T.linear_forward(x2d, weight)


class _CpuPrimitives:
    """The three non-block steps as `SambaInference._run` spells them, at
    the same addresses in the same order."""

    name = "_mojolearn_neural_host"

    def __init__(self):
        self._ext = _backend.load_host_module(self.name)

    def embedding(self, table, ids_flat):
        n = int(ids_flat.size)
        v, d = int(table.shape[0]), int(table.shape[1])
        y = empty((n, d), "<f4")
        self._ext.embedding_forward(
            [addr(y, name="x"), addr_ro(table, name="embed"), addr_ro(ids_flat, name="ids")], [n, v, d])
        return y

    def rms_norm(self, x2d, weight, eps):
        n, d = int(x2d.shape[0]), int(x2d.shape[1])
        y = empty((n, d), "<f4")
        self._ext.rms_norm_forward(
            [addr(y, name="hn"), addr_ro(x2d, name="x"), addr_ro(weight, name="norm")], [n, d, float(eps)])
        return y

    def linear(self, x2d, weight):
        n, d = int(x2d.shape[0]), int(x2d.shape[1])
        v = int(weight.shape[0])
        y = empty((n, v), "<f4")
        self._ext.linear_forward(
            [addr(y, name="logits"), addr_ro(x2d, name="hn"), addr_ro(weight, name="head")], [n, v, d])
        return y


class CausalLMState:
    """The decode state: one block state per layer in stack order (each
    caller-owned and documented by its block class), `batch_size`,
    `max_tokens` (the capacity the transformer caches were allocated for)
    and `positions`, how many tokens each row has consumed."""

    def __init__(self, batch_size, max_tokens, layers, *, owner=None):
        self.batch_size = int(batch_size)
        self.max_tokens = int(max_tokens)
        self.layers = list(layers)
        self.positions = 0
        self._owner = owner

    def __repr__(self):
        return (f"CausalLMState(batch_size={self.batch_size}, max_tokens={self.max_tokens}, "
                f"layers={len(self.layers)}, positions={self.positions})")


def _float_weight(a, name):
    pb = probe(a)
    if is_integer(pb.format):
        raise TypeError(f"mojolearn.models.CausalLM: tensor {name!r} is an integer tensor ({pb.format}); a weight must be F32, F16 or BF16")
    return a


def _argmax_last(logits, b, l, v):
    """The argmax of row `b`'s LAST position, ties to the lowest index: a
    sequential scan with a strict `>` (`_greedy_next_bytes`'s rule)."""
    flat = flat_view(logits, "f")
    out = []
    for row in range(b):
        base = ((row * l) + l - 1) * v
        best = 0
        best_v = flat[base]
        for k in range(1, v):
            x = flat[base + k]
            if x > best_v:
                best = k
                best_v = x
        out.append(best)
    return out


class CausalLM:
    """See the module header. Build with `load` (a checkpoint directory) or
    directly from a `ModelPlan` and a weight dict shaped like
    `_read_weights` returns."""

    def __init__(self, plan, weights, *, weight_format="float32", max_positions=None, device="auto"):
        if weight_format not in _lowbit.FORMATS:
            raise ValueError(f"mojolearn.models.CausalLM: weight_format must be one of {_lowbit.FORMATS}, got {weight_format!r}")
        self.plan = plan
        self.config = plan.config
        self.weight_format = weight_format
        self.device = _route(device)
        self.d_model = int(plan.d_model)
        self.vocab_size = int(plan.vocab_size)
        self.n_layers = int(plan.n_layers)
        self.kind = plan.kind
        self.unused_names = list(weights.get("unused", ()))
        # -- positions
        if plan.kind == "transformer":
            cap = min(int(plan.max_position_embeddings or POSITION_CEILING), POSITION_CEILING)
            if max_positions is None:
                max_positions = cap
            mp = int(max_positions)
            if mp < 1:
                raise ValueError(f"mojolearn.models.CausalLM: max_positions must be positive, got {max_positions!r}")
            if mp > POSITION_CEILING:
                raise UnsupportedModel(
                    f"mojolearn.models: model_type {plan.model_type!r}: max_positions={mp} exceeds the "
                    f"transformer block's absolute-position ceiling {POSITION_CEILING} (DEVIATION 812)")
            if plan.max_position_embeddings is not None and mp > int(plan.max_position_embeddings):
                raise UnsupportedModel(
                    f"mojolearn.models: model_type {plan.model_type!r}: max_positions={mp} exceeds the config's "
                    f"max_position_embeddings={plan.max_position_embeddings}")
            self.max_positions = mp
        else:
            self.max_positions = None if max_positions is None else int(max_positions)
        # -- the non-block tensors, float32 always
        embed = _float_weight(weights["embed"], plan.embed_name)
        if tuple(embed.shape) != (self.vocab_size, self.d_model):
            raise UnsupportedModel(
                f"mojolearn.models: model_type {plan.model_type!r}: vocab_size={self.vocab_size} and "
                f"hidden_size={self.d_model} but {plan.embed_name} has shape {tuple(embed.shape)}")
        norm = _float_weight(weights["norm"], plan.norm_name)
        if tuple(norm.shape) != (self.d_model,):
            raise UnsupportedModel(
                f"mojolearn.models: model_type {plan.model_type!r}: {plan.norm_name} has shape {tuple(norm.shape)}, want ({self.d_model},)")
        if plan.head_name is None:
            head = embed
        else:
            head = _float_weight(weights["head"], plan.head_name)
            if tuple(head.shape) != (self.vocab_size, self.d_model):
                raise UnsupportedModel(
                    f"mojolearn.models: model_type {plan.model_type!r}: {plan.head_name} has shape "
                    f"{tuple(head.shape)}, want ({self.vocab_size}, {self.d_model})")
        for name, a in ((plan.embed_name, embed), (plan.norm_name, norm), (plan.head_name, head)):
            if name is not None and not all_finite(a):
                raise ValueError(f"mojolearn.models.CausalLM: {name} is not finite")
        self._embed, self._norm, self._head = embed, norm, head
        self.norm_eps = float(plan.norm_eps)
        # -- the blocks, once
        classes = _block_classes(self.device)
        cls = classes[plan.kind]
        kwargs = _block_kwargs(plan, cls) if plan.kind == "transformer" else dict(plan.block_kwargs)
        if plan.kind == "transformer":
            kwargs["n_heads"] = int(plan.n_heads)
        layers = weights["layers"]
        if len(layers) != self.n_layers:
            raise ValueError(f"mojolearn.models.CausalLM: {len(layers)} layer dicts for num_hidden_layers={self.n_layers}")
        self._block_class = cls
        self._block_kwargs = kwargs
        self._blocks = [cls(w, **kwargs) for w in layers]
        for i, blk in enumerate(self._blocks):
            if blk.weight_format != weight_format:
                raise RuntimeError(
                    f"mojolearn.models.CausalLM: layer {i} reports weight_format {blk.weight_format!r}, asked {weight_format!r}")
        self._prims = _GpuPrimitives() if self.device == "gpu" else _CpuPrimitives()

    # ------------------------------------------------------------ loading
    @classmethod
    def load(cls, path, *, weight_format="float32", max_positions=None, device="auto"):
        """A checkpoint directory (`config.json` plus `model.safetensors` or
        `model.safetensors.index.json` and its shards)."""
        if weight_format not in _lowbit.FORMATS:
            raise ValueError(f"mojolearn.models.CausalLM.load: weight_format must be one of {_lowbit.FORMATS}, got {weight_format!r}")
        path = os.fspath(path)
        if not os.path.isdir(path):
            raise FileNotFoundError(f"mojolearn.models.CausalLM.load: {path} is not a directory")
        config = HFConfig.from_json(path)
        plan = plan_for(config)
        route = _route(device)
        # refuse an option the live block cannot take BEFORE reading gigabytes
        if plan.kind == "transformer":
            _block_kwargs(plan, _block_classes(route)[plan.kind])
        ckpt = Checkpoint.open(path)
        try:
            weights = cls._read_weights(ckpt, plan, weight_format)
        finally:
            ckpt.close()
        return cls(plan, weights, weight_format=weight_format, max_positions=max_positions, device=route)

    @staticmethod
    def _read_weights(ckpt, plan, weight_format):
        """`{"embed", "norm", "head" (or absent), "layers": [dict...],
        "unused": [...]}` from the checkpoint, the projections packed when a
        low-bit format is asked."""
        need = plan.checkpoint_names()
        missing = [n for n in need if n not in ckpt]
        if missing:
            raise UnsupportedModel(
                f"mojolearn.models: model_type {plan.model_type!r}: the checkpoint lacks {len(missing)} of the "
                f"{len(need)} tensors the plan names; first missing: {missing[:5]}")
        projections = _PROJECTIONS[plan.kind]
        out = {"embed": ckpt.read(plan.embed_name), "norm": ckpt.read(plan.norm_name)}
        if plan.head_name is not None:
            out["head"] = ckpt.read(plan.head_name)
        layers = []
        for i in range(plan.n_layers):
            w = {}
            for key, name, rows in plan.layer_weights(i):
                info = ckpt.info(name)
                packed = key in projections and weight_format != "float32"
                if packed and weight_format == "bfloat16" and info.dtype == "BF16" and rows is None and len(info.shape) == 2:
                    w[key] = ckpt.read(name, bf16="bits")  # the checkpoint's own bits, no conversion
                    continue
                a = _float_weight(ckpt.read(name), name)
                if rows is not None:
                    if a.ndim != 2 or rows[1] > a.shape[0]:
                        raise UnsupportedModel(
                            f"mojolearn.models: model_type {plan.model_type!r}: {name} has shape {tuple(a.shape)}, "
                            f"too small for the row slice [{rows[0]}, {rows[1]}) that carries {key}")
                    a = a[rows[0]:rows[1]]
                w[key] = _lowbit.pack_one(a, weight_format, key) if packed else a
            layers.append(w)
        out["layers"] = layers
        out["unused"] = [n for n in ckpt.names() if n not in need]
        return out

    # ------------------------------------------------------------ surface
    @property
    def blocks(self):
        return list(self._blocks)

    def parameters(self):
        """Every tensor by CHECKPOINT name, as held (packed where packed)."""
        p = self.plan
        out = {p.embed_name: self._embed, p.norm_name: self._norm}
        if p.head_name is not None:
            out[p.head_name] = self._head
        for i, blk in enumerate(self._blocks):
            names = dict((k, n) for k, n, _ in p.layer_weights(i))
            for key, w in zip(blk._W_NAMES, blk._w):
                out.setdefault(names.get(key, f"layers.{i}.{key}"), w)
        return out

    def _ids(self, x, what, step=False):
        pb = probe(x)
        if not is_integer(pb.format):
            raise TypeError(f"mojolearn {what}: ids must be integer, got buffer format {pb.format!r}")
        if step:
            if len(pb.shape) not in (1, 2) or (len(pb.shape) == 2 and pb.shape[1] != 1):
                raise ValueError(f"mojolearn {what}: ids must be (B,) or (B, 1) for a step, got shape {pb.shape}")
            ids = as_i32_c(x, ndim=None, name="ids")[0].reshape((int(pb.shape[0]), 1))
        else:
            if len(pb.shape) != 2:
                raise ValueError(f"mojolearn {what}: ids must be (B, L), got shape {pb.shape}")
            ids = as_i32_c(x, ndim=2, name="ids")[0]
        b, l = int(ids.shape[0]), int(ids.shape[1])
        if b < 1 or l < 1:
            raise ValueError(f"mojolearn {what}: B and L must be positive, got ({b}, {l})")
        if ids.min() < 0 or ids.max() >= self.vocab_size:
            raise ValueError(f"mojolearn {what}: ids must be in [0, {self.vocab_size})")
        return ids

    def _check_positions(self, what, state, l):
        have = state.positions if state is not None else 0
        if self.max_positions is not None and have + l > self.max_positions:
            raise ValueError(
                f"mojolearn {what}: {have} + {l} positions exceed max_positions={self.max_positions} "
                "(the transformer block's absolute-position ceiling is 8192, DEVIATION 812)")
        if state is not None and state.positions + l > state.max_tokens:
            raise ValueError(
                f"mojolearn {what}: the state was allocated for {state.max_tokens} tokens and holds "
                f"{state.positions}; {l} more do not fit")

    def _run(self, ids, state, step, what):
        """The stack: embedding, the blocks, the final norm, the head. ONE
        spelling for the stateless forward, the stateful forward and the
        decode step; only what each block is handed differs."""
        b, l = int(ids.shape[0]), int(ids.shape[1])
        if state is not None:
            if not isinstance(state, CausalLMState):
                raise TypeError(f"mojolearn {what}: state must be a CausalLMState (allocate_state)")
            if state._owner is not self:
                raise ValueError(f"mojolearn {what}: state belongs to another model; allocate a state on this model")
            if state.batch_size != b or len(state.layers) != self.n_layers:
                raise ValueError(
                    f"mojolearn {what}: the state holds {state.batch_size} rows and {len(state.layers)} layers, "
                    f"the call has B = {b} and the model {self.n_layers} layers")
        self._check_positions(what, state, l)
        n, d = b * l, self.d_model
        x = self._prims.embedding(self._embed, ids.reshape((n,))).reshape((b, l, d))
        for i, blk in enumerate(self._blocks):
            if state is None:
                x = blk.forward(x)
            elif step:
                x = blk.step(x, state.layers[i])
            else:
                x = blk.forward(x, state.layers[i])
        hn = self._prims.rms_norm(x.reshape((n, d)), self._norm, self.norm_eps)
        logits = self._prims.linear(hn, self._head)
        if state is not None:
            state.positions += l
        return logits.reshape((b, l, self.vocab_size))

    def forward(self, ids, state=None):
        """`(B, L)` int32 ids in, `(B, L, vocab)` float32 logits out. With
        `state=None` every block runs from a zero state and nothing is kept;
        with a `CausalLMState` the blocks read and update it in place."""
        what = "CausalLM.forward"
        return self._run(self._ids(ids, what), state, False, what)

    __call__ = forward

    def allocate_state(self, batch_size, max_tokens):
        """The zero decode state for `batch_size` rows of up to `max_tokens`
        positions each (`Mamba*Block.allocate_state(B)`,
        `TransformerBlock.allocate_state(B, max_tokens)`)."""
        b, smax = int(batch_size), int(max_tokens)
        if b < 1 or smax < 1:
            raise ValueError("mojolearn CausalLM.allocate_state: batch_size and max_tokens must be positive")
        if self.max_positions is not None and smax > self.max_positions:
            raise ValueError(
                f"mojolearn CausalLM.allocate_state: max_tokens={smax} exceeds max_positions={self.max_positions}")
        layers = []
        for blk in self._blocks:
            layers.append(blk.allocate_state(b, smax) if self.kind == "transformer" else blk.allocate_state(b))
        return CausalLMState(b, smax, layers, owner=self)

    def reset_state(self, state):
        """Replace caches with fresh zero state, preserving capacity and batch size.

        States are owned by their allocating model; cross-model reuse is refused.
        """
        if not isinstance(state, CausalLMState) or state._owner is not self:
            raise ValueError("mojolearn CausalLM.reset_state: state belongs to another model")
        fresh = self.allocate_state(state.batch_size, state.max_tokens)
        state.layers = fresh.layers
        state.positions = 0
        return state

    def step(self, ids, state):
        """One decode token per row: `(B,)` or `(B, 1)` ids in, `(B, vocab)`
        logits out, `state` updated in place. It is `forward` at L = 1 with
        each block's `step` (their one spelling for decode)."""
        what = "CausalLM.step"
        if state is None:
            raise ValueError("mojolearn CausalLM.step: state is required (allocate_state(B, max_tokens) makes the fresh one)")
        ids = self._ids(ids, what, step=True)
        out = self._run(ids, state, True, what)
        return out.reshape((int(ids.shape[0]), self.vocab_size))

    def generate(self, ids, max_new_tokens, *, greedy=True):
        """`(B, L)` ids in, `(B, L + max_new_tokens)` int32 ids out: the
        prompt followed by greedy continuations (argmax of the last
        position, ties to the lowest index), one carried state, the prompt
        as one prefill and every later token as one `step`. No stop token:
        exactly `max_new_tokens` are produced per row."""
        what = "CausalLM.generate"
        if greedy is not True:
            raise NotImplementedError(f"mojolearn {what}: greedy=True is the only decoding implemented; got greedy={greedy!r}")
        n_new = int(max_new_tokens)
        if n_new < 0:
            raise ValueError(f"mojolearn {what}: max_new_tokens must be >= 0, got {max_new_tokens!r}")
        ids = self._ids(ids, what)
        b, l = int(ids.shape[0]), int(ids.shape[1])
        rows = ids.tolist()
        if n_new == 0:
            return Array.from_list(rows, "<i4")
        total = l + n_new
        if self.max_positions is not None and total > self.max_positions:
            raise ValueError(f"mojolearn {what}: {l} prompt + {n_new} new tokens exceed max_positions={self.max_positions}")
        state = self.allocate_state(b, total)
        logits = self._run(ids, state, False, what)
        nxt = _argmax_last(logits, b, l, self.vocab_size)
        for k in range(n_new):
            for row, v in zip(rows, nxt):
                row.append(v)
            if k == n_new - 1:
                break
            step_ids = Array.from_list([[v] for v in nxt], "<i4")
            lg = self._run(step_ids, state, True, what)
            nxt = _argmax_last(lg, b, 1, self.vocab_size)
        return Array.from_list(rows, "<i4")

    def __repr__(self):
        return (f"CausalLM(model_type={self.plan.model_type!r}, layers={self.n_layers}, d_model={self.d_model}, "
                f"vocab={self.vocab_size}, weight_format={self.weight_format!r}, device={self.device!r})")
