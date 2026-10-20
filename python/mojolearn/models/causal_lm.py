# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Load causal language models and run prefill, decoding and generation.

`CausalLM.load` accepts a supported Hugging Face checkpoint directory. Token
arrays use `(batch, length)` shape. `forward` returns vocabulary logits;
`generate` appends a bounded continuation. `stream` exposes bounded reads and
resumable generation state. Pass a matching `models.Tokenizer` for text input
and output.

In `identical` mode, mojolearn returns the same bits on Apple, NVIDIA and AMD GPUs.
"""
import array as _pyarray
import inspect
import os

from .. import _backend
from .. import _numeric_profile
from .. import lowbit as _lowbit
from .._array import Array
from .._buffer import _native, addr, addr_ro, all_finite, as_i32_c, empty
from .._bufcheck import is_integer, memcopy, probe
from .config import (FIXED_TODAY, INTERFACE_DEFAULTS, POSITION_CEILING, HFConfig,
                     UnsupportedModel, plan_for)
from .safetensors import Checkpoint

__all__ = ["CausalLM", "CausalLMState", "GenerationStream"]

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
    for p in sig.parameters.values():  # glue: walks the block constructor signature parameters
        if p.kind is inspect.Parameter.VAR_KEYWORD:
            return None
        names.add(p.name)
    return names - {"self", "weights"}


def _block_kwargs(plan, cls):
    """The keywords to pass to `cls` for `plan`, or `UnsupportedModel` by
    name for an option the live block cannot honor at the plan's value."""
    accepted = _accepted_options(cls)
    kwargs = {}
    for opt, value in plan.block_options.items():  # glue: copies block keyword options
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

    def linear_int15(self, x2d, planes):
        """The head under numeric_profile="fixed15_v1": `mojolearn.linalg.matmul_int15`
        on the head's kept planes (lane/lowbit-blocks), a primitive since
        lane/lowbit-default so a layer-owned model runs it on the head's owner."""
        return _matmul_int15(x2d, planes)


class _CpuPrimitives:
    """The three non-block steps as `SambaInference._run` spells them, at
    the same addresses in the same order."""

    name = "_mojolearn_neural_host"

    def __init__(self):
        # The one loader of `_mojolearn_neural_host` (the CPU inference
        # module's), not a second host-module load site in this module.
        from ..neural_inference import _binding
        self._ext = _binding()

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

    def linear_int15(self, x2d, planes):
        return _matmul_int15(x2d, planes)


def _matmul_int15(x2d, planes):
    from .._linalg_impl import matmul_int15
    return matmul_int15(x2d, planes)


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
    """The argmax of row `b`'s LAST position, ties to the lowest index, as an
    int32 `(b,)` Array: a sequential scan with a strict `>`
    (`_greedy_next_bytes`'s rule). The base binding's `argmax_last_rows_f32`
    reads position `l - 1` of each row on the device (the `argmax_rows_f32`
    rule, DEVIATION 2500: strict `>` from index 0, so a NaN never replaces
    and a NaN at index 0 stays); lane cpu4-python: the host row gather
    `gather_rows_bytes` is gone."""
    out = empty((b,), "<i8")
    if b:
        _native("argmax_last_rows_f32")(addr_ro(logits, name="logits"), [b, l, v], addr(out, name="next ids"))
    return out.astype("<i4")


def _prompt_then_new(prompt, b, l, n_new, new_addr, new_step_major):
    """`(b, l + n_new)` int32: each prompt row followed by its new ids, with
    no Python loop over rows or tokens. The ids are assembled step-major
    (`[l + n_new, b]`) by the base binding's `transpose_f32` (a pure 4-byte
    move, every bit pattern unchanged, so int32 ids pass through it) and
    transposed back to C order. `new_addr` holds the new ids as int32,
    step-major `[n_new, b]` when `new_step_major`, else C-order
    `[b, n_new]`."""
    total = l + n_new
    move = _native("transpose_f32")
    steps = empty((total, b), "<i4")
    at = addr(steps, name="ids")
    move(addr_ro(prompt, name="ids"), at, b, l)
    if new_step_major:
        memcopy(at + l * b * 4, new_addr, n_new * b * 4)
    else:
        move(new_addr, at + l * b * 4, b, n_new)
    out = empty((b, total), "<i4")
    move(at, addr(out, name="ids_out"), total, b)
    return out


class CausalLM:
    """See the module header. Build with `load` (a checkpoint directory) or
    directly from a `ModelPlan` and a weight dict shaped like
    `_read_weights` returns."""

    def __init__(self, plan, weights, *, weight_format="float32", max_positions=None, device="auto",
                 numeric_profile=None):
        return _NATIVE.causal_model_init(self, plan, weights,
                                        (weight_format, max_positions, device, numeric_profile),
                                        _NATIVE_CONTEXT)

    # The old owner is retained only for ParallelCausalLM's existing RPC path.
    # It remains explicit runtime debt until that transport has native ownership.
    def _initialize_legacy(self, plan, weights, *, weight_format="float32", max_positions=None, device="auto",
                 numeric_profile=None):
        if weight_format not in _lowbit.FORMATS:
            raise ValueError(f"mojolearn.models.CausalLM: weight_format must be one of {_lowbit.FORMATS}, got {weight_format!r}")
        self.plan = plan
        self.config = plan.config
        self.weight_format = weight_format
        # lane/lowbit-default (2026-09-29): the profile follows the model's
        # FAMILY. Naming none, a transformer model gets the inference default
        # (fixed15_v1) and a Mamba model fp32_v1, which `numeric_profile`
        # then reports; naming one the family does not compute is refused
        # by name (`_numeric_profile.resolve`).
        self.numeric_profile = _numeric_profile.resolve(
            numeric_profile, "mojolearn.models.CausalLM numeric_profile", family=plan.kind)
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
        if plan.norm_name is None:
            # no final norm (the byte-LM format): the head reads the last residual stream
            norm = None
        else:
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
        for name, a in ((plan.embed_name, embed), (plan.norm_name, norm), (plan.head_name, head)):  # glue: three named top-level weights
            if name is not None and a is not None and not all_finite(a):
                raise ValueError(f"mojolearn.models.CausalLM: {name} is not finite")
        self._embed, self._norm, self._head = embed, norm, head
        self.norm_eps = float(plan.norm_eps)
        # -- the blocks, once
        classes = _block_classes(self.device)
        cls = classes[plan.kind]
        kwargs = _block_kwargs(plan, cls) if plan.kind == "transformer" else dict(plan.block_kwargs)
        if plan.kind == "transformer":
            kwargs["n_heads"] = int(plan.n_heads)
        # lane/lowbit-blocks (2026-09-29): a profile other than fp32_v1
        # reaches the transformer block as its keyword; under fp32_v1 no
        # keyword is passed, so the model builds exactly as before. The head
        # is packed below. Mamba blocks do not compute under a profile
        # (resolve above gave them fp32_v1 or refused the name).
        if self.numeric_profile != _numeric_profile.BASELINE:
            if plan.kind != "transformer":  # resolve refuses this first; kept as the backstop
                raise NotImplementedError(
                    f"mojolearn.models.CausalLM: numeric_profile={self.numeric_profile!r} is implemented for "
                    f"transformer models only; model_type {plan.model_type!r} is {plan.kind!r}. It is never "
                    "replaced by 'fp32_v1' silently.")
            kwargs["numeric_profile"] = self.numeric_profile
        layers = weights["layers"]
        if len(layers) != self.n_layers:
            raise ValueError(f"mojolearn.models.CausalLM: {len(layers)} layer dicts for num_hidden_layers={self.n_layers}")
        self._block_class = cls
        self._block_kwargs = kwargs
        self._blocks = self._make_blocks(cls, layers, kwargs)
        for i, blk in enumerate(self._blocks):  # glue: one native block per model layer
            if blk.weight_format != weight_format:
                raise RuntimeError(
                    f"mojolearn.models.CausalLM: layer {i} reports weight_format {blk.weight_format!r}, asked {weight_format!r}")
        self._prims = _GpuPrimitives() if self.device == "gpu" else _CpuPrimitives()
        # The head under the profile: its planes made ONCE here (one scale per
        # vocabulary row, along d_model), `mojolearn.linalg.matmul_int15`
        # per call (the activation quantized per token by the binding).
        self._head_int15 = None
        if self.numeric_profile != _numeric_profile.BASELINE:
            from .._linalg_impl import quantize_int15
            self._head_int15 = quantize_int15(self._head)

        self._initialize_native_runtime()

    def _make_blocks(self, cls, layers, kwargs):
        return [cls(w, **kwargs) for w in layers]  # glue: builds one block per model layer

    # ------------------------------------------------------------ loading
    @classmethod
    def load(cls, path, *, weight_format="float32", max_positions=None, device="auto", numeric_profile=None):
        """A checkpoint directory (`config.json` plus `model.safetensors` or
        `model.safetensors.index.json` and its shards)."""
        return _NATIVE.causal_model_load(cls, path,
                                        (weight_format, max_positions, device, numeric_profile),
                                        _NATIVE_CONTEXT)

    @staticmethod
    def _read_weights(ckpt, plan, weight_format):
        return _NATIVE.causal_model_read_weights(ckpt, plan, weight_format, _NATIVE_CONTEXT)

    # ------------------------------------------------------------ surface
    @property
    def blocks(self):
        return self._runtime.causal_lm_blocks(self)

    def parameters(self):
        """Return tensors by checkpoint name, retaining their existing storage."""
        return self._runtime.causal_lm_parameters(self)

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
        return self._runtime.causal_lm_forward(self, ids, state, step)

    def forward(self, ids, state=None):
        """Return `(B, L, vocabulary)` logits for integer token ids `(B, L)`.

        A supplied state is updated in place. With no state, the sequence starts
        from fresh caches. In `identical` mode, mojolearn returns the same bits
        on Apple, NVIDIA and AMD GPUs.
        """
        return self._runtime.causal_lm_forward(self, ids, state, False)

    __call__ = forward

    def allocate_state(self, batch_size, max_tokens):
        """Allocate fresh caches for `batch_size` sequences and `max_tokens` positions."""
        return self._runtime.causal_lm_allocate(self, batch_size, max_tokens)

    def reset_state(self, state):
        """Reset this model's state, preserving its batch size and capacity."""
        return self._runtime.causal_lm_reset(self, state)

    def step(self, ids, state):
        """Return `(B, vocabulary)` logits and update state for one token per row."""
        return self._runtime.causal_lm_forward(self, ids, state, True)

    def generate(self, ids, max_new_tokens, *, greedy=True, seed=None,
                 temperature=1.0, top_k=0, top_p=1.0, eos_token_id=None,
                 pad_token_id=None):
        """Return the prompt and up to `max_new_tokens` continuation ids.

        Greedy ties choose the lowest id. Sampling (`greedy=False`) requires
        an explicit unsigned 64-bit seed. Temperature applies before top-k,
        then top-p. `eos_token_id` accepts one id or a sequence of ids. Finished
        batch rows are padded; decoding ends when every row finishes or reaches
        the token limit. With no EOS, the output has exactly `(B, L + max_new_tokens)`
        ids. A missing pad id uses the first EOS id, or zero.
        """
        return self._runtime.causal_lm_generate(
            self, ids, max_new_tokens,
            (greedy, seed, temperature, top_k, top_p, eos_token_id, pad_token_id))

    def stream(self, ids, max_new_tokens, *, greedy=True, seed=None,
               temperature=1.0, top_k=0, top_p=1.0, eos_token_id=None,
               pad_token_id=None):
        """Create a continuation stream; `read(n)` returns at most `n` new ids per row.

        Read size does not change sampled ids. `info()` reports completion and
        each row's generated length, including its EOS token.
        """
        return self._runtime.causal_lm_stream(
            self, ids, max_new_tokens,
            (greedy, seed, temperature, top_k, top_p, eos_token_id, pad_token_id))

    def generate_text(self, text, max_new_tokens, *, tokenizer, greedy=True,
                      seed=None, temperature=1.0, top_k=0, top_p=1.0,
                      eos_token_id=None, pad_token_id=None):
        """Tokenize text and return its decoded continuation, excluding EOS tokens.

        Pass a `models.Tokenizer` loaded for this checkpoint. Sampling and
        stopping use the same options as `generate`.
        """
        return self._runtime.causal_lm_generate_text(
            self, tokenizer, text, max_new_tokens,
            (greedy, seed, temperature, top_k, top_p, eos_token_id, pad_token_id))

    def stream_text(self, text, max_new_tokens, *, tokenizer, greedy=True,
                    seed=None, temperature=1.0, top_k=0, top_p=1.0,
                    eos_token_id=None, pad_token_id=None):
        """Create a text continuation stream; `read(n)` consumes at most `n` tokens.

        A read may return an empty string while a UTF-8 character is incomplete.
        Use `info()['done']` to detect completion.
        """
        return self._runtime.causal_lm_text_stream(
            self, tokenizer, text, max_new_tokens,
            (greedy, seed, temperature, top_k, top_p, eos_token_id, pad_token_id))

    def resume_generation(self, checkpoint, *, tokenizer=None):
        """Resume a generation checkpoint on the same model weights and settings."""
        return self._runtime.causal_lm_resume(self, checkpoint, tokenizer)

    def _generate_resident(self, ids, n_new, total, last_logits=None):
        # Retained diagnostic door; arithmetic and lifecycle are native.
        return self._runtime.causal_lm_resident(self, ids, n_new, last_logits, None)

    def __repr__(self):
        return (f"CausalLM(model_type={self.plan.model_type!r}, layers={self.n_layers}, d_model={self.d_model}, "
                f"vocab={self.vocab_size}, weight_format={self.weight_format!r}, device={self.device!r})")


class GenerationStream:
    """A native continuation with explicit bounded reads and resumable state."""

    def read(self, max_tokens=1):
        """Return the next token array or text fragment; stop at EOS or the limit."""
        return self._runtime.causal_lm_read(self, max_tokens)

    def info(self):
        """Return completion, generated lengths and whether the stream remains usable."""
        return self._runtime.causal_lm_info(self)

    def state_dict(self):
        """Return independent state bytes, sampling options and model fingerprint."""
        return self._runtime.causal_lm_snapshot(self)


# Package-import connection work only. The native owner reads these plain
# metadata records; it never calls the package loaders or block constructors.
def _connect_causal_native():
    import _thread
    from ..tokenizer import _binding as runtime_binding
    from .. import _training_impl as training
    from .._transformer_impl import TransformerBlock, TransformerState, _OPT_NAMES
    from .._mamba_impl import Mamba1Block, Mamba2Block, Mamba1State, Mamba2State
    from ..neural_inference import (TransformerBlockInference, Mamba1BlockInference,
                                    Mamba2BlockInference, _causal_native_connections)
    from .config import _NATIVE_CONTEXT as config_context
    from .safetensors import _NATIVE_CONTEXT as checkpoint_context
    runtime = runtime_binding()._module
    _backend.default_mode()
    bindings = {"cpu:identical": _causal_native_connections()}
    vendor = _backend.vendor()
    if vendor not in (None, "cpu"):
        for mode in ("identical", "fast"):  # glue: connects available compiled modes during package import
            if mode == "fast" and vendor != "metal":
                continue
            entries = {}
            for name, target in (("transformer", "_mojolearn_transformer"),
                                 ("mamba1", "_mojolearn_mamba"),
                                 ("mamba2", "_mojolearn_mamba"),
                                 ("primitives", "_mojolearn_training"),
                                 ("linalg", "_mojolearn_linalg")):  # glue: connects compiled sibling modules at package import
                try:
                    entries[name] = _backend.binding(target, mode)
                except (ImportError, RuntimeError, ValueError, OSError) as exc:
                    entries[name] = str(exc)
            bindings["gpu:" + mode] = entries
    return runtime, {
        "runtime": runtime, "backend": _backend, "selected_vendor": vendor,
        "bindings": bindings, "profiles": _numeric_profile.PROFILES,
        "profile_defaults": _numeric_profile._defaults,
        "rejected_profiles": _numeric_profile.REJECTED,
        "config_context": config_context, "checkpoint_context": checkpoint_context,
        "array_class": Array, "bf16_class": _lowbit.BF16Weight,
        "int8_class": _lowbit.Int8Weight, "projections": _PROJECTIONS,
        "optional_names": _OPT_NAMES, "rlock": _thread.RLock,
        "training_module": training,
        "primitive_classes": {"cpu": _CpuPrimitives, "gpu": _GpuPrimitives},
        "block_classes": {
            "cpu": {"transformer": TransformerBlockInference, "mamba1": Mamba1BlockInference,
                    "mamba2": Mamba2BlockInference},
            "gpu": {"transformer": TransformerBlock, "mamba1": Mamba1Block, "mamba2": Mamba2Block}},
        "generation_classes": (Array, CausalLMState,
                               {"transformer": TransformerState, "mamba1": Mamba1State,
                                "mamba2": Mamba2State}, GenerationStream),
    }


_NATIVE, _NATIVE_CONTEXT = _connect_causal_native()
