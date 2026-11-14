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
from .. import _backend
from .. import _numeric_profile
from .. import lowbit as _lowbit
from .._array import Array
from .._buffer import addr, addr_ro, empty
from .config import POSITION_CEILING, HFConfig, plan_for
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

# The device route ("auto" -> cpu/gpu by vendor), the per-route block classes,
# the integer-weight refusal and the last-position argmax run in Mojo
# (`_model_route`, `context["block_classes"]`, the checkpoint reader and the
# generation entries in bindings/causal_lm_construct_door.mojo and
# bindings/causal_lm_public_door.mojo); their dead Python twins `_route`,
# `_block_classes`, `_float_weight` and `_argmax_last` are gone (lane
# py-lm-neural, 2026-11-14).


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
        return _NATIVE.causal_state_init(self, (batch_size, max_tokens, layers, owner), _NATIVE_CONTEXT)

    def __repr__(self):
        return _NATIVE.causal_state_repr(self)


class CausalLM:
    """See the module header. Build with `load` (a checkpoint directory) or
    directly from a `ModelPlan` and a weight dict shaped like
    `_read_weights` returns."""

    def __init__(self, plan, weights, *, weight_format="float32", max_positions=None, device="auto",
                 numeric_profile=None):
        return _NATIVE.causal_model_init(self, plan, weights,
                                        (weight_format, max_positions, device, numeric_profile),
                                        _NATIVE_CONTEXT)

    # TOMBSTONE: CausalLM._make_blocks, _ids, _check_positions (dead code: no caller; the native
    # owner builds blocks and admits ids and positions) deleted 2026-11-09 by 57b86ba5ac; code
    # recoverable at 57b86ba5ac^.
    # Restore: git apply experiments/removed/dead_code_p5_20261109.patch

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
        return _NATIVE.causal_lm_blocks(self)

    def parameters(self):
        """Return tensors by checkpoint name."""
        return _NATIVE.causal_lm_parameters(self)

    def _run(self, ids, state, step, what):
        return _NATIVE.causal_lm_forward(self, ids, state, step)

    def forward(self, ids, state=None):
        """Return `(B, L, vocabulary)` logits for integer token ids `(B, L)`.

        A supplied state is updated in place. With no state, the sequence starts
        from fresh caches. In `identical` mode, mojolearn returns the same bits
        on Apple, NVIDIA and AMD GPUs.
        """
        return _NATIVE.causal_lm_forward(self, ids, state, False)

    __call__ = forward

    def allocate_state(self, batch_size, max_tokens):
        """Allocate fresh caches for `batch_size` sequences and `max_tokens` positions."""
        return _NATIVE.causal_lm_allocate(self, batch_size, max_tokens)

    def reset_state(self, state):
        """Reset this model's state, preserving its batch size and capacity."""
        return _NATIVE.causal_lm_reset(self, state)

    def step(self, ids, state):
        """Return `(B, vocabulary)` logits and update state for one token per row."""
        return _NATIVE.causal_lm_forward(self, ids, state, True)

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
        return _NATIVE.causal_lm_generate(
            self, ids, max_new_tokens,
            (greedy, seed, temperature, top_k, top_p, eos_token_id, pad_token_id))

    def stream(self, ids, max_new_tokens, *, greedy=True, seed=None,
               temperature=1.0, top_k=0, top_p=1.0, eos_token_id=None,
               pad_token_id=None):
        """Create a continuation stream; `read(n)` returns at most `n` new ids per row.

        Read size does not change sampled ids. `info()` reports completion and
        each row's generated length, including its EOS token.
        """
        return _NATIVE.causal_lm_stream(
            self, ids, max_new_tokens,
            (greedy, seed, temperature, top_k, top_p, eos_token_id, pad_token_id))

    def generate_text(self, text, max_new_tokens, *, tokenizer, greedy=True,
                      seed=None, temperature=1.0, top_k=0, top_p=1.0,
                      eos_token_id=None, pad_token_id=None):
        """Tokenize text and return its decoded continuation, excluding EOS tokens.

        Pass a `models.Tokenizer` loaded for this checkpoint. Sampling and
        stopping use the same options as `generate`.
        """
        return _NATIVE.causal_lm_generate_text(
            self, tokenizer, text, max_new_tokens,
            (greedy, seed, temperature, top_k, top_p, eos_token_id, pad_token_id))

    def stream_text(self, text, max_new_tokens, *, tokenizer, greedy=True,
                    seed=None, temperature=1.0, top_k=0, top_p=1.0,
                    eos_token_id=None, pad_token_id=None):
        """Create a text continuation stream; `read(n)` consumes at most `n` tokens.

        A read may return an empty string while a UTF-8 character is incomplete.
        Use `info()['done']` to detect completion.
        """
        return _NATIVE.causal_lm_text_stream(
            self, tokenizer, text, max_new_tokens,
            (greedy, seed, temperature, top_k, top_p, eos_token_id, pad_token_id))

    def resume_generation(self, checkpoint, *, tokenizer=None):
        """Resume a generation checkpoint on the same model weights and settings."""
        return _NATIVE.causal_lm_resume(self, checkpoint, tokenizer)

    def _generate_resident(self, ids, n_new, total, last_logits=None):
        # Retained diagnostic door; arithmetic and lifecycle are native.
        return _NATIVE.causal_lm_resident(self, ids, n_new, last_logits, None)

    def __repr__(self):
        return _NATIVE.causal_lm_repr(self)


class GenerationStream:
    """A native continuation with explicit bounded reads and resumable state."""

    def read(self, max_tokens=1):
        """Return the next token array or text fragment; stop at EOS or the limit."""
        return _NATIVE.causal_lm_read(self, max_tokens)

    def info(self):
        """Return completion, generated lengths and whether the stream remains usable."""
        return _NATIVE.causal_lm_info(self)

    def state_dict(self):
        """Return independent state bytes, sampling options and model fingerprint."""
        return _NATIVE.causal_lm_snapshot(self)


# Package-import connection work only. The native owner reads these plain
# metadata records; it never calls the package loaders or block constructors.
def _connect_causal_native():
    import _thread
    import hashlib  # Connection-time import; generation owns only the cached C SHA256 entry.
    from ..tokenizer import _binding as runtime_binding
    from .. import _training_impl as training
    from .._transformer_impl import TransformerBlock, TransformerState, TransformerDecodeSession, _OPT_NAMES
    from .._mamba_impl import (Mamba1Block, Mamba2Block, Mamba3Block, Mamba1State, Mamba2State, Mamba3State,
                               Mamba1DecodeSession, Mamba2DecodeSession, Mamba3DecodeSession)
    from ..neural_inference import (TransformerBlockInference, Mamba1BlockInference,
                                    Mamba2BlockInference, Mamba3BlockInference, _causal_native_connections)
    from .config import _NATIVE_CONTEXT as config_context
    from .safetensors import _NATIVE_CONTEXT as checkpoint_context
    from .._native_neural_bridge import connect_training
    from .._native_classical_bridge import _numpy_free_context
    from types import ModuleType
    admission_context = _numpy_free_context()
    admission_context["native_module_type"] = ModuleType
    runtime = runtime_binding()._module
    from .._native_operation_lock import connect as connect_operation_lock
    connect_operation_lock(runtime)
    runtime.native_codec_call("initialize", (Array, _lowbit.BF16Weight, _lowbit.Int8Weight))
    selected_mode = _backend.default_mode()
    bindings = {"cpu:identical": _causal_native_connections()}
    vendor = _backend.vendor()
    if vendor not in (None, "cpu"):
        # Mojo-owned block types permit one tier per process. Connect the
        # user's selected tier first; other tiers retain their normal admission
        # refusal instead of claiming the default before a fast constructor.
        modes = (selected_mode, "identical" if selected_mode == "fast" else "fast")
        for mode in modes:  # glue: connects available compiled modes during package import
            if mode == "fast" and vendor != "metal":
                continue
            entries = {}
            targets = (("transformer", "_mojolearn_transformer"),
                       ("mamba1", "_mojolearn_mamba"), ("mamba2", "_mojolearn_mamba"),
                       ("mamba3", "_mojolearn_mamba"),
                       ("primitives", "_mojolearn_training"), ("linalg", "_mojolearn_linalg"))
            for name, target in targets:  # glue: connects compiled sibling modules at package import
                try:
                    entries[name] = _backend.binding(target, mode)
                except (ImportError, RuntimeError, ValueError, OSError, AttributeError) as exc:
                    entries[name] = f"native binding {target} for {mode} is unavailable: {exc}"
            bindings["gpu:" + mode] = entries
    return runtime, {
        "admission_context": admission_context,
        "runtime": runtime, "backend": _backend, "selected_vendor": vendor, "model_class": CausalLM,
        "block_option_names": ("n_kv_heads", "head_dim", "window", "rope_theta", "rope_scaling",
                               "rope_dim", "max_positions", "qkv_bias", "o_bias", "norm", "norm_eps",
                               "norm_bias", "mlp", "mlp_bias", "qk_norm", "attn_softcap"),
        "bindings": bindings, "profiles": _numeric_profile.PROFILES,
        "profile_defaults": _numeric_profile._defaults,
        "rejected_profiles": _numeric_profile.REJECTED,
        "config_context": config_context, "checkpoint_context": checkpoint_context,
        "array_class": Array, "bf16_class": _lowbit.BF16Weight,
        "int8_class": _lowbit.Int8Weight, "projections": _PROJECTIONS,
        "optional_names": _OPT_NAMES, "rlock": _thread.RLock,
        "training_module": training,
        "training_bindings": connect_training(_backend, vendor, bindings),
        "primitive_classes": {"cpu": _CpuPrimitives, "gpu": _GpuPrimitives},
        "block_classes": {
            "cpu": {"transformer": TransformerBlockInference, "mamba1": Mamba1BlockInference,
                    "mamba2": Mamba2BlockInference, "mamba3": Mamba3BlockInference},
            "gpu": {"transformer": TransformerBlock, "mamba1": Mamba1Block,
                    "mamba2": Mamba2Block, "mamba3": Mamba3Block}},
        "mamba3_state_class": Mamba3State,
        "transformer_session_class": TransformerDecodeSession,
        "mamba_session_classes": {"mamba1": Mamba1DecodeSession, "mamba2": Mamba2DecodeSession,
                                  "mamba3": Mamba3DecodeSession},
        "generation_classes": (Array, CausalLMState,
                               {"transformer": TransformerState, "mamba1": Mamba1State,
                                "mamba2": Mamba2State}, GenerationStream),
    }


_NATIVE, _NATIVE_CONTEXT = _connect_causal_native()

from .. import _native_neural_bridge
_native_neural_bridge.RUNTIME = _NATIVE
_native_neural_bridge.CONTEXT = _NATIVE_CONTEXT
