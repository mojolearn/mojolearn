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
from .._array import Array as _Array
from .._buffer import addr as _addr, addr_ro as _addr_ro, empty as _empty

__all__ = ["CausalLM", "CausalLMState", "GenerationStream"]

#: The block-dict keys that are projection matrices, per kind: the tensors a
#: low-bit format packs. Everything else in a block dict stays float32.
_PROJECTIONS = {
    "transformer": ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                    "gate_proj.weight", "up_proj.weight", "down_proj.weight"),
    "mamba1": ("in_proj.weight", "x_proj.weight", "dt_proj.weight", "out_proj.weight"),
    "mamba2": ("in_proj.weight", "out_proj.weight"),
}

# The device route (None or "auto" -> cpu/gpu by vendor), the per-route block classes,
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
        y = _empty((n, d), "<f4")
        self._ext.embedding_forward(
            [_addr(y, name="x"), _addr_ro(table, name="embed"), _addr_ro(ids_flat, name="ids")], [n, v, d])
        return y

    def rms_norm(self, x2d, weight, eps):
        n, d = int(x2d.shape[0]), int(x2d.shape[1])
        y = _empty((n, d), "<f4")
        self._ext.rms_norm_forward(
            [_addr(y, name="hn"), _addr_ro(x2d, name="x"), _addr_ro(weight, name="norm")], [n, d, float(eps)])
        return y

    def linear(self, x2d, weight):
        n, d = int(x2d.shape[0]), int(x2d.shape[1])
        v = int(weight.shape[0])
        y = _empty((n, v), "<f4")
        self._ext.linear_forward(
            [_addr(y, name="logits"), _addr_ro(x2d, name="hn"), _addr_ro(weight, name="head")], [n, v, d])
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

    # The model takes `device=`; `check` runs the same model on the other
    # device in this process (`_check_both.on_device`). Its twin is built by
    # the compiled constructor from this model's own tensors (a CausalLM has
    # no `get_params`), and there is no fitted state beside them to hand over.
    _mojolearn_device = True
    _mojolearn_readback_plan = False  # vendor_used answers from the admitted block image
    _mojolearn_device_state = None

    def _mojolearn_device_twin(self, device):
        return _NATIVE.causal_model_twin(self, device, _NATIVE_CONTEXT)

    def __init__(self, plan, weights, *, weight_format="float32", max_positions=None, device=None,
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
    def load(cls, path, *, weight_format="float32", max_positions=None, device=None, numeric_profile=None):
        """A checkpoint directory (`config.json` plus `model.safetensors` or
        `model.safetensors.index.json` and its shards).

        `device` is None (this installation's device), ``"gpu"`` or ``"cpu"``;
        ``"auto"`` means the same as None."""
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
        """An iterator over the weight arrays, in checkpoint order, as ``torch.nn.Module.parameters``."""
        return _NATIVE.causal_lm_parameter_list(self)

    def named_parameters(self):
        """An iterator over ``(checkpoint name, array)`` pairs.

        ``dict(model.named_parameters())`` gives the weights by checkpoint name.
        """
        return _NATIVE.causal_lm_named_parameters(self)

    def state_dict(self):
        """The weights as a dict from checkpoint name to array, as ``torch.nn.Module.state_dict``.

        The arrays are copies: editing them does not change the model, and
        taking a state dict does not slow later calls. Use `parameters()` for
        the model's own arrays."""
        return _NATIVE.causal_lm_state_dict(self)

    @property
    def digest_(self):
        """The hash of this model's weights, as one lower-case hex string.

        In `identical` mode, Mojolearn returns the same bits on Apple, NVIDIA
        and AMD GPUs. The same checkpoint loaded on any of those machines, or
        on the CPU, has the same `digest_` within one Mojolearn version. The
        digest covers the model kind and its numeric profile, the embedding,
        the final norm, the head and every block's options and weights. It is
        computed each time it is read, from the model as it is at that moment.

        Raises
        ------
        AttributeError
            If the model runs in `fast` mode (Apple only), which promises no
            identical bits.
        """
        return _NATIVE.causal_lm_digest(self)

    def load_state_dict(self, state_dict):
        """Replace the weights with ``state_dict``, a dict from checkpoint name to array.

        The names must be exactly those of `state_dict()` and every array must
        have the shape of the weight it replaces. The load is all or nothing: a
        refused load leaves the model unchanged. Arrays taken earlier from
        `state_dict()` or `parameters()` keep the previous values."""
        return _NATIVE.causal_model_load_state_dict(self, state_dict, _NATIVE_CONTEXT)

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
                 pad_token_id=None, resident=False):
        """Return the prompt and up to `max_new_tokens` continuation ids.

        Greedy ties choose the lowest id. Sampling (`greedy=False`) requires
        an explicit unsigned 64-bit seed. Temperature applies before top-k,
        then top-p. `eos_token_id` accepts one id or a sequence of ids. Finished
        batch rows are padded; decoding ends when every row finishes or reaches
        the token limit. With no EOS, the output has exactly `(B, L + max_new_tokens)`
        ids. A missing pad id uses the first EOS id, or zero.

        `resident=True` keeps every layer's weights and decode session on the
        GPU between calls and decodes the whole continuation in one device
        pass. It returns the same ids as the default. It supports greedy
        decoding without `eos_token_id` on a GPU model and raises `ValueError`
        otherwise.

        Example::

            ids = model.generate(prompt, 32, resident=True)
        """
        return _NATIVE.causal_lm_generate(
            self, ids, max_new_tokens,
            (greedy, seed, temperature, top_k, top_p, eos_token_id, pad_token_id), resident)

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

    def check(self, ids, *, method=None):
        """Run this model on the GPU and on the CPU, print both sides' reports
        and whether their answers are the same bits.

        In `identical` mode, Mojolearn returns the same bits on Apple, NVIDIA
        and AMD GPUs, and the CPU binaries are held to the same bits. This
        answers `ids` twice on this machine: with this model, and with the
        same model on the other device, built in this process from the same
        weights. It runs in `identical` mode only. When nothing could be
        compared (for example on an installation with no GPU) it prints why.

        Parameters
        ----------
        ids : array-like
            Integer token ids of shape `(batch, length)`.
        method : str, optional
            The method to check. Default: `forward`.

        Returns
        -------
        dict
            The same result as `mojolearn.check(model, ids)`: `ran`, `sides`
            (one record per side) and `verdict["verdict"]`, which is
            `'MATCH'` or `'MISMATCH'`; `ran` is False, with a `reason`, when
            nothing was compared.

        Example::

            result = model.check(ids)
        """
        from .._verify_check import check, format_check
        result = check(self, ids, method=method)
        print(format_check(result))
        return result

    def vendor_used(self):
        """The backend this model runs on.

        Returns
        -------
        str
            ``'metal'``, ``'cuda'`` or ``'hip'``, or ``'cpu'`` for a model
            constructed with ``device="cpu"`` or on a CPU-only installation.
        """
        return _NATIVE.causal_model_vendor_used(self)

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
    runtime.native_codec_call("initialize", (_Array, _lowbit.BF16Weight, _lowbit.Int8Weight))
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
    context = {
        "admission_context": admission_context,
        "runtime": runtime, "backend": _backend, "selected_vendor": vendor, "model_class": CausalLM,
        "block_option_names": ("n_kv_heads", "head_dim", "window", "rope_theta", "rope_scaling",
                               "rope_dim", "max_positions", "qkv_bias", "o_bias", "norm", "norm_eps",
                               "norm_bias", "mlp", "mlp_bias", "qk_norm", "attn_softcap"),
        "bindings": bindings, "profiles": _numeric_profile.PROFILES,
        "profile_defaults": _numeric_profile._defaults,
        "rejected_profiles": _numeric_profile.REJECTED,
        "config_context": config_context, "checkpoint_context": checkpoint_context,
        "array_class": _Array, "bf16_class": _lowbit.BF16Weight,
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
        "generation_classes": (_Array, CausalLMState,
                               {"transformer": TransformerState, "mamba1": Mamba1State,
                                "mamba2": Mamba2State}, GenerationStream),
    }
    if vendor not in (None, "cpu"):
        # device="cpu" blocks beside the GPU set train on the host images,
        # loaded at the first CPU admission, never at import.
        from .._native_family_bridge import cpu_load_later
        from .._native_neural_bridge import connect_training_cpu
        cpu_load_later(context, lambda: connect_training_cpu(_backend, context))
    return runtime, context


_NATIVE, _NATIVE_CONTEXT = _connect_causal_native()

from .. import _native_neural_bridge
_native_neural_bridge.RUNTIME = _NATIVE
_native_neural_bridge.CONTEXT = _NATIVE_CONTEXT
