# SPDX-License-Identifier: Apache-2.0
"""Shape configuration for the decoder language model trainer."""
from dataclasses import dataclass, FrozenInstanceError

_DEFAULT_PROFILE = 'mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1'


# Runtime dimensions, layer count and token vocabulary.
#
# The defaults preserve the published small model's registry and profile.
# Shape admission is not a performance or cross-vendor qualification.
# ``chunked_lm_head_v2=True`` selects the explicit IDENTICAL loss/gradient
# profile with bounded vocabulary panels. Logits prediction still returns
# the full vocabulary; its per-cell GEMM contraction is unchanged.
@dataclass(frozen=True, init=False, repr=False, eq=False)
class ByteLanguageModelConfig:
    """Dimensions of the decoder language model: batch, sequence, layers, vocabulary.

    An immutable description of the model and training-batch shape passed to
    :class:`~mojolearn.LanguageModelTrainer` as ``shape=``. Each layer has an
    attention block with grouped key/value heads and a gated MLP, each with
    its own norm weight; the model has a token embedding and a separate output
    head. The defaults describe the small published byte-level model
    (vocabulary 256). ``LanguageModelConfig`` is the same class.

    Parameters
    ----------
    batch : int, default 2
        Sequences per training step.
    length : int, default 32
        Tokens per sequence, at most 8192.
    d_model : int, default 32
        Model width; must equal ``n_heads * head_dim``.
    n_heads : int, default 4
        Attention heads; must be divisible by ``n_kv``.
    n_kv : int, default 2
        Key/value heads.
    head_dim : int, default 8
        Width of each head; must be even.
    intermediate : int, default 64
        MLP hidden width.
    n_layers : int, default 2
        Number of decoder layers.
    vocab_size : int, default 256
        Token vocabulary size (256 for bytes; ``corpus.n_vocab`` for a BPE
        corpus from :mod:`mojolearn.lm_corpus`).
    chunked_lm_head_v2 : bool, default False
        Compute the loss and its gradient over bounded vocabulary panels, which
        bounds memory for large vocabularies. Logits from prediction still
        cover the full vocabulary.

    Raises
    ------
    ValueError
        If a dimension is not an integer in [1, 2**20], the head constraints
        above are not met, or the shape exceeds signed 32-bit indexing.

    Examples
    --------
    >>> import mojolearn
    >>> shape = mojolearn.LanguageModelConfig(batch=1, length=128, d_model=64, n_heads=4,
    ...                                       n_kv=2, head_dim=16, intermediate=128,
    ...                                       n_layers=2, vocab_size=256)
    >>> shape.n_total > 0
    True
    """
    batch: int = 2
    length: int = 32
    d_model: int = 32
    n_heads: int = 4
    n_kv: int = 2
    head_dim: int = 8
    intermediate: int = 64
    n_layers: int = 2
    vocab_size: int = 256
    chunked_lm_head_v2: bool = False

    def __init__(self, batch=2, length=32, d_model=32, n_heads=4, n_kv=2,
                 head_dim=8, intermediate=64, n_layers=2, vocab_size=256,
                 chunked_lm_head_v2=False):
        return _CONFIG(self, "config_construct", (batch, length, d_model, n_heads, n_kv,
                       head_dim, intermediate, n_layers, vocab_size, chunked_lm_head_v2), _CONFIG_CONTEXT)

    def __post_init__(self):
        return _CONFIG(self, "config_validate", (), _CONFIG_CONTEXT)

    def __repr__(self):
        return _CONFIG(self, "config_repr", (), _CONFIG_CONTEXT)

    def __eq__(self, other):
        return _CONFIG(self, "config_equal", (other,), _CONFIG_CONTEXT)

    def __hash__(self):
        return _CONFIG(self, "config_hash", (), _CONFIG_CONTEXT)

    @property
    def native_shape(self):
        """The dimensions as a tuple of ints, in field order."""
        return _CONFIG(self, "config_native_shape", (), _CONFIG_CONTEXT)

    @property
    def profile(self):
        """The name of the training profile this shape selects."""
        return _CONFIG(self, "config_profile", (), _CONFIG_CONTEXT)

    @property
    def parameter_shapes(self):
        """The shape of every parameter tensor, in packing order."""
        return _CONFIG(self, "config_shapes", (), _CONFIG_CONTEXT)

    @property
    def n_tensors(self):
        """Number of parameter tensors: ``2 + 9 * n_layers``."""
        return _CONFIG(self, "config_n_tensors", (), _CONFIG_CONTEXT)

    @property
    def parameter_names(self):
        """The name of every parameter tensor, in packing order."""
        return _CONFIG(self, "config_names", (), _CONFIG_CONTEXT)

    @property
    def offsets(self):
        """Start offset of every parameter tensor in the flat parameter vector, plus the total."""
        return _CONFIG(self, "config_offsets", (), _CONFIG_CONTEXT)

    @property
    def n_total(self):
        """Total number of parameters."""
        return _CONFIG(self, "config_n_total", (), _CONFIG_CONTEXT)

    def to_dict(self):
        """The fields as a dict, suitable for ``ByteLanguageModelConfig(**d)``."""
        return _CONFIG(self, "config_dict", (), _CONFIG_CONTEXT)


def require_shape(value=None):
    return _CONFIG(value, "config_require", (), _CONFIG_CONTEXT)


def state_shape(state):
    return _CONFIG(state, "config_state", (), _CONFIG_CONTEXT)


def _config_setattr(self, name, value):
    return _CONFIG(self, "config_setattr", (name, value), _CONFIG_CONTEXT)


def _config_delattr(self, name):
    return _CONFIG(self, "config_delattr", (name,), _CONFIG_CONTEXT)


def _connect_config():
    # glue: dataclass storage stays compatible; native code owns public methods.
    from ._native_family_bridge import shared_context, _unavailable
    from .tokenizer import _binding
    context = dict(shared_context(), byte_lm_config_class=ByteLanguageModelConfig,
                   byte_lm_config_frozen_error=FrozenInstanceError, byte_lm_default_profile=_DEFAULT_PROFILE)
    try:
        entry = _binding()._module.byte_lm_trainer_call
    except (ImportError, AttributeError, OSError) as error:
        entry = _unavailable(error)
    return entry, context


_CONFIG, _CONFIG_CONTEXT = _connect_config()
ByteLanguageModelConfig.__setattr__ = _config_setattr
ByteLanguageModelConfig.__delattr__ = _config_delattr
