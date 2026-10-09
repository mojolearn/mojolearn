# SPDX-License-Identifier: Apache-2.0
"""Host-only runtime shape for the configured decoder language model."""
from dataclasses import dataclass, fields
import operator

_DEFAULT_PROFILE = 'mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1'


@dataclass(frozen=True)
class ByteLanguageModelConfig:
    """Runtime dimensions, layer count and token vocabulary.

    The defaults preserve the published small model's registry and profile.
    Shape admission is not a performance or cross-vendor qualification.
    ``chunked_lm_head_v2=True`` selects the explicit IDENTICAL loss/gradient
    profile with bounded vocabulary panels. Logits prediction still returns
    the full vocabulary; its per-cell GEMM contraction is unchanged.
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

    def __post_init__(self):
        for field in fields(self):  # glue: validates config dataclass fields
            value = getattr(self, field.name)
            if field.name == 'chunked_lm_head_v2':
                if not isinstance(value, bool):
                    raise ValueError('chunked_lm_head_v2 must be a boolean')
                continue
            if isinstance(value, bool) or type(value).__name__ in ('bool', 'bool_'):
                raise ValueError('Byte-LM shape requires integer dimensions')
            try:
                value = operator.index(value)
            except TypeError as exc:
                raise ValueError('Byte-LM shape requires integer dimensions') from exc
            if not 0 < value <= 1 << 20:
                raise ValueError('Byte-LM dimensions must be in [1, 2**20]')
            object.__setattr__(self, field.name, value)
        if (self.length > 8192 or self.d_model != self.n_heads * self.head_dim
                or self.n_heads % self.n_kv or self.head_dim % 2):
            raise ValueError('Byte-LM requires L <= 8192, DM = H*HD, H divisible by KV, and even HD')
        if self.chunked_lm_head_v2 and self.vocab_size < 2:
            raise ValueError('chunked_lm_head_v2 requires vocabulary size >= 2')
        b, l, dm, h, kv, hd, ff, layers, vocab = self.native_shape[:9]
        spans = (self.n_total, b * l * vocab, b * l * dm, b * l * ff,
                 b * h * l * l, b * kv * l * hd, b * (l + 1))
        if max(spans) > 2147483647:  # glue: max over seven shape spans
            raise ValueError('Byte-LM shape exceeds signed 32-bit indexing')

    @property
    def native_shape(self):
        dimensions = tuple(getattr(self, field.name) for field in fields(self)  # glue: dataclass shape fields only
                           if field.name != 'chunked_lm_head_v2')  # glue: dataclass shape fields only
        # Preserve the existing nine-integer ABI when the opt-in is absent.
        return dimensions + (1,) if self.chunked_lm_head_v2 else dimensions

    @property
    def profile(self):
        if self.native_shape == (2, 32, 32, 4, 2, 8, 64, 2, 256):
            return _DEFAULT_PROFILE
        b, l, dm, h, kv, hd, ff, layers, vocab = self.native_shape[:9]
        suffix = '-v256-blocks2.fp32.v2' if (layers, vocab) == (2, 256) else f'-v{vocab}-blocks{layers}.fp32.v3'
        head = '-lmhead-chunk256-v2' if self.chunked_lm_head_v2 else ''
        return (f'mojolearn.byte-lm.b{b}-l{l}-d{dm}-h{h}-kv{kv}-hd{hd}'
                f'-ff{ff}{suffix}{head}')

    @property
    def parameter_shapes(self):
        dm, kd, ff = self.d_model, self.n_kv * self.head_dim, self.intermediate
        block = ((dm,), (dm, dm), (kd, dm), (kd, dm), (dm, dm),
                 (dm,), (ff, dm), (ff, dm), (dm, ff))
        return ((self.vocab_size, dm), *(block * self.n_layers), (self.vocab_size, dm))

    @property
    def n_tensors(self):
        return 2 + 9 * self.n_layers

    @property
    def parameter_names(self):
        names = ('norm1_w', 'w_q', 'w_k', 'w_v', 'w_o', 'norm2_w', 'w_gate', 'w_up', 'w_down')
        return ('embed', *(f'block{layer}.{name}' for layer in range(self.n_layers) for name in names), 'lm_head')  # glue: parameter names per layer

    @property
    def offsets(self):
        from ._portable_math import prod
        offsets = [0]
        for shape in self.parameter_shapes:  # glue: offsets over parameter shapes
            offsets.append(offsets[-1] + prod(shape))
        return tuple(offsets)

    @property
    def n_total(self):
        return self.offsets[-1]

    def to_dict(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}  # glue: config dataclass field values


def require_shape(value=None):
    if value is None:
        return ByteLanguageModelConfig()
    if not isinstance(value, ByteLanguageModelConfig):
        raise TypeError('shape must be a ByteLanguageModelConfig')
    return value


def state_shape(state):
    if 'model_shape' not in state:
        return ByteLanguageModelConfig()
    value = state['model_shape']
    expected = {field.name for field in fields(ByteLanguageModelConfig)}  # glue: config dataclass field names
    accepted = (expected, expected - {'chunked_lm_head_v2'},
                expected - {'n_layers', 'vocab_size'},
                expected - {'n_layers', 'vocab_size', 'chunked_lm_head_v2'})
    if not isinstance(value, dict) or set(value) not in accepted:
        raise ValueError('Byte-LM model_shape has missing or unknown dimensions')
    return ByteLanguageModelConfig(**value)
