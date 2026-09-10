# SPDX-License-Identifier: Apache-2.0
"""Host-only runtime shape for the two-block, 256-symbol byte model."""
from dataclasses import dataclass, fields
import operator

_DEFAULT_PROFILE = 'mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1'


@dataclass(frozen=True)
class ByteLanguageModelConfig:
    """Runtime dimensions; the architecture retains two blocks and 256 bytes.

    The defaults preserve the published small model's registry and profile.
    Shape admission is not a performance or cross-vendor qualification.
    """
    batch: int = 2
    length: int = 32
    d_model: int = 32
    n_heads: int = 4
    n_kv: int = 2
    head_dim: int = 8
    intermediate: int = 64

    def __post_init__(self):
        for field in fields(self):
            value = getattr(self, field.name)
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
        b, l, dm, h, kv, hd, ff = self.native_shape
        spans = (self.n_total, b * l * 256, b * l * dm, b * l * ff,
                 b * h * l * l, b * kv * l * hd, b * (l + 1))
        if max(spans) > 2147483647:
            raise ValueError('Byte-LM shape exceeds signed 32-bit indexing')

    @property
    def native_shape(self):
        return tuple(getattr(self, field.name) for field in fields(self))

    @property
    def profile(self):
        if self.native_shape == (2, 32, 32, 4, 2, 8, 64):
            return _DEFAULT_PROFILE
        b, l, dm, h, kv, hd, ff = self.native_shape
        return (f'mojolearn.byte-lm.b{b}-l{l}-d{dm}-h{h}-kv{kv}-hd{hd}'
                f'-ff{ff}-v256-blocks2.fp32.v2')

    @property
    def parameter_shapes(self):
        dm, kd, ff = self.d_model, self.n_kv * self.head_dim, self.intermediate
        block = ((dm,), (dm, dm), (kd, dm), (kd, dm), (dm, dm),
                 (dm,), (ff, dm), (ff, dm), (dm, ff))
        return ((256, dm), *block, *block, (256, dm))

    @property
    def offsets(self):
        from math import prod
        offsets = [0]
        for shape in self.parameter_shapes:
            offsets.append(offsets[-1] + prod(shape))
        return tuple(offsets)

    @property
    def n_total(self):
        return self.offsets[-1]

    def to_dict(self):
        return {field.name: getattr(self, field.name) for field in fields(self)}


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
    if not isinstance(value, dict) or set(value) != {field.name for field in fields(ByteLanguageModelConfig)}:
        raise ValueError('Byte-LM model_shape has missing or unknown dimensions')
    return ByteLanguageModelConfig(**value)
