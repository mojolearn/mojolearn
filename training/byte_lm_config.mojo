# SPDX-License-Identifier: Apache-2.0
"""Host-only shape and registry for the configured decoder language model."""

comptime BYTE_CONFIG_LIMIT = 2147483647
comptime BYTE_DEFAULT_PROFILE = "mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1"

comptime BYTE_MAX_ABS_POSITION = 8192
"""Contract section 3 and DEVIATION 812: the Cody-Waite domain of
`_cephes_sincosf_core`, shared by `portable_sinf` and `portable_cosf`.

NOT A TABLE SIZE. Nothing is tabulated at 8192 entries; this is the largest
absolute position whose sine and cosine the portable kernels can argument-reduce
without losing the bits that identity depends on. The guard below used to call
it a "RoPE table limit", which sent readers hunting for a table that is not the
constraint. The genuine table bounds live elsewhere and are a different check:
`byte_lm_host.mojo` and `byte_lm_host_kernels.mojo` refuse a length past the
rotary table the RUNNING model was actually built for, which is a configured
size, not this ceiling.

Duplicated deliberately rather than imported from
`transformer/impl/llama/modeling_llama.mojo`, whose `MAX_ABS_POSITION` is the
same 8192: that module carries GPU imports and this one is host-only by
contract. `transformer/checks/transformer_fixture.mojo` duplicates it for the
same reason. STRICTLY GREATER THAN refuses, so a length of exactly 8192 is
legal, matching llama."""


def _byte_product(a: Int, b: Int) raises -> Int:
    if a < 0 or b < 0 or (b > 0 and a > BYTE_CONFIG_LIMIT // b):
        raise Error("byte LM: shape product exceeds int32 indexing")
    return a * b


struct ByteConfig(Copyable, Movable):
    var batch: Int
    var length: Int
    var d_model: Int
    var n_heads: Int
    var n_kv: Int
    var head_dim: Int
    var intermediate: Int
    var n_layers: Int
    var vocab_size: Int

    def __init__(out self, batch: Int = 2, length: Int = 32,
                 d_model: Int = 32, n_heads: Int = 4, n_kv: Int = 2,
                 head_dim: Int = 8, intermediate: Int = 64,
                 n_layers: Int = 2, vocab_size: Int = 256):
        self.batch = batch
        self.length = length
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv = n_kv
        self.head_dim = head_dim
        self.intermediate = intermediate
        self.n_layers = n_layers
        self.vocab_size = vocab_size

    def validate(self) raises:
        var fields: List[Int] = [self.batch, self.length, self.d_model,
            self.n_heads, self.n_kv, self.head_dim, self.intermediate, self.n_layers, self.vocab_size]
        for value in fields:
            if value <= 0 or value > 1048576:
                raise Error("byte LM: shape fields must be in [1,1048576]")
        if self.length > BYTE_MAX_ABS_POSITION:
            raise Error(
                String("byte LM: length ")
                + String(self.length)
                + " exceeds the absolute-position ceiling "
                + String(BYTE_MAX_ABS_POSITION)
                + " (DEVIATION 812: the Cody-Waite domain of"
                + " _cephes_sincosf_core, shared by portable_sinf and"
                + " portable_cosf)"
            )
        if self.d_model != _byte_product(self.n_heads, self.head_dim):
            raise Error("byte LM: d_model must equal n_heads*head_dim")
        if self.n_heads % self.n_kv != 0:
            raise Error("byte LM: n_heads must be divisible by n_kv")
        if self.head_dim % 2 != 0:
            raise Error("byte LM: head_dim must be even")
        var m = _byte_product(self.batch, self.length)
        _ = _byte_product(self.batch, self.length + 1)
        _ = _byte_product(m, self.vocab_size)
        _ = _byte_product(m, self.d_model)
        _ = _byte_product(m, self.intermediate)
        _ = _byte_product(_byte_product(m, self.n_heads), self.length)
        _ = _byte_product(m, _byte_product(self.n_kv, self.head_dim))
        var total = 0
        for j in range(self.n_tensors()):
            var n = self._param_count(j)
            if total > BYTE_CONFIG_LIMIT - n:
                raise Error("byte LM: parameter registry exceeds int32 indexing")
            total += n

    def n_tensors(self) -> Int:
        return 2 + 9 * self.n_layers

    def _param_count(self, j: Int) raises -> Int:
        if j < 0 or j >= self.n_tensors():
            raise Error("byte LM: parameter index out of range")
        if j == 0 or j == self.n_tensors() - 1:
            return _byte_product(self.vocab_size, self.d_model)
        var local = (j - 1) % 9
        if local == 0 or local == 5:
            return self.d_model
        if local == 1 or local == 4:
            return _byte_product(self.d_model, self.d_model)
        if local == 2 or local == 3:
            return _byte_product(self.d_model, _byte_product(self.n_kv, self.head_dim))
        return _byte_product(self.d_model, self.intermediate)

    def param_count(self, j: Int) raises -> Int:
        self.validate()
        return self._param_count(j)

    def n_total(self) raises -> Int:
        self.validate()
        var total = 0
        for j in range(self.n_tensors()):
            total += self._param_count(j)
        return total

    def offsets(self) raises -> List[Int]:
        self.validate()
        var result: List[Int] = [0]
        var total = 0
        for j in range(self.n_tensors()):
            total += self._param_count(j)
            result.append(total)
        return result^

    def profile(self) raises -> String:
        self.validate()
        if (self.batch == 2 and self.length == 32 and self.d_model == 32
            and self.n_heads == 4 and self.n_kv == 2 and self.head_dim == 8
            and self.intermediate == 64 and self.n_layers == 2 and self.vocab_size == 256):
            return String(BYTE_DEFAULT_PROFILE)
        var suffix = String("-v256-blocks2.fp32.v2")
        if self.n_layers != 2 or self.vocab_size != 256:
            suffix = String("-v") + String(self.vocab_size) + "-blocks" + String(self.n_layers) + ".fp32.v3"
        return (String("mojolearn.byte-lm.b") + String(self.batch)
            + "-l" + String(self.length) + "-d" + String(self.d_model)
            + "-h" + String(self.n_heads) + "-kv" + String(self.n_kv)
            + "-hd" + String(self.head_dim) + "-ff" + String(self.intermediate)
            + suffix)
