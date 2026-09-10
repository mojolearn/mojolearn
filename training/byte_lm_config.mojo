# SPDX-License-Identifier: Apache-2.0
"""Host-only shape and registry for the two-block, 256-symbol byte LM."""

comptime BYTE_CONFIG_LIMIT = 2147483647
comptime BYTE_DEFAULT_PROFILE = "mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1"


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

    def __init__(out self, batch: Int = 2, length: Int = 32,
                 d_model: Int = 32, n_heads: Int = 4, n_kv: Int = 2,
                 head_dim: Int = 8, intermediate: Int = 64):
        self.batch = batch
        self.length = length
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_kv = n_kv
        self.head_dim = head_dim
        self.intermediate = intermediate

    def validate(self) raises:
        var fields: List[Int] = [self.batch, self.length, self.d_model,
            self.n_heads, self.n_kv, self.head_dim, self.intermediate]
        for value in fields:
            if value <= 0 or value > 1048576:
                raise Error("byte LM: shape fields must be in [1,1048576]")
        if self.length > 8192:
            raise Error("byte LM: length exceeds RoPE table limit 8192")
        if self.d_model != _byte_product(self.n_heads, self.head_dim):
            raise Error("byte LM: d_model must equal n_heads*head_dim")
        if self.n_heads % self.n_kv != 0:
            raise Error("byte LM: n_heads must be divisible by n_kv")
        if self.head_dim % 2 != 0:
            raise Error("byte LM: head_dim must be even")
        var m = _byte_product(self.batch, self.length)
        _ = _byte_product(self.batch, self.length + 1)
        _ = _byte_product(m, 256)
        _ = _byte_product(m, self.d_model)
        _ = _byte_product(m, self.intermediate)
        _ = _byte_product(_byte_product(m, self.n_heads), self.length)
        _ = _byte_product(m, _byte_product(self.n_kv, self.head_dim))
        var total = 0
        for j in range(20):
            var n = self._param_count(j)
            if total > BYTE_CONFIG_LIMIT - n:
                raise Error("byte LM: parameter registry exceeds int32 indexing")
            total += n

    def _param_count(self, j: Int) raises -> Int:
        if j < 0 or j >= 20:
            raise Error("byte LM: parameter index out of range")
        if j == 0 or j == 19:
            return _byte_product(256, self.d_model)
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
        for j in range(20):
            total += self._param_count(j)
        return total

    def offsets(self) raises -> List[Int]:
        self.validate()
        var result: List[Int] = [0]
        var total = 0
        for j in range(20):
            total += self._param_count(j)
            result.append(total)
        return result^

    def profile(self) raises -> String:
        self.validate()
        if (self.batch == 2 and self.length == 32 and self.d_model == 32
            and self.n_heads == 4 and self.n_kv == 2 and self.head_dim == 8
            and self.intermediate == 64):
            return String(BYTE_DEFAULT_PROFILE)
        return (String("mojolearn.byte-lm.b") + String(self.batch)
            + "-l" + String(self.length) + "-d" + String(self.d_model)
            + "-h" + String(self.n_heads) + "-kv" + String(self.n_kv)
            + "-hd" + String(self.head_dim) + "-ff" + String(self.intermediate)
            + "-v256-blocks2.fp32.v2")
