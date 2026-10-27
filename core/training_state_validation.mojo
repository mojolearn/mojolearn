# SPDX-License-Identifier: Apache-2.0
# State admission only: the finite f32 owner was already copied/validated at
# the shared buffer boundary. No optimizer computation changes here.


def _state_product(a: Int, b: Int) raises -> Int:
    if a < 0 or b < 0 or (b > 0 and a > 2147483647 // b):
        raise Error("Byte-LM shape exceeds signed 32-bit indexing")
    return a * b


def training_byte_shape_validate(d: List[Int], chunked: Bool) raises:
    # Same dimension contract as ByteLanguageModelConfig.__post_init__, without
    # selecting the numeric mode of this host boundary's compilation unit.
    if d[1] > 8192 or d[2] != d[3] * d[5] or d[3] % d[4] != 0 or d[5] % 2 != 0:
        raise Error("Byte-LM requires L <= 8192, DM = H*HD, H divisible by KV, and even HD")
    if chunked and d[8] < 2:
        raise Error("chunked_lm_head_v2 requires vocabulary size >= 2")
    var rows = _state_product(d[0], d[1])
    for width in [d[8], d[2], d[6]]:  # small-loop(the three activation widths)
        _ = _state_product(rows, width)
    _ = _state_product(_state_product(rows, d[3]), d[1])
    _ = _state_product(rows, _state_product(d[4], d[5]))
    _ = _state_product(d[0], d[1] + 1)
    var block = 2 * d[2] + 2 * _state_product(d[2], d[2])
    block += 2 * _state_product(_state_product(d[4], d[5]), d[2])
    block += 3 * _state_product(d[6], d[2])
    var total = _state_product(d[7], block) + 2 * _state_product(d[8], d[2])
    if total > 2147483647:
        raise Error("Byte-LM shape exceeds signed 32-bit indexing")


def training_admit_moments_and_flags(variance: Int, count: Int, flags: Int, flag_count: Int) raises:
    var v = MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=variance)
    var f = MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=flags)
    for i in range(count):  # cpu-route: native checkpoint/state validation
        if v[i] < 0:
            raise Error("Byte-LM requires nonnegative second moments and binary flags")
    for i in range(flag_count):  # cpu-route: native checkpoint/state validation
        if f[i] < 0 or f[i] > 1:
            raise Error("Byte-LM requires nonnegative second moments and binary flags")
