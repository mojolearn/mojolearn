# SPDX-License-Identifier: Apache-2.0
from std.testing import assert_equal
from bindings.generation_policy import generation_pick


def main() raises:
    # Independent integer-only SplitMix64 oracle, retained as literal ids;
    # equal logits make the categorical masses exactly 1/4.
    var uniform: List[Float32] = [0, 0, 0, 0]
    var expected: List[Int] = [3, 1, 0, 3, 0, 1, 0, 3]
    for i in range(len(expected)):
        assert_equal(generation_pick(uniform, False, 1, 0, 1, 0, 0, i), expected[i])
    assert_equal(generation_pick(uniform, True, 1, 0, 1, 0, 0, 0), 0)
    # Top-k ties retain smaller token ids; top-p includes the boundary token.
    for i in range(32):
        assert_equal(generation_pick(uniform, False, 1, 1, 1, 17, 2, i), 0)
        assert_equal(generation_pick(uniform, False, 1, 0, 0.25, 17, 2, i), 0)
    var logits: List[Float32] = [-100, 0, -100, -100]
    assert_equal(generation_pick(logits, False, 1, 0, 1, 7, 0, 0), 1)
    print("generation-policy: 74 assertions passed")
