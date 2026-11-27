"""tools/metal_float64_lint.py: the 0.9.0 up_candidates case is caught, its fix passes, guards are honoured, and
the tree at HEAD passes with its allow list. File-only (no compile)."""
from pathlib import Path
import unittest

import metal_float64_lint as lint

ROOT = Path(__file__).resolve().parent.parent

# The chain of the 0.9.0 Apple failure, cut to its shape: sequence/exec_device.mojo launches seq_kernel, which calls
# sequence.dispatch.apply, which calls autograd.nn_conv_ops.apply_nn_conv -> op_nnc_up_vjp -> up_candidates.
EXEC = '''from sequence.dispatch import apply


def seq_kernel[OP: Int](
    p0: FP,
    n: Int64,
):
    """One thread per element."""
    var i = Int(block_idx.x)
    if i < Int(n):
        apply[OP](i, a)


def launch(ctx: DeviceContext):
    ctx.enqueue_function[seq_kernel[OP]](p0, n, grid_dim=1, block_dim=1)
'''
DISPATCH = '''from autograd.nn_conv_ops import OP_NNC_FIRST, OP_NNC_LAST, apply_nn_conv


def apply[OP: Int](t: Int, a: Args):
    comptime if OP == 1:
        op_one(t, a)
    elif OP >= OP_NNC_FIRST and OP <= OP_NNC_LAST:
        apply_nn_conv[OP](t, a)
'''
CONV_HEAD = '''def apply_nn_conv[OP: Int](t: Int, a: Args):
    comptime if OP == OP_NNC_UP_VJP:
        op_nnc_up_vjp(t, a)


def op_nnc_up_vjp(t: Int, a: Args):
    var r = up_candidates(t, 8, Float32(2))


'''
# up_candidates before f6a2ad489a (the Float64 range) and after it (Float32 bounds)
BEFORE = '''def up_candidates(i: Int, osize: Int, scale: Float32) -> Tuple[Int, Int]:
    """The candidate range."""
    if not (scale > Float32(0)):
        return (0, osize)
    var inv = Float64(1) / Float64(scale)
    var lo = Int(Float64(i - 1) * inv) - 2
    var hi = Int(Float64(i + 2) * inv) + 2
    return (lo, hi)
'''
AFTER = '''def up_candidates(i: Int, osize: Int, scale: Float32) -> Tuple[Int, Int]:
    """The candidate range; no Float64 here."""
    if not (scale > Float32(0)):
        return (0, osize)
    var margin = 2 + (osize >> 20)
    var top = Float32(osize)
    var flo = min(max(Float32(i - 1) / scale, Float32(0)), top)
    var fhi = min(max(Float32(i + 2) / scale, Float32(0)), top)
    return (Int(flo) - margin, Int(fhi) + margin)
'''


def tree(conv_tail, extra=()):
    return [("sequence/exec_device.mojo", EXEC), ("sequence/dispatch.mojo", DISPATCH),
            ("autograd/nn_conv_ops.mojo", CONV_HEAD + conv_tail), *extra]


class MetalFloat64Lint(unittest.TestCase):
    def test_up_candidates_before_the_fix_is_caught(self):
        found, roots = lint.scan(tree(BEFORE), jobs=1)
        self.assertEqual(roots, 1)
        self.assertEqual({(p, f) for p, _, f in found}, {("autograd/nn_conv_ops.mojo", "up_candidates")})
        self.assertEqual(len(found), 3)

    def test_up_candidates_after_the_fix_passes(self):
        found, _ = lint.scan(tree(AFTER), jobs=1)
        self.assertEqual(found, [])

    def test_unreached_float64_is_not_flagged(self):
        host = ("core/host_only.mojo", "def host_mean(x: Float64) -> Float64:\n    return x * Float64(0.5)\n")
        found, _ = lint.scan(tree(AFTER, [host]), jobs=1)
        self.assertEqual(found, [])

    def test_non_apple_guards_are_honoured(self):
        guarded = AFTER.replace('    var margin = 2 + (osize >> 20)\n',
                                '    comptime if not TARGET_HAS_APPLE_GPU:\n        var w = Float64(scale)\n'
                                '    comptime if device_has_float64():\n        var v = Float64(scale)\n'
                                '    comptime if TARGET_HAS_APPLE_GPU:\n        var u = Float32(scale)\n'
                                '    else:\n        var z = Float64(scale)\n'
                                '    var margin = 2 + (osize >> 20)\n')
        found, _ = lint.scan(tree(guarded), jobs=1)
        self.assertEqual(found, [])
        alias = ("autograd/flags.mojo", "comptime WIDE_OK = not TARGET_HAS_APPLE_GPU\n")
        aliased = AFTER.replace('    var margin = 2 + (osize >> 20)\n',
                                '    comptime if WIDE_OK:\n        var w = Float64(scale)\n'
                                '    var margin = 2 + (osize >> 20)\n')
        found, _ = lint.scan(tree(aliased, [alias]), jobs=1)
        self.assertEqual(found, [])
        apple = AFTER.replace('    var margin = 2 + (osize >> 20)\n',
                              '    comptime if TARGET_HAS_APPLE_GPU:\n        var w = Float64(scale)\n'
                              '    var margin = 2 + (osize >> 20)\n')
        found, _ = lint.scan(tree(apple), jobs=1)
        self.assertEqual(len(found), 1)

    def test_float64_argument_of_a_reached_function_is_flagged(self):
        sig = AFTER.replace("def up_candidates(i: Int, osize: Int, scale: Float32)",
                            "def up_candidates(\n    i: Int, osize: Int, scale: Float64,\n)")
        found, _ = lint.scan(tree(sig), jobs=1)
        self.assertEqual(len(found), 1)

    def test_allow_list_judging(self):
        found = [("a.mojo", 3, "f"), ("b.mojo", 9, "g")]
        allow, bad = lint.read_allow("a.mojo:f  # SAFE-CONST: literal\nc.mojo:h  # SAFE-HOST: gone\nx.mojo:y  # WHY: no\n")
        self.assertEqual(len(bad), 1)
        unlisted, stale = lint.judge(found, allow)
        self.assertEqual(unlisted, [("b.mojo", 9, "g")])
        self.assertEqual(stale, [("c.mojo", "h")])

    def test_head_passes(self):
        self.assertEqual(lint.main(["--root", str(ROOT), "--jobs", "2"]), 0)


if __name__ == "__main__":
    unittest.main()
