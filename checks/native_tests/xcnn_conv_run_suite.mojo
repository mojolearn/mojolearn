# SPDX-License-Identifier: Apache-2.0
"""The Conv2d / Conv1d / ReLU / BasicBlock RUN half (`x_cnn/conv_run.mojo`),
reached without Python (lane doorsplit-cnn-xseq, 2026-11-11). One case per
group of RUN functions: literal words, raw float32 copies and the layers' own
refusal texts. The native entries the RUN calls are covered by
`xcnn_entry_table_suite.mojo`.

The cases are independent units and run across the native host worker pool.
"""
from core.host_parallel import host_parallelize
from x_cnn.conv_run import (
    ConvGeom, conv_run_check_config, conv_run_pad_code, conv_run_check_padding_mode, conv_run_text_pads,
    conv_run_size_pads, conv_run_explicit, conv_run_conv_pad, conv_run_fan, conv_run_bound, conv_run_same_shape,
    conv_run_words, conv_run_padded, conv_run_out, conv_run_check_rank, conv_run_check_channels, conv_run_check_grad,
    conv_run_check_weight, conv_run_pad_params, conv_run_mw, conv_run_group_prm, conv_run_row_width,
    conv_run_tail_size, conv_run_spatial, conv_run_check_ncl, conv_run_check_ncl_grad, conv_run_four, conv_run_three,
    conv_run_check_relu, conv_run_check_block_grad, conv_run_check_identity, conv_run_take_channels,
    conv_run_put_channels, conv_run_copy_words,
)

comptime XCNN_CONV_RUN_CASES = 16


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _refuses(problem: Error, text: String) raises:
    require(String(problem) == text, "refusal text: got '" + String(problem) + "', expected '" + text + "'")


def _geom() -> ConvGeom:
    # 3x3 kernel, stride 1, padding 1, dilation 1, 4 outputs, one group, bias on
    return ConvGeom(1, 3, 3, 1, 1, 1, 1, 1, 1, 4, True)


def run_case(unit: Int) raises:
    if unit == 0:
        conv_run_check_config(4, 8, 2, [3, 3], [1, 1], [1, 1])
        try:
            conv_run_check_config(3, 8, 2, [3, 3], [1, 1], [1, 1])
            require(False, "indivisible channels accepted")
        except problem:
            _refuses(problem, "mojolearn: positive in_channels and out_channels must be divisible by groups")
        try:
            conv_run_check_config(4, 8, 1, [3, 0], [1, 1], [1, 1])
            require(False, "zero kernel accepted")
        except problem:
            _refuses(problem, "mojolearn: kernel_size, stride and dilation must be positive")
    elif unit == 1:
        require(conv_run_pad_code("zeros") == 0 and conv_run_pad_code("reflect") == 1, "pad code zeros/reflect")
        require(conv_run_pad_code("replicate") == 2 and conv_run_pad_code("circular") == 3, "pad code others")
        conv_run_check_padding_mode("circular")
        try:
            conv_run_check_padding_mode("mirror")
            require(False, "unknown padding mode accepted")
        except problem:
            _refuses(problem, "mojolearn: padding_mode must be zeros, reflect, replicate or circular")
    elif unit == 2:
        var same = conv_run_text_pads("same", [4, 3], [1, 1], [1, 2])
        require(same[0] == 1 and same[1] == 2 and same[2] == 2 and same[3] == 2, "same pads")
        var valid = conv_run_text_pads("valid", [3, 3], [2, 2], [1, 1])
        require(valid[0] == 0 and valid[3] == 0, "valid pads")
        try:
            _ = conv_run_text_pads("same", [3, 3], [2, 1], [1, 1])
            require(False, "strided same accepted")
        except problem:
            _refuses(problem, "mojolearn: padding='same' is not supported for strided convolutions")
        try:
            _ = conv_run_text_pads("full", [3, 3], [1, 1], [1, 1])
            require(False, "unknown padding string accepted")
        except problem:
            _refuses(problem, "mojolearn: invalid padding string; expected valid or same")
    elif unit == 3:
        var two = conv_run_size_pads([1, 2], False)
        require(two[0] == 1 and two[1] == 1 and two[2] == 2 and two[3] == 2, "2-D size pads")
        var one = conv_run_size_pads([1, 3], True)
        require(one[0] == 0 and one[1] == 0 and one[2] == 3 and one[3] == 3, "1-D size pads")
        try:
            _ = conv_run_size_pads([0, -1], False)
            require(False, "negative padding accepted")
        except problem:
            _refuses(problem, "mojolearn: padding must be nonnegative")
    elif unit == 4:
        require(not conv_run_explicit("zeros", [1, 1, 2, 2]), "symmetric zeros is implicit")
        require(conv_run_explicit("zeros", [1, 2, 2, 2]) and conv_run_explicit("reflect", [1, 1, 1, 1]), "explicit")
        var symmetric = conv_run_conv_pad(False, [1, 1, 2, 2])
        var explicit = conv_run_conv_pad(True, [1, 1, 2, 2])
        require(symmetric[0] == 1 and symmetric[1] == 2 and explicit[0] == 0 and explicit[1] == 0, "conv pad")
    elif unit == 5:
        require(conv_run_fan(8, 2, [3, 3]) == 36, "fan")
        require(conv_run_bound(4) == 0.5, "init bound")
    elif unit == 6:
        require(conv_run_same_shape([2, 3], [2, 3]), "same shape")
        require(not conv_run_same_shape([2, 3], [2, 4]) and not conv_run_same_shape([2], [2, 1]), "different shape")
    elif unit == 7:
        var w = conv_run_words(ConvGeom(2, 3, 5, 1, 2, 1, 0, 1, 1, 8, False), [2, 4, 10, 12])
        require(len(w) == 17 and w[0] == 2 and w[1] == 2 and w[2] == 10 and w[3] == 12 and w[4] == 4, "words head")
        require(w[5] == 3 and w[6] == 5 and w[8] == 2 and w[9] == 1 and w[10] == 0, "words kernel/stride/pad")
        require(w[13] == 0 and w[14] == 0 and w[15] == 0 and w[16] == 0, "words tail")
        require(conv_run_words(_geom(), [1, 3, 8, 8])[15] == 1, "bias word")
    elif unit == 8:
        var same = conv_run_padded(False, [1, 2, 3, 4], [1, 3, 8, 8])
        var grown = conv_run_padded(True, [1, 2, 3, 4], [1, 3, 8, 8])
        require(same[2] == 8 and grown[2] == 11 and grown[3] == 15, "padded shape")
        var o = conv_run_out(_geom(), [1, 3, 8, 8], [1, 3, 8, 8])
        require(o[0] == 1 and o[1] == 4 and o[2] == 8 and o[3] == 8, "output shape")
    elif unit == 9:
        conv_run_check_rank([1, 3, 8, 8])
        try:
            conv_run_check_rank([3, 8, 8])
            require(False, "rank-three image accepted")
        except problem:
            _refuses(problem, "mojolearn: Conv2d.forward takes (N, C, H, W)")
        try:
            conv_run_check_channels(4, 3)
            require(False, "channel mismatch accepted")
        except problem:
            _refuses(problem, "mojolearn: input has 4 channels, the layer 3")
    elif unit == 10:
        try:
            conv_run_check_grad([1, 4, 8, 8], [1, 4, 8, 7])
            require(False, "gradient shape mismatch accepted")
        except problem:
            _refuses(problem, "mojolearn: grad_out shape differs from the convolution output shape")
        conv_run_check_weight([4, 3, 3, 3], 4, 3, [3, 3])
        try:
            conv_run_check_weight([4, 3, 3, 3], 4, 2, [3, 3])
            require(False, "weight shape mismatch accepted")
        except problem:
            _refuses(problem, "mojolearn: weight_ shape differs from configured convolution")
    elif unit == 11:
        var p = conv_run_pad_params([1, 2, 3, 4], [1, 1, 2, 2], 1, -1)
        require(len(p) == 9 and p[4] == 1 and p[7] == 2 and p[8] == 1, "pad params")
        require(len(conv_run_pad_params([1, 2, 3, 4], [0, 0, 0, 0], 0, 1)) == 10, "pad params with last word")
        var m = conv_run_mw(13, [5, 6])
        require(len(m) == 3 and m[0] == 13 and m[2] == 6, "residency prefix")
        var g = conv_run_group_prm([2, 6, 3, 4], 6, 3, 1, 0)
        require(g[0] == 3 and g[1] == 2 and g[2] == 6 and g[3] == 12 and g[4] == 2 and g[5] == 2 and g[6] == 0,
                "channel copy words")
    elif unit == 12:
        require(conv_run_row_width([4, 3, 5]) == 15 and conv_run_row_width([0, 3]) == 0, "row width")
        require(conv_run_tail_size([4, 3, 5]) == 15 and conv_run_tail_size([4]) == 1, "tail size")
        require(conv_run_spatial([2, 3, 4, 5]) == 20 and conv_run_spatial([2, 3]) == 1, "spatial size")
    elif unit == 13:
        var f = conv_run_four([2, 3, 7])
        var t = conv_run_three([2, 3, 1, 7])
        require(f[2] == 1 and f[3] == 7 and len(t) == 3 and t[2] == 7, "1-D shapes")
        try:
            conv_run_check_ncl([2, 3])
            require(False, "rank-two Conv1d input accepted")
        except problem:
            _refuses(problem, "mojolearn: Conv1d.forward takes (N, C, L)")
        try:
            conv_run_check_ncl_grad([2, 3, 1, 7])
            require(False, "rank-four Conv1d gradient accepted")
        except problem:
            _refuses(problem, "mojolearn: expected (N, C, L)")
    elif unit == 14:
        try:
            conv_run_check_relu([2, 3], [3, 2])
            require(False, "ReLU gradient mismatch accepted")
        except problem:
            _refuses(problem, "ReLU gradient shape differs from its input")
        try:
            conv_run_check_block_grad([2, 3], [2, 4])
            require(False, "block gradient mismatch accepted")
        except problem:
            _refuses(problem, "mojolearn: BasicBlock grad_out shape differs from forward output")
        try:
            conv_run_check_identity(10, 12)
            require(False, "identity size mismatch accepted")
        except problem:
            _refuses(problem, "mojolearn: the identity's shape is not the block's output shape")
    else:
        # (N=2, C=3, H=1, W=2): channel 1 of each image out, then back into a zeroed image
        var src: List[Float32] = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0]
        var part = List[Float32](length=4, fill=0.0)
        var back = List[Float32](length=12, fill=0.0)
        conv_run_take_channels(Int(part.unsafe_ptr()), Int(src.unsafe_ptr()), [2, 3, 1, 2], 1, 1)
        require(part[0] == 2.0 and part[1] == 3.0 and part[2] == 8.0 and part[3] == 9.0, "take channels")
        conv_run_put_channels(Int(back.unsafe_ptr()), Int(part.unsafe_ptr()), [2, 3, 1, 2], 1, 1)
        require(back[2] == 2.0 and back[9] == 9.0 and back[0] == 0.0 and back[4] == 0.0, "put channels")
        conv_run_copy_words(Int(back.unsafe_ptr()), 10, Int(src.unsafe_ptr()), 2)
        require(back[10] == 0.0 and back[11] == 1.0, "copy words at offset")
        _ = src
        _ = part
        _ = back


def run_checks() raises:
    var failures = List[Int](length=XCNN_CONV_RUN_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def task(unit: Int) {imm flags}:
        try:
            run_case(unit)
        except problem:
            flags[unit] = 1
            print("FAIL x_cnn conv run case", unit, String(problem))
    host_parallelize(task, XCNN_CONV_RUN_CASES)
    var failed = 0
    for failure in failures:  # collect independent case status words
        failed += failure
    _ = failures
    require(failed == 0, "x_cnn conv run cases failed: " + String(failed))
    print("PASS x_cnn conv run", XCNN_CONV_RUN_CASES)
