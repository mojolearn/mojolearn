# SPDX-License-Identifier: Apache-2.0
"""The BatchNorm RUN half (`x_cnn/bn_run.mojo`) and the shared CNN door RUN
words (`x_cnn/door_run.mojo`), reached without Python (lane
doorsplit-cnn-xseq, 2026-11-11). One case per RUN function: literal words
and the layer's own refusal texts. The native entries the RUN calls are
covered by `xcnn_entry_table_suite.mojo`.

The cases are independent units and run across the native host worker pool.
"""
from core.host_parallel import host_parallelize
from core.entry_table import entry_ref_unresolved
from x_cnn.door_run import (
    XcnnRoute, XCNN_F_RESIDENT, XCNN_F2_ALL_OFF, xcnn_size, xcnn_shape_text, xcnn_run_reshape_check,
    xcnn_run_infer_dims, xcnn_run_copy_f32, xcnn_run_fill_f32,
)
from x_cnn.bn_run import (
    BnSettings, bn_run_prep, bn_run_aux_head, bn_run_aux_count, bn_run_untracked_running, bn_run_check_channels,
    bn_run_host_dims, bn_run_check_saved, bn_run_check_nonneg, bn_run_flags, bn_run_dx_shape_check,
)

comptime XCNN_BN_RUN_CASES = 12


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error(message)


def _refuses(problem: Error, text: String) raises:
    require(String(problem) == text, "refusal text: got '" + String(problem) + "', expected '" + text + "'")


def case(unit: Int) raises:
    if unit == 0:
        require(xcnn_size([2, 3, 4]) == 24 and xcnn_size(List[Int]()) == 1, "xcnn_size")
        require(xcnn_shape_text([5]) == "(5,)" and xcnn_shape_text([2, 3]) == "(2, 3)", "xcnn_shape_text")
    elif unit == 1:
        var dims = xcnn_run_infer_dims([2, -1, 3], 24)
        require(dims[1] == 4, "inferred dimension")
        try:
            _ = xcnn_run_infer_dims([-1, -1], 4)
            require(False, "two inferred dimensions accepted")
        except problem:
            _refuses(problem, "ValueError: CNN reshape accepts only one inferred dimension")
        try:
            xcnn_run_reshape_check([2, 3], [4, 2])
            require(False, "size-changing reshape accepted")
        except problem:
            _refuses(problem, "mojolearn: cannot reshape (2, 3) to (4, 2)")
    elif unit == 2:
        var src: List[Float32] = [1.0, 2.0, 3.0]
        var dst = List[Float32](length=3, fill=0.0)
        xcnn_run_copy_f32(Int(dst.unsafe_ptr()), Int(src.unsafe_ptr()), 3)
        xcnn_run_fill_f32(Int(src.unsafe_ptr()), 2, 7.0)
        require(dst[0] == 1.0 and dst[2] == 3.0 and src[0] == 7.0 and src[2] == 3.0, "copy/fill words")
        _ = src
        _ = dst
    elif unit == 3:
        var on = XcnnRoute(entry_ref_unresolved(), True, XCNN_F_RESIDENT, 0, False)
        var off = XcnnRoute(entry_ref_unresolved(), True, XCNN_F_RESIDENT, XCNN_F2_ALL_OFF, False)
        var host = XcnnRoute(entry_ref_unresolved(), False, XCNN_F_RESIDENT, 0, False)
        require(on.resident() and not off.resident() and not host.resident(), "route residency")
        require(off.device_io_off() and not on.device_io_off(), "route io switch")
    elif unit == 4:
        var p = bn_run_prep(BnSettings(4, 1e-5, True, True, 2, False, 0.0))
        require(p.batch_stats and p.counted and p.tracked == 3 and p.factor == 1.0 / 3.0, "cumulative momentum")
    elif unit == 5:
        var p = bn_run_prep(BnSettings(4, 1e-5, True, True, 0, True, 0.1))
        require(p.counted and p.tracked == 1 and p.factor == 0.1, "explicit momentum")
        var e = bn_run_prep(BnSettings(4, 1e-5, False, True, 5, True, 0.1))
        require(not e.batch_stats and not e.counted and e.tracked == 5 and e.factor == 0.0, "eval with tracking")
        var u = bn_run_prep(BnSettings(4, 1e-5, False, False, 0, True, 0.1))
        require(u.batch_stats and not u.counted, "eval without tracking")
    elif unit == 6:
        require(bn_run_aux_count(3) == 23, "aux count")
        var aux = List[Float32](length=23, fill=0.0)
        bn_run_aux_head(Int(aux.unsafe_ptr()), 1e-5, 0.25)
        require(aux[0] == Float32(1e-5) and aux[1] == 0.25 and aux[2] == 0.0, "aux head words")
        _ = aux
    elif unit == 7:
        var running = List[Float32](length=6, fill=0.0)
        bn_run_untracked_running(Int(running.unsafe_ptr()), 3)
        require(running[2] == 0.0 and running[3] == 1.0 and running[5] == 1.0, "untracked running block")
        _ = running
    elif unit == 8:
        var d = bn_run_host_dims([2, 3, 4, 5], 3)
        require(d[0] == 2 and d[1] == 3 and d[2] == 20, "NCHW dims")
        var flat = bn_run_host_dims([2, 3], 3)
        require(flat[2] == 1, "2-D input hw")
        var empty = bn_run_host_dims([0, 3, 4], 3)
        require(empty[2] == 0, "empty batch hw")
    elif unit == 9:
        try:
            _ = bn_run_host_dims([2], 3)
            require(False, "rank-one input accepted")
        except problem:
            _refuses(problem, "IndexError: tuple index out of range")
        try:
            bn_run_check_channels(4, 3)
            require(False, "channel mismatch accepted")
        except problem:
            _refuses(problem, "mojolearn: input has 4 channels, the layer 3")
    elif unit == 10:
        try:
            bn_run_check_saved([2, 3], 3)
            require(False, "rank-two saved shape accepted")
        except problem:
            _refuses(problem, "ValueError: batch normalization requires saved rank-three activation metadata")
        try:
            bn_run_check_saved([2, 4, 5], 3)
            require(False, "changed channels accepted")
        except problem:
            _refuses(problem, "ValueError: batch normalization configuration changed since forward")
        try:
            bn_run_check_nonneg(-1)
            require(False, "negative saved dimension accepted")
        except problem:
            _refuses(problem, "ValueError: batch normalization saved shape must be nonnegative")
    else:
        require(bn_run_flags(True, False, True) == 5 and bn_run_flags(False, True, False) == 2, "residency flags")
        try:
            bn_run_dx_shape_check(5, 6)
            require(False, "gradient size mismatch accepted")
        except problem:
            _refuses(problem, "ValueError: batch normalization gradient must match the saved activation size")


def run_checks() raises:
    var failures = List[Int](length=XCNN_BN_RUN_CASES, fill=0)
    var flags = failures.unsafe_ptr()
    def task(unit: Int) {imm flags}:
        try:
            case(unit)
        except problem:
            flags[unit] = 1
            print("FAIL x_cnn bn run case", unit, String(problem))
    host_parallelize(task, XCNN_BN_RUN_CASES)
    var failed = 0
    for failure in failures:  # collect independent case status words
        failed += failure
    _ = failures
    require(failed == 0, "x_cnn bn run cases failed: " + String(failed))
    print("PASS x_cnn bn run", XCNN_BN_RUN_CASES)
