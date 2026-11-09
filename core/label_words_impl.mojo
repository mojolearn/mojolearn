# SPDX-License-Identifier: Apache-2.0
"""Native label and id words of the Python doors (lane i3-move-data-loops,
2026-11-09): the loops over label, id and code words that lanes P1/P2/P3
wrote in the binding shells (bindings/mani_umap_native.mojo,
bindings/trees_public_gbdt_fit.mojo, bindings/prep_labels_native.mojo),
moved here word for word. The shells keep only the unpack (a buffer
address, a length, a word kind) and the pack. No Python here.

Every loop over independent words is split across the host's cores
(`host_parallelize`, rows per task from `host_predict_task_count_work`);
each word is a function of its own input, so the split moves no bit. The
sort, the distinct-class scan and the group-run scan are serial by nature
(each step continues the previous one)."""
from std.builtin.sort import sort
from core.cast_in import CAST_KIND_SIGNED, CAST_KIND_UNSIGNED, CAST_KIND_BOOL, CAST_KIND_FLOAT
from core.host_parallel import host_parallelize
from core.host_predict_threads import host_predict_task_count_work, host_predict_chunk

comptime LabelF32P = MutPointer[Float32, MutUntrackedOrigin]
comptime LabelF64P = MutPointer[Float64, MutUntrackedOrigin]
comptime LabelI64P = MutPointer[Int64, MutUntrackedOrigin]
comptime LabelI32P = MutPointer[Int32, MutUntrackedOrigin]
comptime LabelIntP = MutPointer[Int, MutUntrackedOrigin]


# ------------------------------------------------------------ one word


def label_word_i64(addr: Int, kind: Int, item: Int) -> Int64:
    """One integer or bool label word of a CastIn kind (signed, unsigned < 8
    bytes, bool) widened exactly."""
    if kind == CAST_KIND_SIGNED:
        if item == 1:
            return Int64(MutPointer[Int8, MutUntrackedOrigin](unsafe_from_address=addr)[0])
        if item == 2:
            return Int64(MutPointer[Int16, MutUntrackedOrigin](unsafe_from_address=addr)[0])
        if item == 4:
            return Int64(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr)[0])
        return MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=addr)[0]
    if kind == CAST_KIND_BOOL:
        return Int64(1) if MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=addr)[0] != 0 else Int64(0)
    if item == 1:
        return Int64(MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=addr)[0])
    if item == 2:
        return Int64(MutPointer[UInt16, MutUntrackedOrigin](unsafe_from_address=addr)[0])
    return Int64(MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=addr)[0])


def label_word_f64(addr: Int, item: Int) -> Float64:
    """One float32 (`item` 4) or float64 label word as float64."""
    if item == 4:
        return Float64(MutPointer[Float32, MutUntrackedOrigin](unsafe_from_address=addr)[0])
    return MutPointer[Float64, MutUntrackedOrigin](unsafe_from_address=addr)[0]


def id_word_coded(addr: Int, code: Int, i: Int) -> Int:
    """Word `i` of a group-id vector whose element kind was decided once
    (bindings/trees_public_gbdt_fit.mojo `_id_code`: 0 <i8, 1 <i4, 2 <u4,
    3 <i2, 4 <u2, 5 i1, 6 u1)."""
    if code == 0:
        return Int(MutPointer[Int64, MutUntrackedOrigin](unsafe_from_address=addr)[i])
    if code == 1:
        return Int(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=addr)[i])
    if code == 2:
        return Int(MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=addr)[i])
    if code == 3:
        return Int(MutPointer[Int16, MutUntrackedOrigin](unsafe_from_address=addr)[i])
    if code == 4:
        return Int(MutPointer[UInt16, MutUntrackedOrigin](unsafe_from_address=addr)[i])
    if code == 5:
        return Int(MutPointer[Int8, MutUntrackedOrigin](unsafe_from_address=addr)[i])
    return Int(MutPointer[UInt8, MutUntrackedOrigin](unsafe_from_address=addr)[i])


# ------------------------------------------------------------ word vectors


def label_words_f64(base: Int, stride: Int, item: Int, n: Int, mut words: List[Float64]) -> Bool:
    """`words` = the n float words at `base + i * stride` (`item` 4 or 8
    bytes) as float64. False when one is NaN (the caller then takes the
    object path). An exact float list unpacked to an `array.array('d')` is
    read with stride 8, item 8."""
    words = List[Float64](unsafe_uninit_length=n)
    if n <= 0:
        return True
    var wp = LabelF64P(unsafe_from_address=Int(words.unsafe_ptr()))
    var tasks = host_predict_task_count_work(n, n)
    var chunk = host_predict_chunk(n, tasks)
    var flags = List[Int](length=max(tasks, 1), fill=0)
    var fp = LabelIntP(unsafe_from_address=Int(flags.unsafe_ptr()))

    def _task(c: Int) {imm base, imm stride, imm item, imm n, imm chunk, imm wp, imm fp}:
        var lo = c * chunk
        var hi = min(lo + chunk, n)
        for i in range(lo, hi):
            var v = label_word_f64(base + i * stride, item)
            if v != v:
                fp[c] = 1
            wp[i] = v

    host_parallelize(_task, tasks)
    var ok = True
    for c in range(tasks):
        if fp[c] != 0:
            ok = False
    _ = flags^
    return ok


def label_words_i64(base: Int, stride: Int, kind: Int, item: Int, n: Int) -> List[Int64]:
    """The n integer or bool words at `base + i * stride` widened to int64
    (`label_word_i64`). An exact int list unpacked to an `array.array('q')`
    is read with stride 8, CAST_KIND_SIGNED, item 8."""
    var words = List[Int64](unsafe_uninit_length=n)
    if n <= 0:
        return words^
    var wp = LabelI64P(unsafe_from_address=Int(words.unsafe_ptr()))
    var tasks = host_predict_task_count_work(n, n)
    var chunk = host_predict_chunk(n, tasks)

    def _task(c: Int) {imm base, imm stride, imm kind, imm item, imm n, imm chunk, imm wp}:
        var lo = c * chunk
        var hi = min(lo + chunk, n)
        for i in range(lo, hi):
            wp[i] = label_word_i64(base + i * stride, kind, item)

    host_parallelize(_task, tasks)
    return words^


def id_words_coded(addr: Int, code: Int, n: Int) -> List[Int]:
    """The n words of a group-id vector (`id_word_coded`) as Int."""
    var words = List[Int](unsafe_uninit_length=n)
    if n <= 0:
        return words^
    var wp = LabelIntP(unsafe_from_address=Int(words.unsafe_ptr()))
    var tasks = host_predict_task_count_work(n, n)
    var chunk = host_predict_chunk(n, tasks)

    def _task(c: Int) {imm addr, imm code, imm n, imm chunk, imm wp}:
        var lo = c * chunk
        var hi = min(lo + chunk, n)
        for i in range(lo, hi):
            wp[i] = id_word_coded(addr, code, i)

    host_parallelize(_task, tasks)
    return words^


# ------------------------------------------------------------ code words


def label_codes_i32_to_f32(codes: List[Int32], dst: LabelF32P):
    """`dst[i] = float32(codes[i])` for every code (the exact
    Int -> Float64 -> Float32 widening the shells wrote)."""
    var n = len(codes)
    if n <= 0:
        return
    var cp = LabelI32P(unsafe_from_address=Int(codes.unsafe_ptr()))
    var tasks = host_predict_task_count_work(n, n)
    var chunk = host_predict_chunk(n, tasks)

    def _task(c: Int) {imm cp, imm dst, imm n, imm chunk}:
        var lo = c * chunk
        var hi = min(lo + chunk, n)
        for i in range(lo, hi):
            dst[i] = Float32(Float64(Int(cp[i])))

    host_parallelize(_task, tasks)


def label_codes_i32_to_f64(src: Int, n: Int) -> List[Float64]:
    """The n int32 code words at `src` widened to float64."""
    var out = List[Float64](unsafe_uninit_length=n)
    if n <= 0:
        return out^
    var sp = LabelI32P(unsafe_from_address=src)
    var op = LabelF64P(unsafe_from_address=Int(out.unsafe_ptr()))
    var tasks = host_predict_task_count_work(n, n)
    var chunk = host_predict_chunk(n, tasks)

    def _task(c: Int) {imm sp, imm op, imm n, imm chunk}:
        var lo = c * chunk
        var hi = min(lo + chunk, n)
        for i in range(lo, hi):
            op[i] = Float64(sp[i])

    host_parallelize(_task, tasks)
    return out^


def label_codes_i32_to_int(codes: List[Int32]) -> List[Int]:
    """The int32 code words widened to Int."""
    var n = len(codes)
    var out = List[Int](unsafe_uninit_length=n)
    if n <= 0:
        return out^
    var cp = LabelI32P(unsafe_from_address=Int(codes.unsafe_ptr()))
    var op = LabelIntP(unsafe_from_address=Int(out.unsafe_ptr()))
    var tasks = host_predict_task_count_work(n, n)
    var chunk = host_predict_chunk(n, tasks)

    def _task(c: Int) {imm cp, imm op, imm n, imm chunk}:
        var lo = c * chunk
        var hi = min(lo + chunk, n)
        for i in range(lo, hi):
            op[i] = Int(cp[i])

    host_parallelize(_task, tasks)
    return out^


# ------------------------------------------------------------ dense codes


def label_dense_codes[dt: DType, //](words: List[Scalar[dt]], dst: LabelF32P):
    """UMAP's categorical target codes: labels other than -1 numbered 0.. in
    ascending value order, -1 kept as -1, written as float32 to `dst[0..n)`.
    One serial sort and distinct scan of the labeled words, then one binary
    search per label, the labels split across the host's cores."""
    var n = len(words)
    var order = List[Scalar[dt]](capacity=n)
    for i in range(n):
        if words[i] != Scalar[dt](-1):
            order.append(words[i])
    sort(order)
    var classes = List[Scalar[dt]](capacity=len(order))
    for i in range(len(order)):
        if len(classes) == 0 or classes[len(classes) - 1] != order[i]:
            classes.append(order[i])
    if n <= 0:
        return
    var k = len(classes)
    var wp = MutPointer[Scalar[dt], MutUntrackedOrigin](unsafe_from_address=Int(words.unsafe_ptr()))
    var kp = MutPointer[Scalar[dt], MutUntrackedOrigin](unsafe_from_address=Int(classes.unsafe_ptr()))
    var tasks = host_predict_task_count_work(n, n * 8)
    var chunk = host_predict_chunk(n, tasks)

    def _task(c: Int) {imm wp, imm kp, imm k, imm dst, imm n, imm chunk}:
        var lo_row = c * chunk
        var hi_row = min(lo_row + chunk, n)
        for i in range(lo_row, hi_row):
            var w = wp[i]
            if w == Scalar[dt](-1):
                dst[i] = Float32(-1.0)
                continue
            var lo = 0
            var hi = k - 1
            while lo < hi:
                var mid = (lo + hi) // 2
                if kp[mid] < w:
                    lo = mid + 1
                else:
                    hi = mid
            dst[i] = Float32(Float64(lo))

    host_parallelize(_task, tasks)
    _ = classes^  # the tasks read the class words through kp
    _ = order^


# ------------------------------------------------------------ group runs


def group_runs_scan(words: List[Int], mut sizes: List[Int]) -> Int:
    """The run lengths of consecutive equal group words into `sizes`; the
    row of the first word that repeats an earlier run (a split group), or
    -1. Serial by nature: each run continues the previous one."""
    sizes = List[Int]()
    var seen = Dict[Int, Int]()
    var last = 0
    for i in range(len(words)):
        var value = words[i]
        if i > 0 and value == last:
            sizes[len(sizes) - 1] += 1
            continue
        if value in seen:
            return i
        seen[value] = i
        sizes.append(1)
        last = value
    return -1


def group_sizes_store_u32(sizes: List[Int], dst: Int):
    """`dst[i] = uint32(sizes[i])` for the run lengths."""
    var p = MutPointer[UInt32, MutUntrackedOrigin](unsafe_from_address=dst)
    for i in range(len(sizes)):
        p[i] = UInt32(sizes[i])
