"""Tooling tests for tools/mojo_lastuse_audit.py: the patterns behind the 2026-11-16 crashes fail, their keep-alive
fixes and the origins that keep a value alive pass, the allow-list is honoured. Pure Python; nothing is compiled."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import mojo_lastuse_audit as lu  # noqa: E402

HELPER = '''
def _a(mut x: List[Int64]) -> Int:
    return Int(x.unsafe_ptr())
'''


def failing(text, helpers=None, path="t.mojo"):
    helpers = lu.helpers_of(text) if helpers is None else helpers
    return [(f["local"], f["sink"]) for f in lu.scan_text(path, text, helpers) if f["fails"]]


class PatternTest(unittest.TestCase):

    def test_untracked_pointer_used_after_last_use_fails(self):
        # xtrees/calibrated_apply.mojo:360 (segfaults-4)
        text = '''
def f(n: Int, scores: MutPointer[Float64, MutUntrackedOrigin]):
    var own = List[Float64](length=n, fill=0.0)
    var sp = own.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    for i in range(n):
        sp[i] = 1.0
    scores[0] = sp[0]
'''
        self.assertEqual(failing(text), [("own", "assign")])
        fixed = text + "    _ = own^\n"
        self.assertEqual(failing(fixed), [])

    def test_address_helper_as_call_argument_fails(self):
        # the five native check suites (native-checks-2): `entry(_a(dst))` frees dst before entry writes it
        text = HELPER + '''
def g() raises:
    var dst: List[Int64] = [0]
    entry(3, _a(dst))
'''
        self.assertEqual(failing(text), [("dst", "call")])
        self.assertEqual(failing(text + "    _ = dst^\n"), [])

    def test_inline_address_argument_fails(self):
        text = '''
def g() raises:
    var dst = List[Int64](length=3, fill=0)
    gather_i64(1, 2, Int(dst.unsafe_ptr()))
'''
        self.assertEqual(failing(text), [("dst", "call")])

    def test_tracked_and_any_origin_pass(self):
        text = '''
def g() raises:
    var a = List[Float32](length=4, fill=0)
    fill(a.unsafe_ptr(), 4)
    var b = List[Float32](length=4, fill=0)
    var bp = b.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()
    fill(bp, 4)
'''
        kinds = {f["local"]: f["kind"] for f in lu.scan_text("t.mojo", text, {})}
        self.assertEqual(kinds, {"a": "tracked", "b": "any"})
        self.assertEqual(failing(text), [])

    # lane gbdt-leaf-body (2026-11-23): an address has no origin, so an any-origin pointer built around it keeps
    # nothing alive (the shape of partial_walk's Python data address, here through `.unsafe_ptr()`)
    ANY_ADDR = '''
comptime F32P = MutPointer[Float32, MutAnyOrigin]

def g() raises:
    var c = List[Float32](length=4, fill=0)
    var cp = MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(c.unsafe_ptr()))
    fill(cp, 4)
    var d = List[Float32](length=4, fill=0)
    fill(F32P(unsafe_from_address=Int(d.unsafe_ptr())), 4)
'''

    def test_an_any_origin_pointer_around_an_address_is_untracked(self):
        found = {f["local"]: (f["kind"], f["sink"]) for f in lu.scan_text("t.mojo", self.ANY_ADDR,
                                                                          lu.helpers_of(self.ANY_ADDR))}
        self.assertEqual(found, {"c": ("untracked", "assign"), "d": ("untracked", "call")})
        self.assertEqual(sorted(failing(self.ANY_ADDR)), [("c", "assign"), ("d", "call")])

    def test_a_later_named_use_of_the_owner_passes(self):
        fixed = self.ANY_ADDR.replace("    fill(cp, 4)\n", "    fill(cp, 4)\n    _ = c^\n") + "    _ = d^\n"
        self.assertEqual(failing(fixed), [])

    def test_loop_keeps_the_value_alive_across_the_back_edge(self):
        text = '''
def g() raises:
    var position = List[Int](length=1, fill=0)
    while next_item(Int(position.unsafe_ptr())):
        work()
'''
        self.assertEqual(failing(text), [])

    def test_moved_value_and_keepalive_reads(self):
        text = '''
def g(mut self) raises -> Int:
    var words = List[Float32](length=4, fill=0)
    var address = Int(words.unsafe_ptr())
    self.store.append(words^)
    return use(address)

def h() raises:
    var src = List[Float64](length=4, fill=0)
    var a: List[Int] = [Int(src.unsafe_ptr())]
    run(a)
    _ = src
    _ = a
'''
        self.assertEqual(failing(text), [])

    def test_value_read_through_the_pointer_is_not_a_pointer(self):
        text = '''
def g() raises -> Int:
    var h = List[Int32](length=1, fill=0)
    var u = Int(h.unsafe_ptr().unsafe_load(0))
    _ = h^
    return u
'''
        self.assertEqual(failing(text), [])

    def test_sibling_blocks_do_not_share_a_name(self):
        text = '''
def g(flag: Bool) raises -> Int:
    var r = 0
    if flag:
        var tmp = List[Int64](length=4, fill=0)
        var address = Int(tmp.unsafe_ptr())
        r = encode(address)
        _ = tmp^
    else:
        var address = 7
        r = address
    return r
'''
        self.assertEqual(failing(text), [])

    def test_imported_helper_resolves_through_its_module(self):
        files = [("pkg/helpers.mojo", HELPER),
                 ("pkg/user.mojo", "from pkg.helpers import _a\n\ndef g() raises:\n    var d: List[Int64] = [0]\n"
                                   "    entry(_a(d))\n"),
                 ("pkg/other.mojo", "from pkg.elsewhere import _a\n\ndef g() raises:\n    var d: List[Int64] = [0]\n"
                                    "    entry(_a(d))\n")]
        found = [f for f in lu.scan(files, workers=2) if f["fails"]]
        self.assertEqual([(f["path"], f["local"]) for f in found], [("pkg/user.mojo", "d")])


class HelperCarrierTest(unittest.TestCase):
    """HELPERS THAT CARRY AN ADDRESS (lane lastuse-helpers, 2026-11-25): the address is taken through a function
    whose return value carries its argument's address, found from the code, not from a list of names."""

    LINEAR2 = '''
comptime _P = MutPointer[Float32, MutUntrackedOrigin]


def _p(values: List[Float32]) -> _P:
    return _P(unsafe_from_address=Int(values.unsafe_ptr()))


def _device_case(which: Int) raises:
    if which == 1:
        var coef: List[Float32] = [Float32(0.25), Float32(-1.5), Float32(2.0), Float32(0.125)]
        var f_order = List[Float32](length=n * FD, fill=0.0)
        for r in range(n):
            f_order[r] = gathered[r]
        var oa = List[Float32](length=n, fill=0.0)
        var ob = List[Float32](length=n, fill=0.0)
        cd_predict_host(ctx, word, _p(coef), _p(oa), n, FD, Float32(0.75))
        cd_predict_host(ctx, _p(f_order), _p(coef), _p(ob), n, FD, Float32(0.75))
FIX1
        _same(oa, ob, "cd predictions")
    elif which == 5:
        var coef: List[Float32] = [Float32(0.5), Float32(-0.25)]
        var sa = ctx.enqueue_create_buffer[DType.float32](n)
        var sb = ctx.enqueue_create_buffer[DType.float32](n)
        qn_decision_function_device(ctx, word, _p(coef), sa, n, FD, True, 2)
        qn_decision_function_device(ctx, _p(gathered), _p(coef), sb, n, FD, True, 2)
FIX5
        _same(sa, sb, "QN decisions")
'''

    GRADS = '''
comptime _FP = MutPointer[Float32, MutUntrackedOrigin]


def _fp(v: List[Float32]) -> _FP:
    return _FP(unsafe_from_address=Int(v.unsafe_ptr()))


def test_chunked_lm_head_device_arrays() raises:
    var hidden = _fill(rows * width, 1)
    var weight = _fill(vocab * width, 2)
    var loss_h = List[Float32](length=1, fill=0.0)
    var mx = List[Float32](length=rows, fill=0.0)
    var dn = List[Float32](length=rows, fill=0.0)
    var dh_h = List[Float32](length=rows * width, fill=0.0)
    var dw_h = List[Float32](length=vocab * width, fill=0.0)
    _ = chunked_lm_head_v2_train_host(ctx, _fp(loss_h), _fp(mx), _fp(dn), _fp(dh_h), _fp(dw_h), _fp(hidden),
                                      _fp(weight), tp, rows, vocab, width)
FIX
    var hh = _put(ctx, hidden)
    var hw = _put(ctx, weight)
    _same(loss_h, dh_h, dw_h)
'''

    def test_vdf_scope_linear2_fold_pre_fix_is_found(self):
        # checks/native_tests/vdf_scope_linear2_fold.mojo before 701b026c5b
        text = self.LINEAR2.replace("FIX1\n", "").replace("FIX5\n", "")
        self.assertEqual(sorted(failing(text)), [("coef", "call"), ("coef", "call"), ("f_order", "call")])

    def test_vdf_scope_linear2_fold_fixed_passes(self):
        text = self.LINEAR2.replace("FIX1", "        _ = coef^\n        _ = f_order^").replace(
            "FIX5", "        _ = coef^")
        self.assertEqual(failing(text), [])

    def test_training_grads_pre_fix_is_found(self):
        # checks/native_tests/training_grads_device_arrays_gpu.mojo before 71b636f52c
        self.assertEqual(sorted(failing(self.GRADS.replace("FIX\n", ""))), [("dn", "call"), ("mx", "call")])

    def test_training_grads_fixed_passes(self):
        self.assertEqual(failing(self.GRADS.replace("FIX", "    _ = mx^\n    _ = dn^")), [])

    def test_helper_is_derived_from_the_code(self):
        h = lu.helpers_of(self.LINEAR2)
        self.assertEqual(h.get("_p"), "untracked")
        self.assertIn(0, h["args:_p"])

    def test_any_origin_pointer_around_an_address_carries_it(self):
        # x_decomp/checks/seam_util.mojo `ptr`: an any-origin pointer built from an Int address keeps nothing alive
        text = '''
comptime F32Ptr = MutPointer[Float32, MutAnyOrigin]


def ptr(values: List[Float32]) -> F32Ptr:
    return F32Ptr(unsafe_from_address=Int(values.unsafe_ptr()))


def g() raises:
    var a = zeros(4)
    gemm(ptr(a), 4)
'''
        self.assertEqual(failing(text), [("a", "call")])

    def test_tracked_or_any_origin_cast_helper_passes(self):
        text = '''
def _keep(mut v: List[Float32]) -> MutPointer[Float32, MutAnyOrigin]:
    return v.unsafe_ptr().unsafe_origin_cast[MutAnyOrigin]()


def g() raises:
    var a = zeros(4)
    gemm(_keep(a), 4)
'''
        self.assertNotIn("_keep", lu.helpers_of(text))
        self.assertEqual(failing(text), [])

    def test_helper_that_reads_through_the_address_is_no_carrier(self):
        # core/prep_member.mojo `_iword`: the pointer is indexed on the spot; an Int value comes back
        text = '''
def _iword(arena: HostZeroArena, offset: Int) -> Int:
    return Int(MutPointer[Int32, MutUntrackedOrigin](unsafe_from_address=Int(arena.unsafe_ptr()))[offset])


def g() raises:
    var arena = HostZeroArena(8)
    if _iword(arena, 0) > 0:
        raise Error(String(_iword(arena, 1)))
'''
        self.assertNotIn("_iword", lu.helpers_of(text))
        self.assertEqual(failing(text), [])

    def test_carried_argument_position_and_keyword(self):
        # core/member_class_weight.mojo `member_cw_rows(..., mut rows)` returns Int(rows.unsafe_ptr())
        text = '''
def rows_of(codes: Int, n: Int, mut rows: List[Float32]) -> Int:
    if n == 0:
        return codes
    return Int(rows.unsafe_ptr())


def g() raises:
    var codes = List[Int32](length=4, fill=0)
    var rows = List[Float32]()
    var at = rows_of(Int(codes.unsafe_ptr()), 4, rows)
    _ = codes^
    use(at)
'''
        self.assertEqual(lu.helpers_of(text)["args:rows_of"], frozenset({2, "rows"}))
        self.assertEqual(failing(text), [("rows", "assign")])
        self.assertEqual(failing(text + "    _ = rows^\n"), [])
        compared = text.replace("    use(at)\n", "    require(at == 0, \"no rows\")\n")
        self.assertEqual(failing(compared), [])
        asserted = text.replace("    var at = rows_of(", "    assert_equal(rows_of(").replace(
            "4, rows)\n", "4, rows), 0)\n").replace("    use(at)\n", "")
        self.assertEqual(failing(asserted), [])
        discarded = text.replace("    var at = rows_of(", "    _ = rows_of(").replace("    use(at)\n", "")
        self.assertEqual(failing(discarded), [])

    def test_method_returning_an_address_of_self(self):
        text = '''
struct Arena:
    var words: List[Int32]

    def base(self) -> Int:
        return Int(self.words.unsafe_ptr())


def g() raises:
    var arena = Arena()
    var base = arena.base()
    run(base)
'''
        self.assertEqual(lu.helpers_of(text).get("meth:base"), "untracked")
        self.assertEqual(failing(text), [("arena", "assign")])
        self.assertEqual(failing(text + "    _ = arena^\n"), [])
        # a value read through the address is no pointer that outlives the owner
        read = text.replace("    run(base)\n", "    var v = MutPointer[Int32, MutAnyOrigin](unsafe_from_address=base)[0]"
                                              "\n    _ = arena^\n    run(v)\n")
        self.assertEqual(failing(read), [])

    def test_constructor_storing_an_address_is_a_carrier(self):
        text = '''
struct View:
    var at: Int

    def __init__(out self, values: List[Float32]):
        self.at = Int(values.unsafe_ptr())


def g() raises:
    var a = zeros(4)
    kernel(View(a))
'''
        self.assertEqual(lu.helpers_of(text).get("View"), "untracked")
        self.assertEqual(failing(text), [("a", "call")])

    def test_sub_buffer_view_frees_nothing(self):
        # training/samba_resident.mojo: `var wv = s.view(j)` is a create_sub_buffer view of s.param
        text = '''
comptime FP = MutPointer[Float32, MutAnyOrigin]


struct Store:
    var param: DeviceBuffer[DType.float32]

    def view(self, j: Int) raises -> DeviceBuffer[DType.float32]:
        return self.param.create_sub_buffer[DType.float32](j, 4)


def _fpb(mut buf: DeviceBuffer[DType.float32]) -> FP:
    return FP(unsafe_from_address=Int(buf.unsafe_ptr()))


def g(s: Store) raises:
    for k in range(9):
        var wv = s.view(k)
        params.append(variable_into(_fpb(wv), 4))
'''
        self.assertEqual(failing(text), [])

    def test_use_ending_a_loop_body_lives_to_the_loop_exit(self):
        text = HELPER + '''
def g() raises:
    var q = List[Int64](length=4, fill=0)
    for p in range(4):
        out[p] = cell(_a(q) + p)
'''
        self.assertEqual(failing(text), [])

    def test_wrapper_of_an_imported_helper_is_found_across_files(self):
        files = [("pkg/helpers.mojo", HELPER),
                 ("pkg/wrap.mojo", "from pkg.helpers import (\n    _a,\n)\n\n\ndef _b(mut v: List[Int64]) -> Int:\n"
                                   "    return _a(v)\n"),
                 ("pkg/user.mojo", "from pkg.wrap import _b\n\ndef g() raises:\n    var d: List[Int64] = [0]\n"
                                   "    entry(_b(d))\n")]
        found = [f for f in lu.scan(files, workers=2) if f["fails"]]
        self.assertEqual([(f["path"], f["local"]) for f in found], [("pkg/user.mojo", "d")])

    def test_helper_over_a_python_data_address_keeps_the_holder_exception(self):
        # bindings/msel_common_door.mojo `msel_addr` wraps classical_address: `pin.owner = codes` holds the data
        text = '''
def msel_addr(array: PythonObject) raises -> Int:
    return Int(py=classical_address(array))


def g(got: PythonObject) raises -> Pin:
    var pin = Pin()
    var codes = got[1]
    var address = msel_addr(codes)
    pin.owner = codes
    pin.view = Codes(address)
    return pin^
'''
        self.assertEqual(failing(text), [])


class QueuedCopyTest(unittest.TestCase):
    """FREED BEFORE THE DRAIN: an asynchronous enqueue_copy still reads a host local that Mojo freed at its last
    named use (hist2_check, partitions_reduce_check and greedy_search_helper.run_tree, 2026-11-18)."""

    PINNED = '''
def g(ctx: DeviceContext) raises:
    var d = ctx.enqueue_create_buffer[DType.uint32](4)
    var h = ctx.enqueue_create_host_buffer[DType.uint32](4)
    h.unsafe_ptr().unsafe_store(0, UInt32(7))
    ctx.enqueue_copy(dst_buf=d, src_ptr=h.unsafe_ptr())
    var z = ctx.enqueue_create_host_buffer[DType.uint32](4)
    z.unsafe_ptr().unsafe_store(0, UInt32(0))
    ctx.synchronize()
    _ = z^
'''

    def test_pinned_staging_freed_before_the_drain_fails(self):
        self.assertEqual(failing(self.PINNED), [("h", "enqueue_copy")])
        self.assertEqual(failing(self.PINNED + "    _ = h^\n"), [])

    def test_keepalive_before_the_drain_still_fails(self):
        # run_tree had `_ = q5^ ... # past the drain` ABOVE its synchronize
        text = self.PINNED.replace("    var z =", "    _ = h^\n    var z =")
        self.assertEqual(failing(text), [("h", "enqueue_copy")])

    def test_list_source_and_its_fix(self):
        text = '''
def g(ctx: DeviceContext) raises:
    var off = List[UInt32]()
    off.append(UInt32(3))
    var d = ctx.enqueue_create_buffer[DType.uint32](1)
    ctx.enqueue_copy(dst_buf=d, src_ptr=off.unsafe_ptr())
    ctx.synchronize()
'''
        # pointwise_scores_calcer: a pageable source dies at the copy, the drain comes after it
        self.assertEqual(failing(text), [("off", "enqueue_copy")])
        self.assertEqual(failing(text + "    _ = off^\n"), [])

    def test_pointer_alias_of_the_staging(self):
        # gls_borders_device: `var hs = h_seg.unsafe_ptr()`, then `src_ptr=hs + chunk`
        text = '''
def g(ctx: DeviceContext, d: DeviceBuffer[DType.uint32]) raises:
    var h_seg = ctx.enqueue_create_host_buffer[DType.uint32](8)
    ctx.synchronize()
    var hs = h_seg.unsafe_ptr()
    hs[0] = 1
    ctx.enqueue_copy(dst_buf=d, src_ptr=hs + 4)
    launch(d)
    ctx.synchronize()
'''
        self.assertEqual(failing(text), [("h_seg", "enqueue_copy")])
        self.assertEqual(failing(text + "    _ = h_seg^\n"), [])

    def test_drain_inside_a_callee_and_a_move_pass(self):
        text = '''
def settle[n: Int](ctx: DeviceContext) raises:
    ctx.synchronize()

def g(ctx: DeviceContext, d: DeviceBuffer[DType.uint32]) raises:
    var h = ctx.enqueue_create_host_buffer[DType.uint32](4)
    ctx.enqueue_copy(dst_buf=d, src_ptr=h.unsafe_ptr())
    settle[0](ctx)
    _ = h^

def k(ctx: DeviceContext, d: DeviceBuffer[DType.uint32], mut holds: List[HostBuffer[DType.uint32]]) raises:
    var h = ctx.enqueue_create_host_buffer[DType.uint32](4)
    ctx.enqueue_copy(dst_buf=d, src_ptr=h.unsafe_ptr())
    holds.append(h^)
'''
        self.assertEqual(failing(text), [])

    def test_drain_in_another_file_is_found_by_scan(self):
        # transitively: b calls wrap, which calls settle, which synchronizes (umap's _ug_get_i32 shape)
        files = [("pkg/a.mojo", "def settle(ctx: DeviceContext) raises:\n    ctx.synchronize()\n\n"
                                "def wrap(ctx: DeviceContext) raises:\n    settle(ctx)\n"),
                 ("pkg/b.mojo", "def g(ctx: DeviceContext, d: DeviceBuffer[DType.uint8]) raises:\n"
                                "    var h = ctx.enqueue_create_host_buffer[DType.uint8](1)\n"
                                "    ctx.enqueue_copy(dst_buf=d, src_ptr=h.unsafe_ptr())\n"
                                "    wrap(ctx)\n    _ = h^\n"),
                 ("pkg/c.mojo", "def g(ctx: DeviceContext, d: DeviceBuffer[DType.uint8]) raises:\n"
                                "    var h = ctx.enqueue_create_host_buffer[DType.uint8](1)\n"
                                "    ctx.enqueue_copy(dst_buf=d, src_ptr=h.unsafe_ptr())\n"
                                "    other(ctx)\n    _ = h^\n")]
        found = [f for f in lu.scan(files, workers=2) if f["fails"]]
        self.assertEqual([(f["path"], f["local"], f["kind"]) for f in found], [("pkg/c.mojo", "h", "queued")])


class AllowTest(unittest.TestCase):

    def test_allow_list_needs_a_reason_and_reports_stale_lines(self):
        allow, errors = lu.load_allow("a.mojo:f:x  # stack array\nb.mojo:g:y\nc.mojo:h:z  # gone\n")
        self.assertEqual(set(allow), {("a.mojo", "f", "x"), ("c.mojo", "h", "z")})
        self.assertEqual(len(errors), 1)
        found = [dict(path="a.mojo", function="f", local="x", fails=True),
                 dict(path="a.mojo", function="f", local="w", fails=True)]
        failing_, allowed, stale = lu.verdict(found, allow)
        self.assertEqual([f["local"] for f in failing_], ["w"])
        self.assertEqual([f["local"] for f in allowed], ["x"])
        self.assertEqual(stale, [("c.mojo", "h", "z")])


class PathAndPythonAddressTest(unittest.TestCase):
    """LAST USE ON THE PATH, AND PYTHON DATA ADDRESSES (lane lastuse-addr-arg, 2026-11-23)."""

    # checks/native_tests/gbdt_public_fits.mojo host_ordered before 90c52f3a7a: the branch returns, so `y` dies at
    # `wb`'s argument although the next branch names it again
    ORDERED = '''
def host_ordered(case_index: Int) raises:
    var x = load_f32("x", 12)
    var y = load_f32("y", 2)
    if case_index == 0:
        var wa = fit(Int(x.unsafe_ptr()), Int(y.unsafe_ptr()), 2)
        var wb = fit(Int(x.unsafe_ptr()), Int(y.unsafe_ptr()), 2)
        require(wa == wb, "differ")
        require(finite(predict(wa, x, 2)), "nonfinite")
        return
    var coded = load_f32("c", 12)
    if case_index == 1:
        var a = fit(Int(coded.unsafe_ptr()), Int(y.unsafe_ptr()), 2)
        var b = fit(Int(coded.unsafe_ptr()), Int(y.unsafe_ptr()), 2)
        require(a == b, "differ")
        _ = y
        return
    elif case_index == 2:
        var c = fit(Int(coded.unsafe_ptr()), 2)
        _ = coded
    else:
        var e = fit(Int(coded.unsafe_ptr()), 3)
        _ = coded
    require(y[0] == coded[0], "late use")
'''

    # bindings/prep_classify_common_door.mojo partial_walk before a1bbc6c2ac
    WALK = '''
def partial_walk(y: PythonObject, n: Int, context: PythonObject) raises -> PythonObject:
    var labels = classical_encode_labels(y, context)
    var out = i32_array(n, context)
    var dest = I32P(unsafe_from_address=Int(py=classical_address(out)))
    _ = out
    var src = I32P(unsafe_from_address=Int(py=classical_address(labels.codes)))
    for i in range(n):
        dest[i] = src[i]
        _ = out
    return out
'''

    def test_a_returning_branch_ends_the_path(self):
        found = [(f["local"], f["line"]) for f in lu.scan_text("t.mojo", self.ORDERED, {}) if f["fails"]]
        self.assertEqual(found, [("coded", 14), ("y", 7)])

    def test_the_keep_alives_of_the_fix_pass(self):
        fixed = self.ORDERED.replace('        require(finite(predict(wa, x, 2)), "nonfinite")\n',
                                     '        require(finite(predict(wa, x, 2)), "nonfinite")\n        _ = y\n')
        fixed = fixed.replace('        _ = y\n        return\n    elif', '        _ = y\n        _ = coded\n'
                                                                       '        return\n    elif')
        self.assertEqual(failing(fixed), [])

    def test_sibling_branches_are_skipped_and_the_join_is_reached(self):
        text = '''
def g(flag: Int) raises:
    var a = List[Int64](length=3, fill=0)
    if flag == 0:
        gather(Int(a.unsafe_ptr()))
    elif flag == 1:
        _ = a^
    else:
        _ = a^
'''
        self.assertEqual(failing(text), [("a", "call")])
        joined = text + "    _ = a\n"
        self.assertEqual(failing(joined), [])

    def test_a_python_data_address_of_a_field_is_untracked(self):
        self.assertEqual(failing(self.WALK), [("labels", "assign")])
        fixed = self.WALK.replace("    return out\n", "    _ = labels.codes\n    return out\n")
        self.assertEqual(failing(fixed), [])

    def test_a_python_data_address_as_a_call_argument_fails(self):
        text = '''
def g(context: PythonObject) raises:
    var out = i32_array(4, context)
    fill_i32(Int(py=classical_address(out)), 4)
'''
        self.assertEqual(failing(text), [("out", "call")])
        self.assertEqual(failing(text + "    _ = out\n"), [])

    def test_what_keeps_a_python_owner_alive_passes(self):
        text = '''
def admitted_held_address(held: Int) -> Int:
    return held

def g(fields: PythonObject, context: PythonObject) raises -> PythonObject:
    var a = i32_array(4, context)
    var r = KmArray(a, Int(py=classical_address(a)))
    var w = fields["_w"]
    fill_i32(Int(py=classical_address(w)), 4)
    var k = i32_array(4, context)
    _ = py_call(keep.append, Python.tuple(k, n))
    fill_i32(Int(py=classical_address(k)), 4)
    var v = i32_array(4, context)
    var same = r.address == Int(py=classical_address(v))
    var h = 3
    fill_i32(admitted_held_address(h), 4)
    try:
        var t = List[Int32](length=4, fill=0)
        try:
            fill_i32(Int(t.unsafe_ptr()), 4)
            raise Error("accepted")
        except e:
            pass
        _ = t
    except e:
        pass
    return r
'''
        self.assertEqual(failing(text), [])
        # the holder is a call that keeps it, not an address helper inside an append
        lost = '''
def g(fields: PythonObject, context: PythonObject) raises:
    var addresses = List[Int]()
    for i in range(3):
        var c = i32_array(4, context)
        addresses.append(Int(py=classical_address(c)))
    run(addresses)
'''
        self.assertEqual(failing(lost), [("c", "call")])
        # bindings/arena_prog_native_door.mojo _prep_run_impl: held only on one branch, then rebound
        partly = '''
def g(record: PythonObject, inout: Bool, context: PythonObject) raises:
    var held = List[PythonObject]()
    var value = record[1]
    if inout:
        held.append(value)
    value = convert(value, context)
    fill_f32(Int(py=classical_address(value)), 4)
    _ = held^
'''
        self.assertEqual(failing(partly), [("value", "call")])


class OwnCopyTest(unittest.TestCase):
    """A check under checks/native_tests/ enqueues no host-to-device copy itself: it calls the kit."""

    UPLOAD = """
def run_checks() raises:
    var ctx = DeviceContext()
    var d = ctx.enqueue_create_buffer[DType.float32](4)
    var h = ctx.enqueue_create_host_buffer[DType.float32](4)
    ctx.enqueue_copy(dst_buf=d, src_ptr=h.unsafe_ptr())
    ctx.synchronize()
    _ = h^
"""

    def _own(self, text, path):
        return [f for f in lu.scan_text(path, text, {}) if f["kind"] == "own-copy"]

    def test_a_native_check_that_enqueues_its_own_upload_fails(self):
        found = self._own(self.UPLOAD, "checks/native_tests/new_check.mojo")
        self.assertEqual([(f["local"], f["line"], f["fails"]) for f in found], [("h", 6, True)])
        self.assertIn("checks/native_tests/kit.mojo", found[0]["why"])

    def test_the_buffer_form_from_a_pinned_host_buffer_fails(self):
        text = self.UPLOAD.replace("src_ptr=h.unsafe_ptr()", "src_buf=h")
        self.assertEqual([f["local"] for f in self._own(text, "checks/native_tests/new_check.mojo")], ["h"])

    def test_the_kit_itself_and_files_elsewhere_are_not_judged(self):
        self.assertEqual(self._own(self.UPLOAD, "checks/native_tests/kit.mojo"), [])
        self.assertEqual(self._own(self.UPLOAD, "neighbors/checks/metric_check.mojo"), [])

    def test_kit_calls_downloads_and_device_copies_pass(self):
        text = """
def run_checks() raises:
    var ctx = DeviceContext()
    var d = ctx.enqueue_create_buffer[DType.float32](4)
    var e = ctx.enqueue_create_buffer[DType.float32](4)
    var h = ctx.enqueue_create_host_buffer[DType.float32](4)
    upload_host_into[DType.float32](ctx, d, h)
    ctx.enqueue_copy(dst_buf=e, src_buf=d)
    ctx.enqueue_copy(dst_ptr=h.unsafe_ptr(), src_buf=e)
    ctx.synchronize()
    _ = h^
"""
        self.assertEqual(self._own(text, "checks/native_tests/new_check.mojo"), [])

    def test_an_allow_line_with_a_reason_lets_a_deliberate_exception_through(self):
        found = lu.scan_text("checks/native_tests/queue_order.mojo", self.UPLOAD, {})
        failing, allowed, stale = lu.verdict(found, {("checks/native_tests/queue_order.mojo", "run_checks", "h"): "r"})
        self.assertEqual([f["kind"] for f in failing], [])
        self.assertIn("own-copy", [f["kind"] for f in allowed])
        self.assertEqual(stale, [])


class CommitTest(unittest.TestCase):

    def test_commit_blobs_and_allow_list_are_read_from_the_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            run = lambda *a: subprocess.run(a, cwd=repo, check=True, capture_output=True)  # noqa: E731
            run("git", "init", "-q")
            (repo / "tools").mkdir()
            (repo / "m.mojo").write_text("def g() raises:\n    var dst = List[Int64](length=3, fill=0)\n"
                                         "    gather(Int(dst.unsafe_ptr()))\n")
            run("git", "add", "-A")
            run("git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "a")
            lines = []
            self.assertEqual(lu.audit(repo, "HEAD", workers=2, out=lines.append), 1)
            self.assertTrue(any("m.mojo:3: dst" in line for line in lines), lines)
            (repo / lu.ALLOW).write_text("m.mojo:g:dst  # test\n")
            run("git", "add", "-A")
            run("git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "-m", "b")
            self.assertEqual(lu.audit(repo, "HEAD", workers=2, out=lines.append), 0)


if __name__ == "__main__":
    unittest.main()
