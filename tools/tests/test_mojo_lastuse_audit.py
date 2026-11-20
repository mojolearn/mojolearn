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
    var c = List[Float32](length=4, fill=0)
    var cp = MutPointer[Float32, MutAnyOrigin](unsafe_from_address=Int(c.unsafe_ptr()))
    fill(cp, 4)
'''
        kinds = {f["local"]: f["kind"] for f in lu.scan_text("t.mojo", text, {})}
        self.assertEqual(kinds, {"a": "tracked", "b": "any", "c": "any"})
        self.assertEqual(failing(text), [])

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
