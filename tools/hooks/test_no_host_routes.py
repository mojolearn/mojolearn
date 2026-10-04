#!/usr/bin/env python3
"""Tests for tools/hooks/no_host_routes.py (pure Python, git only, no build).

Every class of CPU work found by the Oct 2 2026 GPU path audit is added to a
GPU file in a scratch commit on top of HEAD (built with a private index, so
the worktree and the branch never change), and the full-tree check must
refuse it. The negatives (comments, docstrings, tools/, a multi-GPU driver)
must pass, a removed route must leave a stale row that fails until pruned,
and in-flight allowances must serve only the PR head they belong to.

Run: python3 tools/hooks/test_no_host_routes.py   (or pytest on this file)
"""
import contextlib
import io
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import no_host_routes as nhr  # noqa: E402

TOP = subprocess.run(["git", "rev-parse", "--show-toplevel"], cwd=HERE, capture_output=True,
                     text=True).stdout.strip()
os.chdir(TOP)

DEV = "x_decomp/device.mojo"                 # a GPU binding module
ITER = "x_neighbors/iter_device.mojo"        # GPU module; fixture supplies its own host import
PY = "python/mojolearn/linear_model.py"      # a GPU-path estimator module


def _git(*a, env=None, inp=None):
    r = subprocess.run(["git", *a], capture_output=True, text=True, env=env, input=inp)
    if r.returncode != 0:
        raise RuntimeError(r.stderr)
    return r.stdout.strip()


def scratch(changes, base="HEAD"):
    """A commit on top of base with each path's text appended (str) or
    replaced (("=", text)) or a line removed (("-", line))."""
    base = _git("rev-parse", base)
    fd, idx = tempfile.mkstemp(prefix="nhr_idx_")
    os.close(fd)
    env = dict(os.environ, GIT_INDEX_FILE=idx)
    try:
        _git("read-tree", base, env=env)
        for path, ch in changes.items():
            try:
                old = _git("show", f"{base}:{path}") + "\n"
            except RuntimeError:
                old = ""
            if isinstance(ch, tuple) and ch[0] == "=":
                new = ch[1]
            elif isinstance(ch, tuple) and ch[0] == "-":
                lines = old.splitlines(keepends=True)
                hit = [i for i, ln in enumerate(lines) if ln.strip() == ch[1].strip()]
                assert hit, f"line not in {path}: {ch[1]}"
                del lines[hit[0]]
                new = "".join(lines)
            else:
                new = old + ch + "\n"
            blob = _git("hash-object", "-w", "--stdin", inp=new)
            _git("update-index", "--add", "--cacheinfo", f"100644,{blob},{path}", env=env)
        tree = _git("write-tree", env=env)
    finally:
        os.unlink(idx)
    return _git("commit-tree", tree, "-p", base, "-m", "no_host_routes scratch", inp="")


def run_tree(ref, baseline=None):
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rc = nhr.check_tree(ref, baseline, quiet=True)
    return rc, err.getvalue()


# (class, changes, a text the refusal must name)
BLOCKED = [
    ("HostExec in a GPU file", {DEV: "def _p0() raises:\n    var ex = HostExec()"}, "var ex = HostExec()"),
    ("host_parallelize call", {DEV: "def _p1(n: Int):\n    host_parallelize(_rows, n)"},
     "host_parallelize(_rows, n)"),
    ("host_parallelize_pool_env call", {DEV: "def _p2(n: Int):\n    host_parallelize_pool_env(_rows, n)"},
     "host_parallelize_pool_env(_rows, n)"),
    ("from core.host_parallel import", {DEV: "from core.host_parallel import host_parallelize_pool_env"},
     "from core.host_parallel import host_parallelize_pool_env"),
    ("aliased host_parallel call", {DEV: "from core.host_parallel import host_parallelize as hp\n"
                                         "def _p3(n: Int):\n    hp(_rows, n)"}, "hp(_rows, n)"),
    ("module import of host_parallel", {DEV: "from core import host_parallel"}, "from core import host_parallel"),
    ("stdlib parallelize import", {DEV: "from algorithm import parallelize"}, "from algorithm import parallelize"),
    ("stdlib parallelize call", {DEV: "def _p4(n_rows: Int):\n    parallelize[_rows](n_rows)"},
     "parallelize[_rows](n_rows)"),
    ("sync_parallelize call", {DEV: "def _p5(n_rows: Int):\n    sync_parallelize[_rows](n_rows)"},
     "sync_parallelize[_rows](n_rows)"),
    ("*_on_host call", {DEV: "def _p6(n: Int):\n    var l = do_labelling_on_host(n)"}, "do_labelling_on_host(n)"),
    ("*_host_rows call", {DEV: "def _p7(n: Int):\n    lu_solve_host_rows(0, 0, 0, n, 1, 0)"},
     "lu_solve_host_rows(0, 0, 0, n, 1, 0)"),
    ("import from a *_host module",
     {"x_decomp/audit_fixture_host.mojo": ("=", "def solve_blocks(n: Int):\n"
          "    for i in range(n):\n        consume(i)\n"),
      DEV: "from x_decomp.audit_fixture_host import solve_blocks"},
     "x_decomp/audit_fixture_host.mojo:solve_blocks"),
    ("import from a host/ module without _host", {DEV: "from cluster.host.host_cells import host_cells"},
     "cluster/host/host_cells.mojo:host_cells"),
    ("host helper not named *_on_host", {DEV: "from x_neighbors.cc_sparse import _cc_rounds\n"
                                              "def _p8(n: Int):\n    _cc_rounds(0, 0, 0, n)"},
     "_cc_rounds(0, 0, 0, n)"),
    ("MOJOLEARN_*HOST* switch", {DEV: 'def _p9():\n    var v = getenv("MOJOLEARN_XD_NEW_HOST")'},
     "MOJOLEARN_XD_NEW_HOST"),
    ("route switch without HOST", {DEV: 'def _p10():\n    var v = getenv("MOJOLEARN_PR_SPARSE")'},
     "MOJOLEARN_PR_SPARSE"),
    ("*HOST*_MIN threshold", {DEV: "comptime XD_NEW_HOST_MIN = 4096"}, "XD_NEW_HOST_MIN"),
    ("size threshold without HOST whose branch reaches host code",
     {"x_neighbors/audit_fixture_host.mojo": ("=", "def audit_walk(n: Int):\n"
          "    for i in range(n):\n        consume(i)\n"),
      ITER: "from x_neighbors.audit_fixture_host import audit_walk\n"
            "def _p11(ctx: DeviceContext, a: Int, lab: Int, info: Int, n: Int) raises:\n"
            "    if n < SMALL_N_CPU:\n"
            "        audit_walk(n)\n    else:\n        ctx.synchronize()"}, "if n < SMALL_N_CPU:"),
    ("route-table name", {PY: "_HOST_ALGOS = frozenset()"}, "_HOST_ALGOS = frozenset()"),
    ("download, synchronize, host loop, re-upload",
     {DEV: "def _p12(ctx: DeviceContext, dbuf: DeviceBuffer[DType.float32], hb: UnsafePointer[Float32], n: Int) raises:\n"
           "    ctx.enqueue_copy(dst_ptr=hb, src_buf=dbuf)\n    ctx.synchronize()\n"
           "    for i in range(n):\n        hb[i] = hb[i] * 2\n    ctx.enqueue_copy(dst_buf=dbuf, src_ptr=hb)"},
     "ctx.enqueue_copy(dst_ptr=hb, src_buf=dbuf)"),
    ("download, then a host fold over n (d2h-host-work)",
     {DEV: "def _p20(ctx: DeviceContext, d: DeviceBuffer[DType.float32], n: Int) raises -> Float32:\n"
           "    var h = download_f32(ctx, d, n)\n    var total = Float32(0.0)\n"
           "    for i in range(n):\n        total = total + h[i]\n    return total"},
     "for i in range(n):"),
    ("download, then a host walk that branches (d2h-host-work)",
     {DEV: "def _p21(ctx: DeviceContext, d: DeviceBuffer[DType.int32], hb: UnsafePointer[Int32], n_rows: Int) raises:\n"
           "    ctx.enqueue_copy(dst_ptr=hb, src_buf=d)\n    ctx.synchronize()\n"
           "    for i in range(n_rows):\n        if hb[i] == 0:\n            raise Error(\"bad\")"},
     "for i in range(n_rows):"),
    ("one-thread launch over n",
     {DEV: "def _p13(ctx: DeviceContext, x: UnsafePointer[Float32], n: Int) raises:\n"
           "    ctx.enqueue_function[_fold_kernel](x, Int32(n), grid_dim=1, block_dim=1)"},
     "ctx.enqueue_function[_fold_kernel](x, Int32(n), grid_dim=1, block_dim=1)"),
    ("one-block launch over n",
     {DEV: "def _p14(ctx: DeviceContext, x: UnsafePointer[Float32], n: Int) raises:\n"
           "    ctx.enqueue_function[_fold_kernel](\n        x, Int32(n),\n        grid_dim=1, block_dim=256,\n    )"},
     "ctx.enqueue_function[_fold_kernel]("),
    ("one-block launch spelled grid_dim=(1, 1, 1) over n (one-block-n)",
     {DEV: "def _p22(ctx: DeviceContext, x: UnsafePointer[Float32], n: Int) raises:\n"
           "    ctx.enqueue_function[_fold_kernel](\n        x, Int32(n),\n        grid_dim=(1, 1, 1), block_dim=(256, 1, 1),\n    )"},
     "ctx.enqueue_function[_fold_kernel]("),
    ("one block per class over n (block-per-n)",
     {DEV: "def _p24(ctx: DeviceContext, x: UnsafePointer[Float32], n_rows: Int, C: Int) raises:\n"
           "    ctx.enqueue_function[_mean_kernel](\n        x, Int32(n_rows), Int32(C),\n        grid_dim=(C, 1, 1), block_dim=(256, 1, 1),\n    )"},
     "ctx.enqueue_function[_mean_kernel]("),
    ("small-launch note naming an argument the launch does not pass",
     {DEV: "def _p23(ctx: DeviceContext, x: UnsafePointer[Float32], n: Int) raises:\n"
           "    ctx.enqueue_function[_fold_kernel](  # small-launch(d: feature count): the vector is d long\n"
           "        x, Int32(n),\n        grid_dim=(1, 1, 1), block_dim=(256, 1, 1),\n    )"},
     "ctx.enqueue_function[_fold_kernel]("),
    ("tid == 0 serial loop over n",
     {DEV: "def _p15_kernel(x: UnsafePointer[Float32], n: Int):\n    var tid = thread_idx.x\n"
           "    if tid == 0:\n        for i in range(n):\n            x[0] += x[i]"}, "if tid == 0:"),
    ("np.argsort in a GPU-path module", {PY: "order = np.argsort(y, kind='stable')"}, "np.argsort"),
    ("np.unique / np.cumsum", {PY: "cls, inv = np.unique(y, return_inverse=True)\nc = np.cumsum(w)"}, "np.cumsum"),
    ("np.bincount / np.sort", {PY: "b = np.bincount(codes)\ns = np.sort(v)"}, "np.bincount"),
    ("Python loop over rows", {PY: "for i in range(n_samples):\n    out[i] = f(X[i])"},
     "for i in range(n_samples):"),
    ("sklearn import", {PY: "from sklearn.isotonic import isotonic_regression"}, "from sklearn.isotonic"),
    ("comprehension over data in a def (py-data-loop)", {PY: "def _pz1(X):\n    return [v * 2 for v in X]"},
     "[v * 2 for v in X]"),
    ("numpy compute in a def (py-np-compute)", {PY: "def _pz2(X):\n    return np.mean(X, axis=0)"}, "np.mean"),
    ("sum() over data in a def (py-reduce)", {PY: "def _pz3(y):\n    return sum(y) / len(y)"}, "sum(y)"),
    ("array reduction method in a def (py-array-method)", {PY: "def _pz4(X):\n    return X.argmax()"},
     "X.argmax()"),
    ("glue note whose reason is under three words",
     {PY: "def _pz5(X):\n    for v in X:  # glue: fine\n        pass"}, "for v in X:"),
    ("Python thread pool", {PY: "from concurrent.futures import ThreadPoolExecutor"}, "ThreadPoolExecutor"),
    ("process pool", {PY: "import multiprocessing"}, "import multiprocessing"),
    ("forces MOJOLEARN_VENDOR=cpu", {PY: "os.environ['MOJOLEARN_VENDOR'] = 'cpu'"}, "MOJOLEARN_VENDOR"),
    ("load_host_module", {PY: "mod = load_host_module('x_linear')"}, "load_host_module"),
    ("import of a CPU-side Python module", {PY: "from ._gbdt_host import HostGBDT"}, "from ._gbdt_host"),
    ("host threads in a new file named *ghost*",
     {"x_decomp/fast_ghost_rows.mojo": ("=", "from core.host_parallel import host_parallelize\n\n"
                                        "def ghost_rows(n: Int):\n    def _r(t: Int):\n        pass\n"
                                        "    host_parallelize(_r, n)\n"),
      DEV: "from x_decomp.fast_ghost_rows import ghost_rows"}, "host_parallelize(_r, n)"),
    ("host threads in a new file under host/",
     {"x_decomp/host/rows2.mojo": ("=", "from core.host_parallel import host_parallelize\n\n"
                                   "def rows2(n: Int):\n    def _r(t: Int):\n        pass\n"
                                   "    host_parallelize(_r, n)\n"),
      "x_decomp/host/__init__.mojo": ("=", ""),
      DEV: "from x_decomp.host.rows2 import rows2"}, "host_parallelize(_r, n)"),
    ("host compute in a file named *multi_gpu*",
     {"x_decomp/multi_gpu_fold.mojo": ("=", "from core.host_parallel import host_parallelize\n\n"
                                       "def fold_rows(n: Int):\n    def _rows(t: Int):\n        var s = 0\n"
                                       "        for i in range(n):\n            s += i\n"
                                       "    host_parallelize(_rows, 4)\n"),
      DEV: "from x_decomp.multi_gpu_fold import fold_rows"}, "host_parallelize(_rows, 4)"),
    ("host_parallelize split over two lines", {DEV: "def _p16(n: Int):\n    host_parallelize\n        (_rows, n)"},
     "host_parallelize"),
    ("a checks/ module a GPU binding imports",
     {"cholesky/checks/trsm.mojo": "def _p17(n: Int):\n    host_parallelize(_rows, n)"}, "host_parallelize(_rows, n)"),
    ("re-adding an in-flight PR's line on another branch",
     {"x_linear/ops.mojo": "def _p18(groups: Int):\n    host_parallelize(group, groups)"},
     "host_parallelize(group, groups)"),
    ("download, then a host element copy loop (mojo-host-loop, not d2h-host-work)",
     {DEV: "def _p25(ctx: DeviceContext, d: DeviceBuffer[DType.float32], out: UnsafePointer[Float32], n: Int) raises:\n"
           "    var h = download_f32(ctx, d, n)\n    for i in range(n):\n        out[i] = h[i]"}, "[mojo-host-loop]"),
    ("a host finiteness scan of a binding argument (mojo-host-loop)",
     {DEV: "def _p26(p: UnsafePointer[Float32], count: Int) raises:\n"
           "    for i in range(count):\n        if not isfinite(p.unsafe_load(i)):\n"
           "            raise Error(\"non-finite\")"}, "for i in range(count):"),
    ("a host loop over a List of data (mojo-host-loop)",
     {DEV: "def _p27(values: List[Float32]) raises:\n"
           "    for value in values:\n        if value != value:\n            raise Error(\"nan\")"},
     "for value in values:"),
    ("a host copy into a List before a run (mojo-host-loop)",
     {DEV: "def _p28(x: UnsafePointer[Float32], n_query: Int, n_features: Int) raises -> List[Float32]:\n"
           "    var q = List[Float32]()\n    for i in range(n_query * n_features):\n        q.append(x[i])\n"
           "    return q^"}, "for i in range(n_query * n_features):"),
    ("small-loop note naming a bound the loop does not use",
     {DEV: "def _p29(p: UnsafePointer[Float32], n: Int) raises:\n"
           "    for i in range(n):  # small-loop(d: feature count): the vector is d long\n"
           "        p[i] = p[i] * 2"}, "for i in range(n):"),
    ("GPU-path Python calls a native host helper by name (py-native-host)",
     {PY: "def _pz6(arr):\n    return _native(\"all_finite_f32\")(arr._addr, arr.size)"}, "[py-native-host]"),
    ("GPU-path Python calls a native host helper as a binding attribute (py-native-host)",
     {PY: "def _pz7(b, w, n):\n    b.x_trees_samme_step(w, n)"}, "b.x_trees_samme_step(w, n)"),
    ("cpu-route note whose reason is under three words",
     {PY: "def _pz8(arr):\n    return _native(\"all_finite_f32\")(arr)  # cpu-route: fine"}, "[py-native-host]"),
    ("duplicating an existing debt line",
     {"cluster/estimator.mojo": "def _p19(groups: Int):\n    host_parallelize(_abs_sum_task, groups)"},
     "host_parallelize(_abs_sum_task, groups)"),
]

PASSES = [
    ("tools/ is not shipped", {"tools/route_helper.py": ("=", "_HOST_ALGOS = frozenset({'sgd'})\n")}),
    ("a comment", {DEV: "# HostExec was removed; host_parallelize(_rows, n) too"}),
    ("a docstring", {DEV: 'def _q0():\n    """Never uses HostExec or host_parallelize(_rows, n) here."""\n    pass'}),
    ("a multi-GPU driver (one host thread per device)",
     {"x_decomp/multi_gpu_drive.mojo": ("=", "from core.host_parallel import host_parallelize\n"
                                        "from std.gpu.host import DeviceContext\n\n"
                                        "def drive(n: Int) raises:\n    def _task(rank: Int):\n"
                                        "        var ctx = DeviceContext(device_id=rank)\n"
                                        "        ctx.synchronize()\n    host_parallelize(_task, n)\n"),
      DEV: "from x_decomp.multi_gpu_drive import drive"}),
    ("download, then a trace-only fold",
     {DEV: "def _q2(ctx: DeviceContext, mut trace: IdentityTrace, d: DeviceBuffer[DType.float32], n: Int) raises:\n"
           "    var h = download_f32(ctx, d, n)\n    if trace.enabled:\n        var t = Float32(0.0)\n"
           "        for i in range(n):\n            t = t + h[i]"}),
    ("download, then a loop over a small k",
     {DEV: "def _q3(ctx: DeviceContext, d: DeviceBuffer[DType.float32], n_features: Int) raises -> Float32:\n"
           "    var h = download_f32(ctx, d, n_features)\n    var t = Float32(0.0)\n"
           "    for j in range(n_features):  # small-loop(n_features: feature count): one sum per feature, read once\n"
           "        t = t + h[j]\n    return t"}),
    ("a reviewed small-launch note on a one-block launch over d-sized data",
     {DEV: "def _q4(ctx: DeviceContext, x: UnsafePointer[Float32], n: Int) raises:\n"
           "    ctx.enqueue_function[_fold_kernel](  # small-launch(n: parameter count): an L-BFGS vector phase, never rows\n"
           "        x, Int32(n),\n        grid_dim=(1, 1, 1), block_dim=(256, 1, 1),\n    )"}),
    ("a reviewed glue note on an argument loop",
     {PY: "def _qz1(params):\n    for k in params:  # glue: copies estimator keyword arguments\n        pass"}),
    ("scalar min/max and numpy allocation",
     {PY: "def _qz2(a, b):\n    return np.zeros(min(a, b) + max(a, 1))"}),
    ("a module-level table and a loop word in a string",
     {PY: "_QZ3 = {k: i for i, k in enumerate(('a', 'b'))}\n"
          "def _qz4():\n    raise ValueError('one value for each in X')"}),
    ("a host loop inside a kernel (thread work, not host work)",
     {DEV: "def _q5_kernel(x: UnsafePointer[Float32], n: Int):\n    var tid = Int(thread_idx.x)\n"
           "    for i in range(tid, n, 256):\n        x[i] = x[i] * 2"}),
    ("a host loop that drives the device per step (orchestration)",
     {DEV: "def _q6(ctx: DeviceContext, x: UnsafePointer[Float32], n_steps: Int) raises:\n"
           "    for s in range(n_steps):\n        x[s] = 0\n"
           "        ctx.enqueue_function[_fold_kernel](x, Int32(s), grid_dim=64, block_dim=256)"}),
    ("a host loop over a constant bound",
     {DEV: "def _q7(p: UnsafePointer[Float32]) raises:\n    for i in range(16):\n        p[i] = p[i] * 2"}),
    ("a host-column function in a GPU module",
     {DEV: "def _q8_host_oracle(p: UnsafePointer[Float32], n: Int) raises:\n"
           "    for i in range(n):\n        p[i] = p[i] * 2"}),
    ("a reviewed small-loop note on a host loop over a small bound",
     {DEV: "def _q9(p: UnsafePointer[Float32], n_classes: Int) raises:\n"
           "    for c in range(n_classes):  # small-loop(n_classes: class count): one prior per class, set once\n"
           "        p[c] = p[c] * 2"}),
    ("a reviewed cpu-route note on a native host helper call",
     {PY: "def _qz5(arr):\n    return _native(\"all_finite_f32\")(arr)  # cpu-route: CPU-only install verification digest"}),
    ("a native helper that walks arguments, not data (_NATIVE_GLUE)",
     {PY: "def _qz6(b, params):\n    return b.x_cnn_conv_shape(params)"}),
    ("a device-backed export of the base binding",
     {PY: "def _qz7(addr, n, bound):\n    return _native(\"check_indices_i64\")(addr, n, bound)"}),
    ("a module no GPU binding imports",
     {"x_decomp/unused_scratch.mojo": ("=", "def f(n: Int):\n    host_parallelize(_rows, n)\n")}),
]


def test_head_is_clean():
    rc, err = run_tree("HEAD")
    assert rc == 0, err


def test_blocked_classes():
    missed = []
    for name, changes, needle in BLOCKED:
        rc, err = run_tree(scratch(changes))
        if rc != 1 or needle not in err:
            missed.append(f"{name}: rc={rc} {err.strip()[:300]}")
    assert not missed, "\n".join(missed)


def test_passes():
    wrong = []
    for name, changes in PASSES:
        rc, err = run_tree(scratch(changes))
        if rc != 0:
            wrong.append(f"{name}: {err.strip()[:300]}")
    assert not wrong, "\n".join(wrong)


def test_removed_route_leaves_a_stale_row_until_pruned():
    # Supply our own debt so removing production host routes cannot invalidate
    # the regression (or tempt tests to restore a removed route).
    route = "var audit_fixture_exec = HostExec()"
    baseline = _git("show", f"HEAD:{nhr.BASELINE}") + "\n"
    baseline += "\t".join(["host-exec", "hostexec", "test-fixture", "debt", DEV, "0", route]) + "\n"
    debt = scratch({DEV: "def _audit_debt() raises:\n    " + route,
                    nhr.BASELINE: ("=", baseline)})
    rc, err = run_tree(debt)
    assert rc == 0, err
    tip = scratch({DEV: ("-", route)}, base=debt)
    rc, err = run_tree(tip)
    assert rc == 1 and "no longer match" in err, err
    fd, path = tempfile.mkstemp(prefix="nhr_bl_", suffix=".tsv")
    os.close(fd)
    try:
        with open(path, "w") as f:
            f.write(_git("show", f"{tip}:{nhr.BASELINE}") + "\n")
        with contextlib.redirect_stderr(io.StringIO()):
            nhr.prune_baseline(tip, path)
        rc, err = run_tree(tip, path)
        assert rc == 0, err
        before = len(nhr.load_baseline(_git("show", f"{tip}:{nhr.BASELINE}")))
        with open(path) as f:
            after = len(nhr.load_baseline(f.read()))
        assert after == before - 1, (before, after)
    finally:
        os.unlink(path)


def test_inflight_rows_serve_only_their_pr_head():
    # Real PR heads eventually merge or disappear from shallow CI fetches.
    # Construct known ancestry locally instead of assuming any live PR state.
    head = _git("rev-parse", "HEAD")
    pr = scratch({"tools/audit_ancestry_fixture.txt": ("=", "PR branch\n")}, base=head)
    child = scratch({"tools/audit_ancestry_fixture.txt": "descendant"}, base=pr)
    sibling = scratch({"tools/audit_ancestry_fixture.txt": ("=", "other branch\n")}, base=head)
    state = "inflight@" + pr
    assert nhr._inflight_ok(state, pr), "allowance must serve its PR head"
    assert nhr._inflight_ok(state, child), "allowance must serve descendants"
    assert not nhr._inflight_ok(state, head), "allowance cannot serve its parent"
    assert not nhr._inflight_ok(state, sibling), "allowance cannot serve another branch"
    assert not nhr._inflight_ok("inflight", pr), "unbound allowances must fail closed"
    rows = nhr.load_baseline(_git("show", f"HEAD:{nhr.BASELINE}"))
    assert not any(r["state"] == "inflight" for r in rows), "baseline contains an unbound allowance"


def test_native_host_exports_are_classified_by_code():
    tree = nhr.Tree("HEAD")
    ex = tree.host_exports()
    for name in ("all_finite_f32", "argmax_rows_f32", "gather_rows_bytes", "x_trees_samme_step"):
        assert name in ex, f"{name} loops on the host and must be a native host helper"
    for name in ("check_indices_i64", "unique_inverse", "cast_f64_to_f32"):
        assert name not in ex, f"{name} runs on the device and is not a native host helper"


def test_owed_rules_never_enter_the_baseline():
    row = "py-native-host\tnative-host\tL12\tdebt\tpython/mojolearn/_buffer.py\t0\tx"
    try:
        nhr.load_baseline(nhr._HDR + "\n" + row + "\n")
    except SystemExit as e:
        assert "owed" in str(e), e
    else:
        raise AssertionError("a baseline row of an owed rule must be refused")


def test_owed_findings_are_reported_not_refused():
    err = io.StringIO()
    with contextlib.redirect_stderr(err):
        rc = nhr.check_tree("HEAD")
    out = err.getvalue()
    assert rc == 0, out[-2000:]
    assert "OWED (not baseline debt)" in out and "mojo-host-loop" in out and "py-native-host" in out, out[-2000:]
    rows = nhr.owed_rows("HEAD").splitlines()
    assert rows[0] == "rule\tclass\tpath\tline\ttext" and len(rows) > 1


def test_fixing_an_owed_finding_leaves_no_stale_row():
    rows = [r.split("\t") for r in nhr.owed_rows("HEAD").splitlines()[1:]]
    py = [r for r in rows if r[0] == "py-native-host"]
    assert py, "HEAD carries no owed py-native-host finding to remove"
    path, no = py[0][2], int(py[0][3])
    text = _git("show", f"HEAD:{path}").splitlines()[no - 1]
    # replace the call line by a pass at the same indent: the owed row goes away
    lines = _git("show", f"HEAD:{path}").splitlines()
    ind = text[:len(text) - len(text.lstrip())]
    lines[no - 1] = ind + "pass"
    tip = scratch({path: ("=", "\n".join(lines) + "\n")})
    rc, err = run_tree(tip)
    assert "[py-native-host]" not in err, err[-1500:]


def test_cli_tree_and_diff_modes():
    tip = scratch({DEV: "def _p0() raises:\n    var ex = HostExec()"})
    r = subprocess.run([sys.executable, os.path.join(HERE, "no_host_routes.py"), "--tree", tip],
                       capture_output=True, text=True)
    assert r.returncode == 1 and "HostExec" in r.stderr and "by owner" in r.stderr, r.stderr
    r = subprocess.run([sys.executable, os.path.join(HERE, "no_host_routes.py"), "HEAD", tip,
                        "--main", "HEAD"], capture_output=True, text=True)
    assert r.returncode == 1 and "HostExec" in r.stderr, r.stderr
    r = subprocess.run([sys.executable, os.path.join(HERE, "no_host_routes.py"), "--tree", "HEAD"],
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as e:
                failed += 1
                print(f"FAIL {name}\n{e}")
    print(f"{'FAILED' if failed else 'OK'}: {failed} failed")
    sys.exit(1 if failed else 0)
