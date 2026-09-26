"""tools/check_no_recursion_around_kernels.py on small synthetic sources.

    .pixi/envs/test/bin/python -m pytest tools/tests/test_check_no_recursion_around_kernels.py -q

Each case feeds `find_cycles` a dict of in-memory files, so nothing is
compiled and the repo is not read (except by the last test, which runs the
real check and asserts only that it read files and did not crash).
"""
import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "check_no_recursion_around_kernels",
    REPO / "tools" / "check_no_recursion_around_kernels.py",
)
chk = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chk)

# The 57-line reproducer's shape: op -> _rows -> op, op calls matmul.
REPRO = '''
from max.gpu.host import DeviceBuffer, DeviceContext
from linalg.matmul import matmul


def op(ctx: DeviceContext, n: Int, distribute: Bool = True) raises:
    if distribute and n > 1000000:
        _rows(ctx, n)
        return
    matmul[transpose_b=True, target="gpu"](tz, tx, ty, ctx)


def _rows(ctx: DeviceContext, n: Int) raises:
    op(ctx, n - 1, False)
'''


def _offending(sources):
    return [(m, libs) for m, libs, _ in chk.find_cycles(sources) if libs]


def test_reproducer_shape_is_flagged():
    found = _offending({"svmlike/k.mojo": REPRO})
    assert len(found) == 1
    members, libs = found[0]
    assert [n for _, n in members] == ["_rows", "op"]
    assert libs == ["linalg.matmul.matmul"]


def test_no_cycle_no_finding():
    src = REPRO.replace("    op(ctx, n - 1, False)\n", "    pass\n")
    assert _offending({"a.mojo": src}) == []


def test_direct_self_recursion_is_flagged():
    src = REPRO.replace("        _rows(ctx, n)\n", "        op(ctx, n - 1, False)\n")
    found = _offending({"a.mojo": src})
    assert [[n for _, n in m] for m, _ in found] == [["op"]]


def test_library_kernel_reached_through_an_imported_helper():
    helper = '''
from linalg.matmul import matmul


def gemm(ctx: DeviceContext) raises:
    matmul[transpose_b=True, target="gpu"](tz, tx, ty, ctx)
'''
    user = '''
from core.gemm import gemm


def fit(ctx: DeviceContext, again: Bool) raises:
    if again:
        init(ctx)
    gemm(ctx)


def init(ctx: DeviceContext) raises:
    fit(ctx, False)
'''
    found = _offending({"core/gemm.mojo": helper, "cluster/k.mojo": user})
    assert len(found) == 1
    assert found[0][1] == ["linalg.matmul.matmul"]


def test_cycle_reaching_only_our_kernels_is_not_offending():
    src = '''
def op(ctx: DeviceContext, again: Bool) raises:
    if again:
        op(ctx, False)
    ctx.enqueue_function[my_kernel](grid_dim=1, block_dim=1)
'''
    cycles = chk.find_cycles({"a.mojo": src})
    assert len(cycles) == 1 and cycles[0][1] == [] and cycles[0][2] is True


def test_names_in_strings_and_comments_are_not_calls():
    src = '''
from linalg.matmul import matmul


def check_x(ctx: DeviceContext) raises:
    # check_x(ctx) again would recurse
    print("check_x(", "...)")
    matmul[target="gpu"](a, b, c, ctx)
'''
    assert chk.find_cycles({"a.mojo": src}) == []


def test_closure_calls_belong_to_the_parent():
    src = '''
from linalg.matmul import matmul


def op(ctx: DeviceContext, again: Bool) raises:
    def task(i: Int) raises {mut ctx}:
        op(ctx, False)
    if again:
        task(0)
    matmul[target="gpu"](a, b, c, ctx)
'''
    found = _offending({"a.mojo": src})
    assert [[n for _, n in m] for m, _ in found] == [["op"]]


def test_reviewed_cycle_is_reported_but_passes(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(chk, "ROOT", str(tmp_path))
    (tmp_path / "k.mojo").write_text(REPRO)
    assert chk.main([str(tmp_path / "k.mojo")]) == 1
    assert "FAIL" in capsys.readouterr().out
    monkeypatch.setitem(chk.REVIEWED, ("k.mojo", ("_rows", "op")), "gated")
    assert chk.main([str(tmp_path / "k.mojo")]) == 0
    assert "REVIEWED" in capsys.readouterr().out


def test_reading_nothing_is_not_a_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(chk, "ROOT", str(tmp_path))
    assert chk.main([str(tmp_path / "missing.mojo")]) == 2


def test_runs_on_the_repo():
    out = subprocess.run(
        [sys.executable, str(REPO / "tools" / "check_no_recursion_around_kernels.py")],
        capture_output=True, text=True,
    )
    assert out.returncode in (0, 1), out.stderr
    assert " files, " in out.stdout
