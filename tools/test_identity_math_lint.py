# SPDX-License-Identifier: Apache-2.0
"""tools/identity_math_lint.py on small sources (no build)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import identity_math_lint as lint  # noqa: E402

SRC = '''from std.math import sqrt, exp as exp, floor
import std.math as m2


def kernel_body(x: Float32) -> Float32:
    """sqrt(x) in a docstring is not a call."""
    # exp(x) in a comment is not a call
    var s = "log(x)"
    return sqrt(x) + exp(x) + floor(x)


def helper(x: Float32) -> Float32:
    def inner(y: Float32) -> Float32:
        return sqrt(y)
    return inner(x) + my_sqrt(x) + v.sqrt(x)
'''


def test_sites_are_calls_of_imported_math_with_their_function():
    found = lint.sites("kde/impl/k.mojo", SRC)
    assert [(ln, fn, op) for _, ln, fn, op in found] == [(9, "kernel_body", "exp"), (9, "kernel_body", "sqrt"),
                                                         (14, "inner", "sqrt")]


def test_qualified_and_parenthesized_imports():
    src = "from std.math import (\n    log1p,\n    tanh,\n)\nimport math\n\ndef f(x: Float32) -> Float32:\n" \
          "    return log1p(x) + tanh(x) + math.erf(x)\n"
    assert [op for *_, op in lint.sites("a.mojo", src)] == ["erf", "log1p", "tanh"]
    assert lint.sites("a.mojo", "def f(x: Float32) -> Float32:\n    return sqrt(x)\n") == []  # not imported


def test_scope_excludes_checks_oracles_bench_and_the_seam_library():
    assert lint.in_scope("x_prep/py2mojo.mojo")
    for path in ("checks/numerics.mojo", "gbdt/checks/x.mojo", "gbdt/host/gbdt_oracle.mojo", "bench/a.mojo",
                 "experiments/x.mojo", "docs/a.mojo", "x_prep/py2mojo.py"):
        assert not lint.in_scope(path), path


def test_allow_list_judges_unlisted_and_stale():
    found = lint.sites("kde/impl/k.mojo", SRC)
    allow, bad = lint.read_allow(
        "# comment\n"
        "kde/impl/k.mojo:kernel_body:sqrt  # SAFE-FAST: only under comptime FAST (k.mojo:3)\n"
        "kde/impl/k.mojo:kernel_body:exp  # SAFE-FAST: same branch\n"
        "kde/impl/k.mojo:gone:log  # SAFE-EXACT: removed function\n"
        "kde/impl/k.mojo:inner:sqrt  # FINE: no class\n"
        "kde/impl/k.mojo:inner:sqrt\n")
    assert len(bad) == 2
    unlisted, stale = lint.judge(found, allow)
    assert [(s[2], s[3]) for s in unlisted] == [("inner", "sqrt")]
    assert stale == [("kde/impl/k.mojo", "gone", "log")]


def test_the_repository_allow_list_is_well_formed():
    text = (lint.ROOT / lint.ALLOW).read_text()
    allow, bad = lint.read_allow(text)
    assert bad == [] and allow


def test_the_mode_policy_table_is_out_of_scope_and_its_rows_are_known():
    assert not lint.in_scope(lint.POLICY_MODULE)
    assert (lint.ROOT / lint.POLICY_MODULE).is_file()
    table = (lint.ROOT / lint.POLICY_MODULE).read_text()
    for bit, row in enumerate(lint.POLICY_ROWS):
        assert f"comptime POLICY_ROW_{row.upper()} = {bit}\n" in table, row


def test_a_safe_row_line_names_a_row_of_the_table():
    allow, bad = lint.read_allow(
        "a/k.mojo:f:exp  # SAFE-ROW: vendor_math: else arm of comptime if not ROW_VENDOR_MATH\n"
        "a/k.mojo:g:exp  # SAFE-ROW: no_such_row: reason\n"
        "a/k.mojo:h:exp  # SAFE-ROW: reason with no row\n")
    assert list(allow) == [("a/k.mojo", "f", "exp")]
    assert len(bad) == 2
