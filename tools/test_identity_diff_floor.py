# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""`identity_break.py --diff` AND THE TABLE WHERE NOTHING WAS COMPARED.

Closed 2026-09-20. `--require-columns` defaulted to 0, and with no floor the
`bad` counter reads only MOVED, DIVERGENT, RELOAD-MOVED, BATCH_MOVED and
RLPAIR_MOVED. ONE-COLUMN (only one column hashed the cell), REFUSED (no
column did) and NOT-COMPARED (no column carries the part) all exited 0, so a
diff whose every row said "nothing was compared" reported success. One of the
37 executable `tools/*.sh` sites that run `--diff` passed the flag.

`--lanes` on its own was fully inert for the same reason: the check for a
named lane no JSON carries lived inside `if require_columns:`, so
`--lanes a-lane-nobody-hashed` printed a scoped table of zero rows and exited
0.

These go through the CLI, because the default being fixed is the CLI's; the
`diff()` function's own default is left at 0 so no programmatic caller moves.
"""
import json
import os
import subprocess
import sys
import tempfile

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IB = os.path.join(ROOT, "tools", "identity_break.py")
_H = "0123456789abcdef"


def _cell(h=_H, verdict="STABLE"):
    return dict(verdict=verdict, hashes=[h, h], parts=[{}],
                infer=[h, h], infer_verdict="STABLE",
                model=[h, h], model_verdict="STABLE")


def _column(vendor, cells):
    return dict(vendor=vendor, mode="identical", commit="c" * 40, package={},
                complete=True, cells=cells)


def _diff(tmp_path, columns, *extra):
    paths = []
    for i, col in enumerate(columns):
        p = tmp_path / f"col{i}.json"
        p.write_text(json.dumps(col))
        paths.append(str(p))
    r = subprocess.run([sys.executable, IB, "--diff"] + paths + list(extra),
                       capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def _two_agreeing():
    return [_column("box-a", {"ols/base": _cell(), "kmeans/base": _cell()}),
            _column("box-b", {"ols/base": _cell(), "kmeans/base": _cell()})]


def test_two_columns_that_agree_still_exit_zero(tmp_path):
    """The control. Without it the tests below prove only that `--diff` now
    refuses everything."""
    rc, out = _diff(tmp_path, _two_agreeing())
    assert rc == 0, out
    assert "summary: IDENTICAL=2" in out


def test_a_real_divergence_still_exits_one(tmp_path):
    a, b = _two_agreeing()
    b["cells"]["ols/base"] = _cell("ffffffffffffffff")
    rc, out = _diff(tmp_path, [a, b])
    assert rc == 1 and "summary: DIVERGENT=1, IDENTICAL=1" in out


def test_a_cell_only_one_column_hashed_fails(tmp_path):
    """THE TABLE THAT USED TO PASS: ONE-COLUMN means nothing was compared."""
    a, b = _two_agreeing()
    del b["cells"]["kmeans/base"]
    rc, out = _diff(tmp_path, [a, b])
    assert rc == 1, out
    assert "REQUIRE FAIL kmeans/base train: ONE-COLUMN rests on 1 real hash" in out


def test_every_cell_refused_on_both_columns_fails(tmp_path):
    a, b = _two_agreeing()
    for col in (a, b):
        for k in col["cells"]:
            col["cells"][k] = _cell(verdict="REFUSED")
    rc, out = _diff(tmp_path, [a, b])
    assert rc == 1, out
    assert "summary: REFUSED=2" in out
    assert "rests on 0 real hash" in out


def test_lanes_alone_can_report_a_lane_no_json_carries(tmp_path):
    """`--lanes` used to narrow the table and then be unable to say the
    narrowing left nothing, because the check sat inside
    `if require_columns:`."""
    rc, out = _diff(tmp_path, _two_agreeing(), "--lanes", "pca")
    assert rc == 1, out
    assert "REQUIRE FAIL pca: no JSON carries a cell for this lane" in out
    rc, out = _diff(tmp_path, _two_agreeing(), "--lanes", "pca", "--require-columns", "0")
    assert rc == 1, "the lane gap is a gap with or without a floor"


def test_the_floor_can_be_turned_off_by_name(tmp_path):
    """A caller that means `no floor` says so, and gets exactly the old
    behaviour. This is what keeps the change from being a check that cannot
    pass."""
    a, b = _two_agreeing()
    del b["cells"]["kmeans/base"]
    rc, out = _diff(tmp_path, [a, b], "--require-columns", "0")
    assert rc == 0, out
    assert "ONE-COLUMN=1" in out


def test_a_higher_floor_still_reads_what_it_always_did(tmp_path):
    rc, out = _diff(tmp_path, _two_agreeing(), "--require-columns", "3")
    assert rc == 1, out
    assert "REQUIRE FAIL: --require-columns 3 with 2 JSONs given" in out


def test_one_json_has_no_floor_and_says_so(tmp_path):
    """`--diff` with a single column compares nothing with anything; demanding
    two hashes of it would be a check that cannot pass."""
    rc, out = _diff(tmp_path, _two_agreeing()[:1])
    assert "the --require-columns floor does not apply" in out
    assert rc == 0


def test_owed_json_still_demands_an_explicit_count(tmp_path):
    """`--owed-json` reads a cell as OWED when it is short of the count, so
    the count has to be the one the caller meant, not a default."""
    p = tmp_path / "owed.json"
    rc, out = _diff(tmp_path, _two_agreeing(), "--owed-json", str(p))
    assert rc != 0
    assert "--owed-json needs an EXPLICIT --require-columns" in out


@pytest.mark.parametrize("record,columns,require,want_rc,token", [
    ("2026-09-14_166-lanes",
     ("apple-m4.json", "nvidia-h100-sm_90a.json", "amd-mi325x-gfx942.json"),
     "3", 1, "summary: DIVERGENT=1, IDENTICAL=1331"),
    ("2026-09-14_kmeans-sqrt-fix",
     ("apple-m4.json", "nvidia-h100-sm_90a.json", "amd-mi325x-gfx942.json"),
     "3", 0, "summary: IDENTICAL=72"),
])
def test_no_committed_record_moved(record, columns, require, want_rc, token):
    """NO COMMITTED CELL MOVED. The two comparisons the CPU identity gate
    makes over committed columns, with the counts it greps for."""
    base = os.path.join(ROOT, "bench", "results", "identity_break", record)
    if not os.path.isdir(base):
        pytest.skip(f"{base} is not in this checkout")
    r = subprocess.run([sys.executable, IB, "--diff"]
                       + [os.path.join(base, c) for c in columns]
                       + ["--require-columns", require],
                       capture_output=True, text=True)
    assert r.returncode == want_rc, r.stdout[-2000:]
    assert token in r.stdout
    assert f"require-columns {require} over" in r.stdout


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-q"]))
