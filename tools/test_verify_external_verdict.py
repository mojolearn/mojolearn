# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""THE OUTSIDE PARTY'S VERDICT, AND WHETHER IT CAN SAY FAIL.

`tools/verify_external.sh` is what someone who does not trust us runs to
reproduce a column on their own box. Until 2026-09-20 its verdict could not
fail three ways at once:

  * `python3 ... --diff ... | tee FILE; DIFF_RC=$?` reads TEE's status, so a
    diff that exited 1 was recorded as 0;
  * `re.search(r"summary: ")` takes the FIRST match, which is the train
    table, so a DIVERGENT under `summary (infer/model):`, `summary (batch):`
    or `summary (rlpair):` never reached the verdict;
  * the floor was `ident > 0`, so ONE IDENTICAL cell among forty ONE-COLUMN
    prints read PASS.

Each test below builds the input that SHOULD fail. They run the verdict
script exactly as the shell runs it, lifted out of the heredoc, so they
cannot drift from the shipped text.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

SH = Path(__file__).resolve().parent / "verify_external.sh"


def _verdict_source():
    m = re.search(r"<<'PY'\n(.*?)\nPY\n", SH.read_text(), re.S)
    assert m, "verify_external.sh no longer carries its verdict heredoc"
    return m.group(1)


def _verdict(tmp_path, column, diff_text, diff_rc=0):
    col = tmp_path / "column.json"
    col.write_text(json.dumps(column))
    txt = tmp_path / "diff.txt"
    txt.write_text(diff_text)
    script = tmp_path / "verdict.py"
    script.write_text(_verdict_source())
    p = subprocess.run([sys.executable, str(script), str(col), str(txt), str(diff_rc)],
                       capture_output=True, text=True)
    return p.returncode, p.stdout + p.stderr


_CELL = dict(verdict="STABLE", hashes=["a" * 16, "a" * 16],
             infer_verdict="STABLE", model_verdict="STABLE")


def _column(n=3, **kw):
    col = dict(vendor="probe-box", complete=True,
               cells={f"lane{i}/base": dict(_CELL) for i in range(n)})
    col.update(kw)
    return col


def _clean_diff(n=3):
    return (f"summary: IDENTICAL={n}\n\n"
            f"summary (infer/model): IDENTICAL={2 * n}\n"
            f"summary (batch): IDENTICAL={n}\n"
            "summary (rlpair): \n")


def test_a_clean_column_still_passes(tmp_path):
    """The control. Without it the three tests below prove only that the
    verdict says FAIL to everything."""
    rc, out = _verdict(tmp_path, _column(), _clean_diff())
    assert rc == 0, out
    assert "VERDICT: PASS" in out


def test_a_divergent_train_cell_still_fails(tmp_path):
    """The one arm that worked before this change."""
    rc, out = _verdict(tmp_path, _column(),
                       "summary: IDENTICAL=2, DIVERGENT=1\n\nsummary (infer/model): IDENTICAL=6\n", 1)
    assert rc == 1
    assert "VERDICT: FAIL" in out


def test_a_divergent_infer_model_cell_fails(tmp_path):
    """THE INPUT THAT USED TO PASS: the train table is spotless and the
    held-out rows diverged. `re.search` stopped at the first summary line."""
    text = (f"summary: IDENTICAL=3\n\n"
            "summary (infer/model): DIVERGENT=6\n"
            "summary (batch): IDENTICAL=3\n")
    rc, out = _verdict(tmp_path, _column(), text)
    assert rc == 1, out
    assert "summary (infer/model) carries DIVERGENT=6" in out


@pytest.mark.parametrize("group,body", [
    ("batch", "BATCH_MOVED=3"),
    ("rlpair", "RLPAIR_MOVED=4"),
    ("infer/model", "RELOAD-MOVED=2"),
    ("infer/model", "MOVED=2"),
])
def test_every_later_summary_line_can_fail_the_verdict(tmp_path, group, body):
    text = f"summary: IDENTICAL=3\n\nsummary ({group}): {body}\n"
    rc, out = _verdict(tmp_path, _column(), text)
    assert rc == 1, out
    assert f"summary ({group}) carries" in out


def test_one_identical_among_forty_one_column_fails(tmp_path):
    """THE INPUT THAT USED TO PASS: `ident > 0` was the whole floor."""
    text = ("summary: IDENTICAL=1, ONE-COLUMN=40\n\n"
            "summary (infer/model): ONE-COLUMN=82\n")
    rc, out = _verdict(tmp_path, _column(41), text)
    assert rc == 1, out
    assert "ONE-COLUMN=40" in out


def test_a_column_the_diff_never_compared_fails(tmp_path):
    """Forty cells in the column, three rows in the table. The count has to
    be over the column, not over whatever the diff happened to print."""
    rc, out = _verdict(tmp_path, _column(40), _clean_diff(3))
    assert rc == 1, out
    assert "cells IDENTICAL" in out


def test_a_nonzero_diff_exit_fails(tmp_path):
    """What `| tee` used to swallow before the status ever reached here."""
    rc, out = _verdict(tmp_path, _column(), _clean_diff(), 1)
    assert rc == 1, out
    assert "the diff itself exited 1" in out


def test_neither_run_nor_diff_output_goes_through_a_pipe():
    """`$?` after `cmd | tee FILE` is TEE's. This is `#!/bin/sh`, where
    neither `pipefail` nor `PIPESTATUS` is portable, so the shipped text must
    not pipe either command at all."""
    text = SH.read_text()
    piped = [l for l in text.splitlines()
             if "identity_break.py" in l and "| tee" in l]
    assert not piped, piped
    assert "DIFF_RC=$?" in text and "RUN_RC=$?" in text


def test_an_incomplete_or_partial_column_fails(tmp_path):
    """A truncated column credits nothing."""
    rc, out = _verdict(tmp_path, _column(complete=False), _clean_diff())
    assert rc == 1 and "INCOMPLETE" in out, out
    rc, out = _verdict(tmp_path, _column(partial_column=True, parts_omitted=["batch"]),
                       _clean_diff())
    assert rc == 1 and "PARTIAL" in out, out


def test_an_older_record_whose_parts_are_absent_still_passes(tmp_path):
    """NOT-COMPARED and N/A are weaker results, not failures: a record that
    predates a part must still be reproducible."""
    text = ("summary: IDENTICAL=3\n\n"
            "summary (infer/model): IDENTICAL=4, N/A=2\n"
            "summary (batch): NOT-COMPARED=3\n"
            "summary (rlpair): \n")
    rc, out = _verdict(tmp_path, _column(), text)
    assert rc == 0, out


def test_lane_revision_drops_are_subtracted_not_ignored(tmp_path):
    """`identity_break.diff` drops the cells of a lane a column hashed at an
    older LANE_REVISIONS and says so. Those cells are absent by the diff's
    own decision, so the floor is over what remained -- but the drop is
    printed, and a drop the diff did NOT announce still fails."""
    announced = ("NOTE: column probe-box hashed kmeans at an older lane revision "
                 "(LANE_REVISIONS); its 2 cell(s) there are not compared and read as absent\n"
                 + _clean_diff(3))
    rc, out = _verdict(tmp_path, _column(5), announced)
    assert rc == 0, out
    assert "dropped 2 of this column's 5 cells" in out
    rc, out = _verdict(tmp_path, _column(5), _clean_diff(3))
    assert rc == 1, out
