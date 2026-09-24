"""A release pass with nothing to check (no lane changed since its anchor)
writes an inherited record at this commit that qualifies as the next anchor."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import verify_lanes as vl  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FIX = "base,denormal,odd"


def _rev(spec):
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", spec], capture_output=True, text=True).stdout.strip()


def _anchor_record(base, commit, backend="cpu", complete=True):
    d = base / commit[:12] / backend
    d.mkdir(parents=True)
    (d / "column.json").write_text(json.dumps({"schema": "column", "cells": {"lane": "digest"}}))
    (d / "manifest.json").write_text(json.dumps(dict(commit=commit, lanes=["a", "b"], fixtures=FIX, repeats=1,
                                                     backend=backend, probe_group="core", metal_shards=1,
                                                     covers={"mode": "all"}, dirty_lanes=[])))
    (d / "run-summary.json").write_text(json.dumps(dict(complete=complete, validation_failures=[])))
    return d


def test_inherited_record_copies_the_column_and_becomes_the_next_anchor(tmp_path):
    anchor, head = _rev("HEAD~1"), _rev("HEAD")
    src = _anchor_record(tmp_path, anchor)
    assert vl.verified_anchor("cpu", FIX, base=str(tmp_path)) == anchor
    out = vl.inherit_record("cpu", anchor, {"mode": "since-record", "since": anchor}, FIX, {"x": "why"},
                            base=str(tmp_path), commit=head)
    assert out == str(tmp_path / head[:12] / "cpu")
    assert (Path(out) / "column.json").read_bytes() == (src / "column.json").read_bytes()
    m = json.loads((Path(out) / "manifest.json").read_text())
    assert m["commit"] == head and m["lanes"] == [] and m["runner"] == "inherited"
    assert m["inherited_from"]["commit"] == anchor and m["inherited_from"]["lanes"] == 2
    s = json.loads((Path(out) / "run-summary.json").read_text())
    assert s["complete"] is True and s["validation_failures"] == [] and s["inherited_from"] == anchor
    # the inherited record is now the newest finished pass and chains to its anchor
    assert vl.verified_anchor("cpu", FIX, base=str(tmp_path)) == head


def test_nothing_is_inherited_from_an_incomplete_or_missing_anchor(tmp_path, capsys):
    anchor, head = _rev("HEAD~1"), _rev("HEAD")
    assert vl.inherit_record("cpu", anchor, {"mode": "since-record", "since": anchor}, FIX, {},
                             base=str(tmp_path), commit=head) is None
    _anchor_record(tmp_path, anchor, complete=False)
    assert vl.inherit_record("cpu", anchor, {"mode": "since-record", "since": anchor}, FIX, {},
                             base=str(tmp_path), commit=head) is None
    assert not (tmp_path / head[:12]).exists()
    assert "nothing inherited" in capsys.readouterr().out
