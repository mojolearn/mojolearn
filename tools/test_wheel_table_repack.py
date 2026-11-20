# SPDX-License-Identifier: Apache-2.0
"""tools/wheel_table_repack.py: the admitted table enters a built wheel with every other member, the
bindings included, byte-identical (checked by digest). Temporary wheels only."""
import json
import pathlib
import sys
import zipfile

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import wheel_table_repack as W  # noqa: E402

BINDING = b"\x7fELF fake binding bytes"


def make_wheel(path, table=b'{"records": []}', with_table=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    files = {"mojolearn/__init__.py": b"x = 1\n", "mojolearn/_mojolearn.so": BINDING,
             "mojolearn/identity_columns/COMMIT": b"c" * 40 + b"\n"}
    if with_table:
        files[W.TABLE_MEMBER] = table
    record = "".join("%s,%s,%d\n" % (n, W.record_hash(d), len(d)) for n, d in files.items())
    record += "mojolearn_core-0.8.66.dist-info/RECORD,,\n"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for n, d in files.items():
            z.writestr(n, d)
        z.writestr("mojolearn_core-0.8.66.dist-info/RECORD", record)
    return path


def test_only_the_table_and_its_record_line_change(tmp_path):
    wheel = make_wheel(tmp_path / "in" / "mojolearn_core-0.8.66-py3-none-manylinux_2_28_x86_64.whl")
    table = tmp_path / "table.json"
    table.write_text(json.dumps({"records": [1, 2]}))
    doc = W.repack(wheel, table, tmp_path / "out")
    out = pathlib.Path(doc["output"]["path"])
    assert out.name == wheel.name and doc["schema"] == W.SCHEMA
    with zipfile.ZipFile(out) as z:
        assert z.read(W.TABLE_MEMBER) == table.read_bytes()
        assert z.read("mojolearn/_mojolearn.so") == BINDING
        record = z.read("mojolearn_core-0.8.66.dist-info/RECORD").decode()
    assert "%s,%s,%d" % (W.TABLE_MEMBER, W.record_hash(table.read_bytes()), len(table.read_bytes())) in record
    assert doc["native_members"] == {"mojolearn/_mojolearn.so": W.sha256(BINDING)}
    assert doc["unchanged_members"] == 3 and doc["table"]["new_sha256"] == W.sha256(table.read_bytes())
    before, after = W.members(wheel), W.members(out)
    assert {n for n in before if before[n] != after[n]} == {W.TABLE_MEMBER, "mojolearn_core-0.8.66.dist-info/RECORD"}


def test_a_plugin_without_the_table_is_refused_and_ships_as_built(tmp_path):
    wheel = make_wheel(tmp_path / "in" / "mojolearn_nvidia-0.8.66.whl", with_table=False)
    table = tmp_path / "table.json"
    table.write_text("{}")
    with pytest.raises(W.RepackError, match="ship it as built"):
        W.repack(wheel, table, tmp_path / "out")


def test_an_unreadable_table_or_an_in_place_output_is_refused(tmp_path):
    wheel = make_wheel(tmp_path / "in" / "w.whl")
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    with pytest.raises(ValueError):
        W.repack(wheel, bad, tmp_path / "out")
    good = tmp_path / "t.json"
    good.write_text("{}")
    with pytest.raises(W.RepackError, match="overwrite"):
        W.repack(wheel, good, tmp_path / "in")


def test_main_writes_a_receipt(tmp_path, capsys):
    wheel = make_wheel(tmp_path / "in" / "w.whl")
    table = tmp_path / "t.json"
    table.write_text("{}")
    assert W.main([str(wheel), "--table", str(table), "--out", str(tmp_path / "out")]) == 0
    receipt = json.loads((tmp_path / "out" / "w.whl.repack.json").read_text())
    assert receipt["input"]["sha256"] == W.sha256(wheel.read_bytes())
    assert "REPACKED w.whl" in capsys.readouterr().out
