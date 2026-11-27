# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""tools/generated_source_check.py, the one byte check of a leg's build/generated-sources/<run>/ copies, and its
three readers (packaging/linux/pack_wheel.py, tools/check_linux_release_qualification.py native_inventory,
tools/linux_surface_qualification.sh). A changed byte fails; nothing is skipped unread. No compiler.

    python3 -m pytest -q tools/test_generated_source_check.py
"""
import hashlib
import sys
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import generated_source_check as gsc  # noqa: E402
import generated_sources  # noqa: E402

RUN = "build/generated-sources/1795798918547688684-283"
INPUT = "core/native_archive_unicode_names.mojo"
TABLE = "core/archive_unicode_names_generated.mojo"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def tree(tmp_path, input_copy=b"pin\n", table_copy=b"table\n"):
    (tmp_path / "core").mkdir()
    (tmp_path / INPUT).write_bytes(b"pin\n")
    (tmp_path / "tools").mkdir()
    (tmp_path / "tools" / "gen_archive_unicode.py").write_bytes(b"gen\n")
    for rel, data in ((f"{RUN}/generated-inputs/{INPUT}", input_copy), (f"{RUN}/generated/{TABLE}", table_copy),
                      (f"{RUN}/generated-sources.json", b"{}\n")):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(data)
    return tmp_path


@pytest.fixture
def generator():
    """The tracked generator writes b"table\\n" (expected_sha stubbed: the real one needs the pinned tables)."""
    with mock.patch.object(generated_sources, "expected_sha", lambda rel, inputs, candidates=(): sha(b"table\n")):
        yield


def test_equal_copies_verify_and_other_evidence_is_not_a_copy(tmp_path, generator):
    root = tree(tmp_path)
    assert gsc.verify_copy(root, f"{RUN}/generated-inputs/{INPUT}")
    assert gsc.verify_copy(root, f"{RUN}/generated/{TABLE}")
    assert gsc.leg_generated_digest(root, f"{RUN}/generated-sources.json") is None
    assert not gsc.verify_copy(root, f"{RUN}/generated-sources.json")
    assert gsc.leg_generated_digest(root, INPUT) is None


def test_a_changed_input_byte_fails(tmp_path, generator):
    root = tree(tmp_path, input_copy=b"pin!\n")
    with pytest.raises(gsc.GeneratedCopyError, match="differs"):
        gsc.verify_copy(root, f"{RUN}/generated-inputs/{INPUT}")


def test_a_changed_output_byte_fails(tmp_path, generator):
    root = tree(tmp_path, table_copy=b"tablE\n")
    with pytest.raises(gsc.GeneratedCopyError, match="differs"):
        gsc.verify_copy(root, f"{RUN}/generated/{TABLE}")


def test_unknown_or_unbacked_copies_refuse(tmp_path, generator):
    root = tree(tmp_path)
    with pytest.raises(gsc.GeneratedCopyError, match="not a generated source"):
        gsc.leg_generated_digest(root, f"{RUN}/generated/core/other.mojo")
    with pytest.raises(gsc.GeneratedCopyError, match="not in the source root"):
        gsc.leg_generated_digest(root, f"{RUN}/generated-inputs/core/gone.mojo")


def test_native_inventory_checks_the_copies_and_does_not_list_them(tmp_path, generator):
    import check_linux_release_qualification as gate
    root = tree(tmp_path)
    listed = {rel for rel, _ in gate.native_inventory(root)}
    assert INPUT in listed and not any(r.startswith(gsc.PREFIX) for r in listed)
    (root / f"{RUN}/generated/{TABLE}").write_bytes(b"tablE\n")
    with pytest.raises(ValueError, match="Generated source copy differs"):
        gate.native_inventory(root)


def test_every_reader_uses_the_one_helper():
    pack = (ROOT / "packaging/linux/pack_wheel.py").read_text()
    assert "generated_source_check.leg_generated_digest(source_root, name)" in pack
    assert "def _leg_generated_digest" not in pack
    assert "generated_source_check.verify_copy(root, rel, path)" in (
        ROOT / "tools/check_linux_release_qualification.py").read_text()
    assert "generated_source_check.verify_copy(root, rel, path)" in (
        ROOT / "tools/linux_surface_qualification.sh").read_text()
