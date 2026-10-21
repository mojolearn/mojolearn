# SPDX-License-Identifier: Apache-2.0
"""Static coverage controls only; no product import or native compilation."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("native_inventory", HERE / "inventory.py")
INVENTORY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(INVENTORY)


def manifest():
    return json.loads((HERE / "manifest.json").read_text())


def test_all_explicit_root_and_expansion_exports_remain_visible():
    expected = set()
    paths = [ROOT / "python/mojolearn/__init__.py", *sorted((ROOT / "python/mojolearn").glob("_expansion_*.py"))]
    for path in paths:
        expected.update("mojolearn." + name for name in INVENTORY.explicit_exports(path))
    rows = {row["algorithm"] for row in manifest()["inventory"]}
    assert not expected - rows


def test_ready_rows_have_exact_fixture_witnesses_and_explicit_scopes():
    data = manifest()
    cases = {case["id"]: case for case in data["cases"]}
    for row in data["inventory"]:
        if row["status"] == "ready":
            assert row["cases"], row
            for name in row["cases"]:
                assert row["algorithm"] in cases[name]["algorithms"], row
        else:
            assert row["reason"], row
    for case in cases.values():
        assert case["scope"] in {"native-core", "native-estimator"}
        assert case["coverage"]
        assert case["marker"] in (ROOT / case["source"]).read_text()
        assert set(case["backends"]) <= {"cpu", "metal", "cuda", "hip"}
        assert case["canonical_algorithms"]


def test_aliases_have_proven_equal_origin_and_do_not_inflate_census():
    data = manifest()
    rows = {row["algorithm"]: row for row in data["inventory"]}
    for row in rows.values():
        if "alias_of" in row:
            canonical = rows[row["alias_of"]]
            assert "alias_of" not in canonical
            assert row["symbol_origin"] == canonical["symbol_origin"]
    canonicals = [row for row in rows.values() if "alias_of" not in row]
    assert len(rows) == data["census"]["public_export_rows"]
    assert len(rows) - len(canonicals) == data["census"]["alias_rows"]
    assert sum(row["status"] != "not_algorithm" for row in canonicals) == data["census"]["canonical_algorithm_entries"]
    assert data["census"]["canonical_algorithm_entries"] < len(rows)


def test_boundary_blockers_survive_native_engine_coverage():
    rows = {row["algorithm"]: row for row in manifest()["inventory"]}
    for name in ("mojolearn.Sequential", "mojolearn.ops.Session"):
        assert rows[name]["status"] == "ready"
        assert "CPython" in rows[name]["boundary_blocker"]
    for name in ("mojolearn.models.CausalLM", "mojolearn.models.ParallelCausalLM", "mojolearn.SambaStack"):
        assert rows[name]["status"] == "blocked"
        assert "CPython" in rows[name]["reason"]


def test_canonicalization_is_reproducible_without_importing_product():
    data = manifest()
    assert INVENTORY.canonicalize(ROOT, data) == manifest()
