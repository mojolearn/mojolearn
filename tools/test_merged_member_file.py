# SPDX-License-Identifier: Apache-2.0
"""merged_bindings.member_file: tools name one member's merged native file from the manifest or an installed
index, never by a hard-coded group stem (the 2026-11-04 host split, docs/build/HOT_GROUPS.md)."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import merged_bindings as mb  # noqa: E402


def test_manifest_lookup_after_the_host_split():
    assert mb.member_file("host", "_mojolearn_tokenizer_host") == "_mojolearn_native_tokenizer_host.so"
    assert mb.member_file("host", "_mojolearn_runtime_classical_host") == "_mojolearn_native_classical_host.so"
    assert mb.member_file("host", "_mojolearn_core_host") == "_mojolearn_native_host.so"
    assert mb.member_file("host", "_mojolearn_x_metrics_host") == "_mojolearn_native_series_host.so"
    # 2026-11-07 host split (docs/build/HOT_GROUPS.md section 6)
    assert mb.member_file("host", "_mojolearn_runtime_trees_host") == "_mojolearn_native_classical_trees_host.so"
    assert mb.member_file("host", "_mojolearn_runtime_linear_host") == "_mojolearn_native_classical_prep_host.so"
    assert mb.member_file("host", "_mojolearn_runtime_neural_host") == "_mojolearn_native_neural_doors_host.so"
    assert mb.member_file("host", "_mojolearn_runtime_manifold_host") == "_mojolearn_native_orchestration_host.so"


def test_index_lookup_and_ambiguity():
    index = {"format": 1, "files": {"_a_host.so": ["_m1", "_m2"], "_b_host.so": ["_m3"]}}
    assert mb.member_file("host", "_m3", index=index) == "_b_host.so"
    with pytest.raises(ValueError):
        mb.member_file("host", "_absent", index=index)
    with pytest.raises(ValueError):
        mb.member_file("host", "_m1", index={"files": {"x.so": ["_m1"], "y.so": ["_m1"]}})


def test_host_groups_split_along_their_clusters():
    groups = mb.manifest()["dirs"]["host"]["groups"]
    assert groups["_mojolearn_native_classical_host"] == ["_mojolearn_runtime_classical_host"]
    assert groups["_mojolearn_native_tokenizer_host"] == ["_mojolearn_tokenizer_host"]
    split = {"_mojolearn_native_classical_trees_host": ["_mojolearn_runtime_trees_host"],
             "_mojolearn_native_classical_family_host": ["_mojolearn_runtime_family_host"],
             "_mojolearn_native_classical_prep_host": ["_mojolearn_runtime_prep_host", "_mojolearn_runtime_linear_host"],
             "_mojolearn_native_classical_decomp_host": ["_mojolearn_runtime_decomp_host"],
             "_mojolearn_native_catalog_host": ["_mojolearn_runtime_catalog_host"],
             "_mojolearn_native_xseq_host": ["_mojolearn_runtime_xseq_host"],
             "_mojolearn_native_byte_trainer_host": ["_mojolearn_runtime_byte_trainer_host"],
             "_mojolearn_native_parallel_host": ["_mojolearn_runtime_parallel_host"],
             "_mojolearn_native_neural_doors_host": ["_mojolearn_runtime_neural_host"],
             "_mojolearn_native_orchestration_host": ["_mojolearn_byte_lm_host", "_mojolearn_runtime_model_selection_host",
                                                      "_mojolearn_runtime_manifold_host"]}
    assert {stem: groups[stem] for stem in split} == split
    every = [m for names in groups.values() for m in names]
    assert len(every) == len(set(every)) and all(stem.endswith("_host") and stem not in every for stem in groups)


def test_qualify_reads_the_installed_index(tmp_path):
    import qualify_release_host_boundary as q
    host = tmp_path / "mojolearn/host"
    host.mkdir(parents=True)
    (host / "_mojolearn_merged.json").write_text(json.dumps(
        {"format": 1, "files": {"_mojolearn_native_tokenizer_host.so": ["_mojolearn_tokenizer_host"],
                                "_mojolearn_native_series_host.so": ["_mojolearn_tsa_host"]}}))
    assert q.host_file(tmp_path) == "_mojolearn_native_tokenizer_host.so"


def test_incremental_gate_host_comes_from_the_manifest():
    import release_incremental_gate as g
    assert g.HOST == "mojolearn/host/" + mb.member_file("host", "_mojolearn_tokenizer_host")
