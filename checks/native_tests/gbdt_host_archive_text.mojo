# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The host GBDT model-text reader behind `mojolearn._gbdt_host.parse_model_text`,
on the literal texts of test_gbdt_host_modes.py
`test_ctr_and_tensor_ctr_records_without_tables_refuse` (lane
tests-to-mojo-trees, 2026-11-16; the Python function is deleted).

The door (`bindings/host_tree_native_door.mojo`, operation `host_tree_parse`)
converts the text to a native String and calls
`core.host_archive_gbdt.host_archive_gbdt_text` on it; this check calls that
entry with the same texts. TEXT is one float feature with one border and no
trees; cases 0-2 insert one record after `losses 0`.

  0  `ctr_columns 1`                    -> refused, "1 CTR columns and 0 CTR tables"
  1  `tensor_ctr_registry 1 1`          -> refused, "declares 1 tables and carries 0"
  2  `feature_freq_tensor 2 hash_hi 0`  -> refused, "before its registry"
  3  TEXT itself                        -> parses, n_features 1

The entry is host code on both columns (it reads text, no device).
"""
from core.host_archive_gbdt import host_archive_gbdt_text
from std.testing import TestSuite, assert_true as require
from checks.native_tests.testsuite_cases import check_cases

comptime GBDT_HOST_ARCHIVE_TEXT_CASES = 4

comptime _HEAD = "format mojolearn-model 2\nfeatures 1 0\ntrees 0\nlosses 0\n"
comptime _FEATURE = "feature 0 folds 1 one_hot 0 type float nan as_is borders 1 0.5/3f000000\n"


def _refusal(record: String, needle: String, label: String) raises:
    var text = String(_HEAD) + record + "\n" + String(_FEATURE)
    var message = String("")
    try:
        _ = host_archive_gbdt_text(text)
    except error:
        message = String(error)
    require(message.byte_length() > 0, label + ": the record was accepted: " + record)
    require(needle in message, label + ": expected '" + needle + "' in: " + message)


def run_case(index: Int) raises:
    var label = "gbdt_host_archive_text case " + String(index)
    if index == 0:
        _refusal("ctr_columns 1", "1 CTR columns and 0 CTR tables", label)
    elif index == 1:
        _refusal("tensor_ctr_registry 1 1", "declares 1 tables and carries 0", label)
    elif index == 2:
        _refusal("feature_freq_tensor 2 hash_hi 0", "before its registry", label)
    else:
        var parsed = host_archive_gbdt_text(String(_HEAD) + String(_FEATURE))
        require(parsed.metadata["n_features"] == 1, label + ": n_features is not 1")


def test_cases() raises:
    check_cases("gbdt_host_archive_text", GBDT_HOST_ARCHIVE_TEXT_CASES, run_case)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
