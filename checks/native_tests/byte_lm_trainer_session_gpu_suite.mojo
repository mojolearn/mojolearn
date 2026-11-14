# SPDX-License-Identifier: Apache-2.0
"""The byte LM trainer session units on the image's GPU column.

The units run in order: every resident session of the image shares the one
device the trainer binds, and each unit's training steps keep their kernels'
own parallelism on it."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from checks.vendor import COMPILED_VENDOR
from core.byte_lm_entry_table import ByteLmEntryKey, byte_lm_vendor_word
from training.byte_lm_glue_entry import publish_byte_lm_entries
from checks.native_tests.byte_lm_trainer_session_cases import BYTE_LM_TRAINER_SESSION_CASES, case, require


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var vendor = byte_lm_vendor_word(COMPILED_VENDOR)
    require(vendor > 0, "byte_lm_trainer_session_gpu needs a GPU build (cuda, hip or metal)")
    # The publication `_mojolearn_byte_lm` makes at module init.
    publish_byte_lm_entries()
    var failed = 0
    for unit in range(BYTE_LM_TRAINER_SESSION_CASES):
        try:
            case(ByteLmEntryKey(Int(GLOBAL_NUMERIC_MODE), vendor), unit)
        except error:
            failed += 1
            print("FAIL byte_lm_trainer_session_gpu unit", unit, String(error))
    require(failed == 0, "byte_lm_trainer_session_gpu failed " + String(failed) + " units")
    print("PASS byte_lm_trainer_session_gpu:", BYTE_LM_TRAINER_SESSION_CASES, "resident session units, in order")
