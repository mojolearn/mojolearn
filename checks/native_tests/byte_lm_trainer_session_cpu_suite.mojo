# SPDX-License-Identifier: Apache-2.0
"""The byte LM trainer session units on the host column (vendor word 0),
each on its own sessions, run across host cores."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from core.byte_lm_entry_table import ByteLmEntryKey
from training.byte_lm_host_glue_entry import publish_byte_lm_host_entries
from checks.native_tests.byte_lm_trainer_session_cases import BYTE_LM_TRAINER_SESSION_CASES, case, require


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    # The publication `_mojolearn_byte_lm_host` makes at module init.
    publish_byte_lm_host_entries()
    var failures = List[Int](length=BYTE_LM_TRAINER_SESSION_CASES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            case(ByteLmEntryKey(Int(GLOBAL_NUMERIC_MODE), 0), unit)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL byte_lm_trainer_session_cpu unit", unit, String(error))
    host_parallelize(task, BYTE_LM_TRAINER_SESSION_CASES)
    var failed = 0
    for value in failures:
        failed += value
    require(failed == 0, "byte_lm_trainer_session_cpu failed " + String(failed) + " units")
    print("PASS byte_lm_trainer_session_cpu:", BYTE_LM_TRAINER_SESSION_CASES, "resident session units, parallel")
