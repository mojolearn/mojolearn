# SPDX-License-Identifier: Apache-2.0
"""The CPU columns of the GP entry table (core/gp_entry_table.mojo; lane
gp-entry-table, 2026-11-12). Publication then lookup returns the published
pointer for both CPU images, which coexist in one process (separate column
slots); a repeated identical publication is accepted; an unpublished tier and
an invalid key refuse. Then the four GPC CPU fixtures run through the
RESOLVED entries (fit on the reference column, query on the inference column
for fixtures 2 and 3, as gp_native_cpu_suite pairs them), one fixture per
host core. Run first in a fresh process (the publication fills the process
slots)."""
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL
from core.host_parallel import host_parallelize
from core.entry_table import entry_table_same_call
from core.gp_native_api import GPDriver
from core.gp_native_provider import GPProviderAPI, gp_native_v1
from core.gp_native_host import GPHostKernels
from core.gp_native_host_infer import GPHostInferenceKernels
from core.gp_entry_table import (
    GPEntryKey, GP_COLUMN_DEVICE, GP_COLUMN_HOST, GP_COLUMN_HOST_INFER, gp_table_publish, gp_table_resolve,
)
from checks.native_tests.gp_native_classifier_cases import GP_CLASSIFIER_CPU_FIXTURES, classifier_fixture
from checks.native_tests.gp_state_suite import require


def _refuses(key: GPEntryKey) -> Bool:
    try:
        _ = gp_table_resolve(key)
    except:
        return True
    return False


def run_checks() raises:
    comptime assert GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    var mode = Int(GLOBAL_NUMERIC_MODE)
    var host_key = GPEntryKey(mode, 0, GP_COLUMN_HOST)
    var infer_key = GPEntryKey(mode, 0, GP_COLUMN_HOST_INFER)
    require(_refuses(host_key), "gp table: lookup before publication did not refuse")
    var host_api = GPProviderAPI[GPHostKernels]()
    var infer_api = GPProviderAPI[GPHostInferenceKernels]()
    gp_table_publish(host_key, Int(host_api.entries.unsafe_ptr()), host_api.entries[0])
    gp_table_publish(host_key, Int(host_api.entries.unsafe_ptr()), host_api.entries[0])
    gp_table_publish(infer_key, Int(infer_api.entries.unsafe_ptr()), infer_api.entries[0])
    var host = gp_table_resolve(host_key)
    var infer = gp_table_resolve(infer_key)
    require(entry_table_same_call(host.call, gp_native_v1[GPHostKernels]), "gp table: host lookup returned another pointer")
    require(entry_table_same_call(infer.call, gp_native_v1[GPHostInferenceKernels]),
            "gp table: inference lookup returned another pointer")
    require(_refuses(GPEntryKey(1 - mode, 0, GP_COLUMN_HOST)), "gp table: the other tier's unpublished slot did not refuse")
    require(_refuses(GPEntryKey(mode, 0, GP_COLUMN_DEVICE)), "gp table: the unpublished device column did not refuse")
    require(_refuses(GPEntryKey(mode, 2, GP_COLUMN_HOST)), "gp table: a host column with a GPU vendor did not refuse")
    var conflict = False
    try:
        gp_table_publish(host_key, Int(infer_api.entries.unsafe_ptr()), infer_api.entries[0])
    except:
        conflict = True
    require(conflict, "gp table: a second image in the same slot did not refuse")
    var failures = List[Int](length=GP_CLASSIFIER_CPU_FIXTURES, fill=0)
    var fp = failures.unsafe_ptr()
    def task(unit: Int) {imm fp}:
        try:
            # Each fixture resolves its own refs (a read-only acquire load).
            var mode_word = Int(GLOBAL_NUMERIC_MODE)
            var fit = gp_table_resolve(GPEntryKey(mode_word, 0, GP_COLUMN_HOST))
            var inference = gp_table_resolve(GPEntryKey(mode_word, 0, GP_COLUMN_HOST_INFER))
            var query = GPDriver(inference.call) if unit >= 2 else GPDriver(fit.call)
            classifier_fixture(GPDriver(fit.call), query, unit, True)
        except error:
            fp.unsafe_store(unit, 1)
            print("FAIL gp_entry_table_cpu fixture", unit, String(error))
    host_parallelize(task, GP_CLASSIFIER_CPU_FIXTURES)
    var failed = 0
    for value in failures:
        failed += value
    _ = host_api^
    _ = infer_api^
    require(failed == 0, "gp_entry_table_cpu failed " + String(failed) + " fixtures")
    print("PASS gp_entry_table_cpu: table admission + ", GP_CLASSIFIER_CPU_FIXTURES, "GPC fixtures through resolved entries")
