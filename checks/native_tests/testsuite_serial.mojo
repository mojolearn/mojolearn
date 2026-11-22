# SPDX-License-Identifier: Apache-2.0
"""The serial case table and the column constant the TestSuite checks of batch 1 need beside
`testsuite_cases` (docs/plans/TESTSUITE_CONVERSION.md, step 7).

- `check_cases_serial(label, n, body)`: `body(0) ... body(n - 1)` one after another on the calling thread (the
  device column: the cases share the one device context and its pool, so they may not run concurrently); every
  failing row is named in the one raised error, as `check_cases` does.
- `CPU_COLUMN_BUILD`: True in the host column build (`-D MOJOLEARN_COLUMN_CPU=1`), for the `comptime if` that picks
  a column's executor.

These belong in `testsuite_cases.mojo` (the recipe says so); batch 1 keeps them here because nine lanes convert at
once and that file is shared. The merge moves them there and this module goes.

No `main()` and no `test_*`: no file's discovery picks anything up from here.
"""
from std.sys.compile import is_defined

from checks.native_tests.testsuite_cases import _raise_failures

comptime CPU_COLUMN_BUILD = is_defined["MOJOLEARN_COLUMN_CPU"]()


def check_cases_serial[F: def(Int) raises -> None](label: String, n: Int, ref body: F) raises:
    """Run `body(0) ... body(n - 1)` in order; raise one error naming every failing row."""
    var failed = List[Int](length=n, fill=0)
    var messages = List[String](length=n, fill=String(""))
    for unit in range(n):  # driver-loop: over the cases, in order on the one device
        try:
            body(unit)
        except problem:
            messages[unit] = String(problem)
            failed[unit] = 1
    _raise_failures(label, n, failed, messages, List[String]())
