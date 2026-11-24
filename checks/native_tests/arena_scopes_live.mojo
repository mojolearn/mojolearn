# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The process count of live `resident()` scope states
(bindings/arena_cache_run.mojo `arena_scopes_live`; lane vdf-tail-b,
2026-11-24): `mc_slots_open` reads no context while it is zero.

One ordered sequence on the one process count (run in a fresh process: the
first check needs no scope state alive): none alive -> False; one new state
-> True; a second -> True; the first finalized -> still True; the second
finalized -> False; a finalize of an already released handle changes
nothing. A single ordered sequence over one process-wide word: nothing to
split across cores.
"""
from std.testing import TestSuite
from checks.native_tests.testsuite_cases import require
from bindings.arena_cache_run import arena_scope_new, arena_scope_finalize, arena_scopes_live


def test_scopes_live_count() raises:
    require(not arena_scopes_live(), "no scope state alive at start")
    var a = arena_scope_new(0)
    require(arena_scopes_live(), "one scope state alive")
    var b = arena_scope_new(16)
    require(arena_scopes_live(), "two scope states alive")
    _ = arena_scope_finalize(a)
    require(arena_scopes_live(), "the second scope state is still alive")
    _ = arena_scope_finalize(b)
    require(not arena_scopes_live(), "every scope state released")
    _ = arena_scope_finalize(b)  # a released handle: the acquire refuses, the count is not touched
    require(not arena_scopes_live(), "a second finalize does not move the count")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
