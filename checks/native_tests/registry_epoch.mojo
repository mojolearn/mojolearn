# SPDX-License-Identifier: Apache-2.0
# native-check: cpu
"""The per-registry weight epoch (core/weight_epoch.mojo `RegistryKey`; lane
fix-samba-weight-epoch-per-model, 2026-11-13): a write stamped on one
registry leaves every other registry's resident key current, a process-epoch
bump leaves every registry key current, and a full table never leaves a key
current that should reload, and a key's read is the one word it resolved
at load. Host only, no device, no Python.

Serial by nature: every fixture reads and writes the ONE process-wide epoch
table (a SharedSlot), and the overflow fixture clears it, so the fixtures are
not independent units and are not split across the host pool. Each fixture is
a few hundred table operations on the calling thread.
"""
from std.testing import TestSuite
from core.weight_epoch import (
    RegistryKey,
    registry_epoch,
    registry_epoch_bump,
    weight_epoch_bump,
    SAMBA_WEIGHT_EPOCH_GLOBAL,
)

#: Fake registry addresses (never dereferenced: the key compares words).
comptime _A = 0x10000
comptime _B = 0x20000


def require(ok: Bool, message: String) raises:
    if not ok:
        raise Error("registry_epoch: " + message)


def _two_models_alternate() raises:
    # Two stacks' resident copies: loaded once each, then read alternately.
    var ka = RegistryKey.take(_A)
    var kb = RegistryKey.take(_B)
    for _ in range(4):  # driver-loop: alternating calls of two models
        require(ka.current(_A), "A's copy went stale with no write")
        require(kb.current(_B), "B's copy went stale with no write")
    # A write to A: A reloads once, B stays current.
    registry_epoch_bump(_A)
    require(not ka.current(_A), "a write to A left A's copy current")
    require(kb.current(_B), "a write to A invalidated B's copy")
    ka = RegistryKey.take(_A)
    require(ka.current(_A), "A's reloaded copy is not current")
    require(kb.current(_B), "B went stale after A's reload")
    # A write to B: the mirror.
    registry_epoch_bump(_B)
    require(ka.current(_A), "a write to B invalidated A's copy")
    require(not kb.current(_B), "a write to B left B's copy current")


def _process_epoch_leaves_registries() raises:
    # Another family's weight write (the process epoch) re-uploads no Samba
    # registry: the bug this lane fixes.
    var ka = RegistryKey.take(_A)
    var kb = RegistryKey.take(_B)
    weight_epoch_bump()
    require(ka.current(_A), "a process-epoch bump invalidated A")
    require(kb.current(_B), "a process-epoch bump invalidated B")


def _keys_are_per_address() raises:
    var ka = RegistryKey.take(_A)
    require(not ka.current(_B), "A's key is current for B's address")
    require(not RegistryKey().current(_A), "an empty key is current")
    var before = registry_epoch(_A)
    registry_epoch_bump(0)
    require(registry_epoch(_A) == before, "a zero address stamped a registry")
    require(ka.current(_A), "a zero-address bump invalidated A")


def _full_table_reloads_every_key() raises:
    var ka = RegistryKey.take(_A)
    registry_epoch_bump(_A)
    ka = RegistryKey.take(_A)
    var kb = RegistryKey.take(_B)
    # Writes to registries no copy was keyed by take no slot and leave A and
    # B current (hot-path audit 2026-11-13: slots are given at load only).
    for i in range(1100):  # driver-loop: over distinct fake registries never keyed
        registry_epoch_bump(0x100000 + 64 * i)
    require(ka.current(_A) and kb.current(_B), "a write to an unkeyed registry invalidated A or B")
    # Key more distinct registries than the table holds (twice over): every
    # slot is reused with a fresh stamp, so A and B reload once.
    for i in range(2100):  # driver-loop: over distinct fake registries filling the table
        _ = RegistryKey.take(0x100000 + 64 * i)
    require(not ka.current(_A), "A's copy survived the reuse of its slot")
    require(not kb.current(_B), "B's copy survived the reuse of its slot")
    ka = RegistryKey.take(_A)
    kb = RegistryKey.take(_B)
    require(ka.current(_A) and kb.current(_B), "keys taken after the reuse are not current")
    registry_epoch_bump(_A)
    require(not ka.current(_A) and kb.current(_B), "per-registry stamping broke after the reuse")


def _read_is_one_word() raises:
    # The read is one atomic load of the word resolved at `take`: the key
    # holds that word, and it is the registry's slot word (no lookup later).
    var ka = RegistryKey.take(_A)
    require(ka.word != 0, "a taken key holds no epoch word")
    require(RegistryKey.take(_A).word == ka.word, "one registry resolved to two words")
    require(RegistryKey.take(_B).word != ka.word, "two registries share one word")
    require(registry_epoch(_A) == ka.epoch, "the held word and the table disagree")


def _writer_keeps_its_key() raises:
    # lane fx-samba-step-resident: `stamp_and_take`, the stamp of a write made
    # FROM a resident copy (a Samba train step's download). The writer's key
    # is current; every key taken before it reloads; another registry is
    # untouched; any later write leaves the writer's key not current.
    var other = RegistryKey.take(_A)
    var kb = RegistryKey.take(_B)
    var mine = RegistryKey.stamp_and_take(_A)
    require(mine.current(_A), "the writer's own key is not current after its stamp")
    require(not other.current(_A), "a copy keyed before the writer's stamp stayed current")
    require(kb.current(_B), "the writer's stamp on A invalidated B")
    require(mine.word == RegistryKey.take(_A).word, "the writer's key holds another word")
    require(registry_epoch(_A) == mine.epoch, "the writer's key and the table disagree")
    registry_epoch_bump(_A)
    require(not mine.current(_A), "a later write left the writer's key current")
    # a second writer stamp supersedes the first
    var first = RegistryKey.stamp_and_take(_A)
    var second = RegistryKey.stamp_and_take(_A)
    require(not first.current(_A) and second.current(_A), "two writer stamps: the older key is current")
    # a registry no copy was keyed by gets a slot and a current key; no address, no key
    var fresh = RegistryKey.stamp_and_take(0x30000)
    require(fresh.current(0x30000), "a first-ever writer stamp gave a key that is not current")
    require(not RegistryKey.stamp_and_take(0).current(0), "a zero address gave a current key")


def test_fixtures_in_order() raises:
    """One test, the six fixtures in their order: they share the one process-wide epoch table (docstring)."""
    comptime if SAMBA_WEIGHT_EPOCH_GLOBAL:
        # The B arm keys every registry by the process epoch: the per-model
        # expectations below do not hold by design.
        print("registry_epoch: built with MOJOLEARN_SAMBA_WEIGHT_EPOCH_GLOBAL; per-registry checks skipped")
        return
    else:
        _two_models_alternate()
        _process_epoch_leaves_registries()
        _keys_are_per_address()
        _full_table_reloads_every_key()
        _read_is_one_word()
        _writer_keeps_its_key()


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
