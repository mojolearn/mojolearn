# lane/rf-mutex-mechanism: what the fence repair did NOT settle

**Opened 2026-09-16, when the fence repair merged to `main` and the mechanism hunt was
deliberately decoupled from the release. THIS BRANCH IS NOT ON THE RELEASE'S CRITICAL
PATH. Nothing here blocks 0.8.7.**

Read `docs/lanes/RF_MUTEX_RECONCILIATION_2026-09-16.md` on `main` first. It is the full
account. This file is only the part that stayed open, kept on its own branch so that
merging a correct repair does not read as having settled why the defect happens.

## The one-sentence state

The stock cross-block mutex protocol is formally invalid, the acquire fence closes it,
the fence provably reaches the instruction stream on all three columns, and a forest A/B
of the equivalent acquire compare-exchange stopped the movement twice against a control
that moved. **Nobody has demonstrated the mechanism at the primitive, and the check built
to do that has returned a NULL twice.**

## Why the current probe is the wrong instrument, which is itself a finding

`core/device_mutex_check.mojo` on gfx942, twice, most recently at 64 to 128 blocks, 32 to
64 claims each, 4096 claims per configuration, a HOLD widening the critical section, four
repeats of both arms:

| arm | lost updates |
|---|---|
| stock, no fence | **0** |
| fence | 0 |
| skip-write sabotage | 4096, REJECTED every time |

The sabotage proves the count assertion can fail, so this is a null and not a broken
check. The forest reaches the window in 1% to 5% of fits; a hot counter hammered 4096
times does not reach it at all. **That gap is the clue.** Whatever opens the window looks
more like the forest's publish than like a single contended integer.

## Owed, in the order that would settle the most

1. **A FOREST-SHAPED PROBE.** Many nodes rather than one mutex, a real `Split` struct as
   the payload rather than two words, a merge that reads-modifies-writes the whole
   struct, and enough blocks that placement spans XCDs. Target the 1% to 5% rate the
   forest shows rather than expecting a per-claim failure. If the stock arm of THAT loses
   a candidate where the fence arm does not, the mechanism is demonstrated away from
   random forests and the repair is proved in the strong sense.
2. **A forest A/B at the shipping spelling**, with the SECTION guard
   (`tools/compare_binary_sections.py`) in place of the whole-file digest that could
   never report VOID. The fence has never been A/B'd on a device; the acquire
   compare-exchange has, twice.
3. **An rf identity cell with the repair in.** The fence adds an ordering constraint and
   no arithmetic, so no bit should move, and that is an argument rather than a
   measurement. Both Apple builds so far were taken with `MOJOLEARN_SKIP_BUILD_GATE` and
   compared nothing. Take it on the base fixture against the current reference, under the
   Metal lock, at one core.
4. **ExtraTrees and fused kNN, entirely unmeasured.** They carry the same protocol and
   are now repaired by the same helper, on the strength of the argument alone. Note the
   asymmetry: the kNN `-2 -> -1` consumer has a single consumer, which restricts the
   interleavings, so a matching spelling there is not independent proof of the same
   failure. The kNN producer has multiple producers and the argument applies directly.
5. **A run on real NVIDIA silicon.** `core/device_mutex_check.mojo` now cross-compiles
   for `sm_90a` and its PTX has been read, but it has never executed on an NVIDIA GPU.
6. **An emission audit of every other discarded atomic in the repository.** That a
   discarded acquire load emits nothing is a general fact about this compiler, not a fact
   about mutexes, and nothing has checked whether the same mistake is spelled elsewhere.

## Instruments that already exist, so the next session does not rebuild them

All of these merged to `main` with the repair.

- `tools/compare_binary_sections.py` compares two builds by SECTION, Mach-O or ELF, with
  `--expect same|differ`. Never compare segments; `__TEXT` starts at offset 0 and carries
  the `mktemp` install name, so a segment digest gives a false DIFFER.
- `-D MOJOLEARN_RF_MUTEX_CLAIM_STOCK=1` compiles the pre-repair claim for every caller.
  The A/B control. Never a default.
- `-D MOJOLEARN_MUTEX_SECTION_SABOTAGE=1` is the build-only positive control that must
  move the sections, so the comparator is seen able to differ.
- `-D MOJOLEARN_MUTEX_SPIN_RELAXED=1` is the differential-build instrument that answers
  "does this ordering survive into the instruction stream" on a column with no
  disassembler to hand.
- `tools/device_mutex_ab_leg.sh` is the guarded leg. It refuses to report any contrast
  unless the section gate first proves the arms are two programs on that target.
- `tools/check_mutex_handoff_model.py` is the bounded model check, with the negative
  control that exits 1.

## Rules this lane has already paid for

- **Run the byte compare BEFORE spending a box, never after.** Andrew's rule, set after
  the harness defect was found.
- **Do not rent anything for this work without asking.** Standing, and restated when this
  branch was opened.
- A zero from a probe that did not run is not a zero. The four instances this lane hit
  are listed at the end of the reconciliation doc on `main`.

Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>
