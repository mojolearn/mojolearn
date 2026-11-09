# Rules consolidation, 2026-11-09 (proposal on `lane/rules-consolidation`, not on main)

Andrew, 2026-11-09: eliminate duplicates outright, resolve overlaps, make `AGENTS.md` the one rule file for every
agent, keep in `CLAUDE.md` only what is Claude-specific, and cut long rationale.

Line numbers below are in the files at commit 2c1b720133 (`AGENTS.md` 666 lines, `CLAUDE.md` 197 lines). The complete
old text of both is kept verbatim in [RULES_HISTORY.md](RULES_HISTORY.md). "A:" is old `AGENTS.md`, "C:" is old
`CLAUDE.md`. "New" names the section of the rewritten `AGENTS.md` unless it says `CLAUDE.md`.

Result: `AGENTS.md` 666 -> 432 lines (56 of them the five hard rules copied in verbatim), `CLAUDE.md` 197 -> 42 lines
(863 -> 474 loaded lines). Six conflicts that neither date nor scope settled were put to Andrew and are decided in
section (d).

## (a) Every topic: where it was, the current rule, where it is now

| topic | old location | current rule | new location |
|---|---|---|---|
| The name | A:3-10, C:5-12 | Prose says Mojolearn; code names stay lower case in backticks. | The name is Mojolearn |
| Bits between versions | A:12-25, A:87-93, A:261-266, A:212-213, C:88 | Bits may change between versions; only cross-vendor identity within one version is required. | Identity and bits |
| Three-vendor contract | A:231-233, A:403-408, C:88, C:119-120, C:187-188 | Same bits on Apple, NVIDIA and AMD in `identical`. | Identity and bits |
| Host column | C:88 ("recorded, not required"), C:156 ("match across the GPUs and the CPU") | The CPU column is required, recorded first; GPUs are checked against it (decided 2026-11-09). | Identity and bits |
| No vendor-specific references, keep history | A:99-107 | Never hide a disagreement behind a per-vendor reference; keep historical tables. | Identity and bits |
| `fast` needs no identity | C:89 | `fast` is judged on speed and quality only. | Identity and bits |
| Focus on `identical`, `fast` investment stopped | A:217-235 | No dedicated `fast` work unless Andrew reopens it. | Identity and bits |
| Fix everything before a release | A:27-39 | All work in, measure, fix, measure again, then freeze. | Release order |
| Verify once, pre-release | A:64-74, C:152-164, C:142-150, A:273-275 | One global verify on the frozen source; no ID checks in development; no re-verification of green work. | Release order |
| What verification consists of | A:261-289 | Bounded fixtures and installed-wheel smokes; never promote a compile to runtime success. | Release order |
| Reuse of old runs | A:291-316, C:166-176, C:145, C:163-164 | Reuse is opt-in everywhere, behind an explicit flag (decided 2026-11-09). | Reuse of old runs is opt-in |
| Our CPU is never timed | C:35, C:92, C:94 | Never race or time our CPU or a host route. | Measurement and fairness |
| Opponents scored once | C:94, A:413, A:477 | No opponent re-score. | Measurement and fairness |
| One run per arm | C:79, A:488-489 | One scored run per arm; the harness may take one excluded warmup. | Measurement and fairness |
| Never rent Apple | C:93 | The verbatim hard rule. | Hard rules (Andrew) |
| Measurement process items 1-6 | A:401-413, C:113-125 | Freeze per round; NV+AMD average decides; update boards; fix problems; opponent clocks. | Measurement process |
| Full-dataset candidate evaluation | A:380-398 | End-to-end A/B on the full dataset; component runs are screening. | Measurement process |
| Incremental experiment delivery | A:479-506 | Decide each candidate when its evidence is complete; reason beside the switch. | Measurement process, Experiments |
| Race fairness | A:415-477 | Same algorithm, same explicit parameters; quality shown, never a gate; the rule is code. | Race fairness |
| CPU race resources | A:508-521 | Full machine for CPU opponent arms; record the real allocation. | Full-machine CPU race resources |
| No dimension targeting | A:369-376, C:104-111 | No rule keyed on a board shape or dataset name. | No dimension targeting |
| GPU path is GPU only and parallel | C:87, C:91 | No host steps, no serial default; driver loops only, tagged. | The GPU path |
| No Python in the runtime | A:556-653, C:90, C:127-140 | One binding call per public method; everything else in Mojo; no callbacks. | No Python in the runtime |
| Direct CPython allowed | A:142-160 | C-API calls are allowed during execution; audit hooks. | No Python in the runtime |
| Exported graph boundary | A:206-215 | Never run a user's Python `forward` or exporter in product execution. | No Python in the runtime |
| sklearn tags | A:576-594, C:134-140 | Tag metadata may be Python; no callbacks into it. | No Python in the runtime |
| Python fallback for a Mojo bug | A:558-574 vs A:618-627, A:651-653 | `python/mojolearn/_runtime_fallbacks.py` is the complete frozen list (decided 2026-11-09). | No Python in the runtime |
| Wait for Mojo and Modular | A:349-367, C:96-102 | No workarounds for unsupported toolchain features; record the ask. | Wait for Mojo and Modular |
| Adapters | A:534-554 | Front-ends only: no identity lanes; qualified by installed-wheel smoke. | Framework adapters are the same API in another format |
| Lanes: code then queue | C:26-59, A:484-486 | Lanes write and push; never test, time, ssh or rent. | Lanes |
| Lanes and compiling | C:42-46, A:484-486 vs A:162-204 | Only the orchestrator compiles, plus one named integration compile-fix lane (decided 2026-11-09). | Lanes |
| Worktrees, push, /private/tmp, bash | C:47-55 | One worktree per lane; never `--no-verify`; nothing in `/private/tmp`; bash. | Lanes |
| Final reply | C:56-64, A:346-347 | Short: branch, changes with `file:line`, deletions, bits, define, queued, owed. | Lanes |
| Peer communication | A:76-85 | Only to execute authorized work. | Lanes |
| Brief reminder | A:336-347, C:17 | Every brief carries the log reminder. | Lanes |
| Compile slots | C:47, A:186-187 | Mac compiles go through `tools/compile_slot.sh`. | Compiles |
| Incremental development compilation | A:116-140 | Smallest affected groups; reuse unchanged groups. | Compiles |
| `lq` and the boxes | C:29-40 | `lq` is the lane's only way to a box; box list. | Boxes |
| Hard rules: Apple machines, full CPU, parallel code, parallel tests, remove what is unnecessary | private `~/.claude/CLAUDE.md` (read only), C:93, A:174-178 | Copied verbatim with dates. | Hard rules (Andrew) |
| Measurement build | not in either file | `tools/release.py candidate` builds wheels to measure; it is not a release candidate. | Fix everything before a release |
| Box jobs persist | C:38 | Jobs start through the three wrappers and persist to R2 from the box. | Boxes |
| Results leave the box | A:318-334 | Logs and results on the Mac and in R2; archive before teardown. | Boxes |
| Artifact retention before teardown | A:523-532 | Prove by bytes where every output is stored before terminating. | Boxes |
| No background watchers | A:253-259 | Watchers stay disabled. | Boxes |
| Experiments | C:61-66 | Define per experiment; winner becomes default; loser never reaches main; one EXPERIMENTS row. | Experiments |
| Deleting an experiment | C:68-79 | Permanent tombstone record, four items. Kept VERBATIM. | Deleting an experiment |
| Mojo rules that cost a build round | A:41-62 | Kept VERBATIM. | Mojo rules that cost a build round |
| Logs and context | A:237-251, C:14-24, A:111-114, A:412 | Logs to files; bounded grep and tails; filtered output is not proof. | Logs and context |
| Public docs | A:655-666, C:178-197 | User voice; omit, never caveat; `fast` Apple only. | Public docs and docstrings |
| Briefs directory, relaunch | C:81-83 | Briefs saved before launch; relaunch from brief. | `CLAUDE.md` Briefs |
| Never read subagent transcripts | C:23, A:250-251 | Kept for every agent, and restated for Claude. | Logs and context; `CLAUDE.md` |
| Model roles | not in either file (memory only) | Fable judges and orchestrates; Opus lanes implement; Fable lanes for judgment. | `CLAUDE.md` Who does what |

## (b) Duplicates removed (27)

The copy kept is the one in the new `AGENTS.md`; the other copies are deleted.

1. The name: A:3-10 and C:5-12 (identical text). One kept.
2. Wait for Mojo and Modular: A:349-355 and C:96-102 (identical first paragraph). One kept.
3. No dimension targeting: A:369-376 and C:104-111. A's kept (C's still said "NVIDIA and AMD", see (c) 2).
4. Measurement process items 1-6: A:401-413 and C:113-125 (near identical). One kept, A's wording for item 2.
5. Opponent clocks (item 6): A:413 and C:125 (identical). One kept, shortened.
6. Reuse good old runs: A:291-316 and C:166-176. One kept (as question 2 text A).
7. Public docs: A:655-666 and C:178-197. Merged into one list; C's extra bullets (api_exclude name and reason, the
   162M model) kept.
8. Logs: A:237-251, C:14-24, A:111-114, A:412 (item 5), C:124. One section.
9. Lane brief log reminder: A:336-344 and C:16-17. One bullet under Lanes.
10. Never read subagent transcripts: A:250-251 and C:23. One bullet.
11. Bits may change between versions: A:12-25 (two statements, 11-05 and 11-08), A:89-93, A:261-266, A:212-213,
    A:374-375, C:88. One rule.
12. Three-vendor identity: A:89, A:231, A:406-407, C:88, C:119-120, C:187. One rule.
13. Verify once pre-release: A:64-74 and C:152-164. One section.
14. No re-verification of green work: A:273-275 and C:147-150. One bullet.
15. Lanes never compile: A:484-486 and C:42-46. One text (question 1 text A).
16. Compile slot: A:186-187 and C:47. One bullet.
17. Reuse unchanged compiled groups: A:108-110, A:123-126, A:267-272. One bullet.
18. Compilation is not verification: A:169-170, A:487, A:271-272. One bullet.
19. No Python: the summary in C:90 and C:127-132 against the full text A:596-653. One section.
20. sklearn tags: A:576-594 and C:134-140. One bullet.
21. Opponents scored once / audits never rerun: C:94, A:413 ("NO opponent re-score"), A:477, A:449-450. One bullet each
    in two sections (measurement, fairness).
22. `fast` is Apple only in docs: A:666, C:192, C:187. One bullet.
23. Never rent Apple: C:93 (also in Andrew's private global file, untouched). One bullet.
24. Fix problems when found (item 4) kept once.
25. Record winners and losers beside the switch: A:394 and A:501-504. One bullet.
26. A promoted change starts a new freeze: A:401-402 and A:505. In item 1 and the candidate list.
27. Final reply or handoff format: A:346-347 and C:56-64. One bullet.

## (c) Superseded passages (14), moved to RULES_HISTORY.md

The later dated owner statement wins unless it is narrower.

1. **ID checks per lane.** C:32-33 "Same-bits checks use `lq add <box> ID <branch> ...` ... Queue it on nv and amd"
   is superseded by A:69-74 / C:158-160 (Andrew, 2026-11-05): "No same-bits checks during development ... This
   supersedes older instructions that ask a lane to queue ID checks." The `lq ... ID` instruction is gone from the
   rules.
2. **Two-vendor identity.** C:33 "identity is the nv device digest equal to the amd device digest" and C:108-110 "bits
   only have to match across NVIDIA and AMD" are superseded by the three-vendor rule (Andrew, 2026-10-13; C:88, A:375).
3. **Light gate vs fix everything.** A:280-286 "Full-dataset performance measurements ... are separate campaigns, not
   automatic publication prerequisites" (2026-10-25) is superseded for our own arm by A:35-37 (Andrew, 2026-11-08):
   "Measure before the release ... Only a candidate with no open regression is frozen." Opponent races remain not a
   prerequisite (opponents are scored once).
4. **Compile-only, no final verification.** A:167-169 "Do not run runtime, identity or cross-vendor verification
   campaigns for this work" (2026-10-27, the provider migration) is narrower than and older than A:64-68
   (2026-11-05). Kept as: development merges need a compile, not a verification campaign.
5. **Fresh reference campaign.** A:94-98 "freeze ... and regenerate reference hashes for EVERY algorithm" was a
   one-time instruction for the sampler change (2026-10-29); every release now re-records (2026-11-05).
6. **"Neither vendor materially slower".** A:499-500 (2026-10-05) is superseded by A:403-405 (Andrew, 2026-10-08):
   the average of the two vendors decides; a split is flagged and does not block.
7. **"Compare quality first".** Already marked superseded inside A:443; only the current rule is kept.
8. **Slow-host exclusion.** Removed 2026-10-14 (A:430-431, A:472-473); one clause kept ("No host is excluded for being
   slow").
9. **Earlier Apple `fast` directions.** A:228-230 "supersedes earlier directions to pursue Apple `fast`"; the lane
   section heading "Experiments (Apple FAST and every speed lane)" (C:61) becomes "Experiments".
10. **Boundary-only CPython.** A:145-146 "supersedes earlier boundary-only or blanket CPython/import prohibitions";
    only the current rule is kept.
11. **Python API glue exceptions.** A:596 "supersedes earlier Python API/glue exceptions"; only the current rule is
    kept.
12. **Four compile slots.** A:186 and C:47 say "4 machine-wide slots"; `tools/compile_slot.sh` defaults to 2 since
    Andrew's 2026-10-28 instruction. The rule now points at the script and gives no number.
13. **"Our CPU arms wherever measured".** A:510 (2026-10-05) predates no rule but has no instance: our CPU is never
    raced or timed (C:35, C:92). The section now speaks of CPU opponent arms.
14. **Campaign-only instructions** with no standing content: the provider migration delivery notes (A:194-204), the
    tokenizer exception (A:203-204), "this campaign covers all NVIDIA/AMD GPU-opponent recipes" (A:450), the
    Daybreak interruption notes (A:569-574, kept inside question 3 text B's history).

## (d) Questions put to Andrew, and his decisions (2026-11-09)

All six are decided and applied. No `OPEN` marker remains in `AGENTS.md`; each losing text is listed, dated, at the end
of RULES_HISTORY.md.

1. **May anything other than the orchestrator compile or rent a build box?** DECIDED: only the orchestrator
   compiles, plus ONE designated integration compile-fix lane per integration branch that the orchestrator names in
   its brief. The 2026-10-27 "any peer may rent a CPU box and merge to main for this migration" text is retired.
2. **Is reuse of old identity runs default or opt-in?** DECIDED: opt-in everywhere. Every release records every
   lane on every device once; a reused older record is accepted only with the explicit reuse flag.
   `tools/record_identity_column.sh` still defaults to reuse and must be changed by a follow-up lane (not changed
   here: the default is exercised by `tools/test_identity_auto_reuse.py` and six other test files, so it cannot be
   flipped with certainty by reading). What to change: `tools/record_identity_column.sh:256-261` (the `FULL` case
   and the `elif` that enters reuse when no `LANES`, `SHARD`, `ONLY_CHANGED` or `FULL` is given) so that the default
   records every lane and reuse needs a named flag (for example `REUSE=1`); its header comment at `:50-65`; and the
   tests `tools/test_identity_auto_reuse.py`, `tools/test_identity_once.py`,
   `tools/test_identity_columns_changed_lanes.py`, `tools/test_release_identity_run.py`,
   `tools/test_record_identity_installed.py`, `tools/test_identity_wheel_record.py`, `tools/test_box_merged_verify.py`.
3. **Python fallback for a Mojo bug.** DECIDED: a frozen list. `python/mojolearn/_runtime_fallbacks.py` is the
   complete list (one entry today: `len(Array)` on certain native owners). Nothing is added without Andrew's
   explicit word; an agent that finds a Mojo bug stops, writes the reproducer and the ask, and reports. "No
   additional permission is required" is dropped. These paths are never called Python-free.
4. **Must the CPU column match?** DECIDED: yes. Required, recorded first on GitHub runners, GPUs checked against it.
5. **Which AMD box?** DECIDED: the DigitalOcean AMD GPU droplet; MI300X is the default and any other model is
   flagged as different hardware in the results.
6. **Hard rules from Andrew's private file.** DECIDED: copied verbatim with their dates into one section, "Hard
   rules (Andrew)" (five rules). The weaker copies ("never add, rent, extend or release an Apple machine", "use
   every allocated core") were removed in favor of the hard rules.

Also added on Andrew's word: one line in "Fix everything before a release" defining a measurement build
(`tools/release.py candidate`) as not a release candidate.

## (e) Rationale cut, and where it went

All of it is verbatim in RULES_HISTORY.md.

- Race fairness: the good/broken lists are shortened to one bullet each; the audit record layout
  (`race["audit"]`, `admitted_opponents`, pairing fingerprints) and the evidence fields are tool documentation and
  stay in `tools/race_fairness.py` and `tools/bench_board_evidence.py`.
- No Python in the runtime: five restatements (10-06, 10-19, 10-20 and two audit paragraphs) became two bullets.
- Full-dataset candidate evaluation and incremental experiment delivery: 45 lines became 13.
- Lightweight release verification: the explanation of development builds as evidence.
- Results leave the box: the list of box paths, `tools/box_archive.sh` detail and `MOJOLEARN_DOWN_NO_ARCHIVE`.
- Incremental development compilation: the worked `--group` example and the exit code note.
- Adapters: the smoke commands and the list of removed and remaining entry names.
- Opponent clocks: the per-library enumeration (torch twins, cuML/cuVS/cuGraph).
- Measurement artifact retention: the EBS example.
- The quoted Andrew statements that only restate the rule beside them.

## Tools that name these files

No tool, test, hook or workflow reads the content of `AGENTS.md` or `CLAUDE.md`. `tools/release_requirement_dependencies.py:291`
lists both as tooling files by name (unchanged). About 45 comments cite a section by title. Every cited title still
exists as a heading in the new `AGENTS.md`:

- "Framework adapters are the same API in another format" (release.py, identity_break.py, adapter_gate.py,
  verify_coverage_audit.py, coverage.py, coverage_sabotage.py, public_runtime_inventory.py, adapters/__init__.py,
  docs/TOMBSTONES.md)
- "Race fairness: what the measurements are for" (race_fairness.py, test_race_fairness.py, main_board_ingest.py)
- "Measurement process" and its item 6, OPPONENT CLOCKS (board_clock_audit.py, bench_board.py, main_board_ingest.py,
  bench_board_metrics.py, six_lane_grid_decide.py): still item 6
- "Reuse good old runs, automatically" (release.py): the heading is now "Reuse of old runs is opt-in (was: Reuse good
  old runs, automatically)", so the cited phrase is still found
- "Full-machine CPU race resources" (paper_compare_job.sh)
- "No dimension targeting" (six_lane_grid.py)
- "No Python in the runtime" (mojo_python_callback_audit.py, test_no_python_runtime.py); the anchor
  `AGENTS.md#no-python-in-the-runtime` still resolves
- "Experiments" and "Deleting an experiment", step 4 (six_lane_grid_decide.py, docs/build/HOT_GROUPS.md)

Fourteen comments say "CLAUDE.md" for a rule that now lives in `AGENTS.md` (the `fast` quality rule, never time a CPU
route, the box names, the compile rule, opponent clocks): `tools/fast_quality_rule.py`, `af_quality.py`,
`af_board_apply.py`, `six_lane_grid_fast.py`, `six_lane_grid_decide.py`, `six_lane_grid_decide.sh`,
`six_lane_grid_lq.py`, `measurement_coverage.py`, `record_changed_lanes.py`, `release_lq_smoke.py`, `compile_slot.sh`,
`bench_board_metrics.py`, `main_board_ingest.py`, `hooks/no_host_routes.py`, `apple_fast_ops/README.md`,
`ops/README.md`. They are comments, and `CLAUDE.md` imports `AGENTS.md`, so they are not wrong for a Claude reader;
they were left as they are and can be re-pointed in one mechanical commit after this lands.
