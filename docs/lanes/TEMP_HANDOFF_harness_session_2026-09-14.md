# TEMPORARY. Handoff for the session that replaces mojolearn-97 (the harness owner). Delete when its owed list is empty.

Written 2026-09-14 night at Andrew's request; his Fable session is out of time. This is the
whole state the next session needs, the file ownership that keeps two sessions from colliding,
and the exact next actions in order. The peer session is `mojolearn-ea` (ListAgents shows it);
message it first thing and say you are the new harness owner.

## 1. Read these first, in this order

1. `docs/lanes/TEMP_claim_surface_plan_2026-09-14.md`: the eight workstreams A to H, who owns
   which, the split rule, and the prompt the peer works from.
2. `docs/lanes/BRIEF_claim_surface_census_2026-09-14.md`: what the identity claim covers versus
   the parameter and code surface (the reason the harness grew from 46 to 151 lanes today).
3. `docs/lanes/BRIEF_resident_and_dtlimit_moved_2026-09-14.md`: how the two Mamba deviations
   were found and closed; sections 3.2 and 3.3 are the method (which column moved, the per-fit
   dump, the poison-and-band gate). Read it for the METHOD; the deviations are closed.
4. `bench/results/identity_break/2026-09-14_120-lanes-2712fix/README.md`: the current four-box
   record; `2026-09-14_136-lanes/` is the one landing now and becomes the CPU gate's columns.
5. Memory (auto-loaded): `claim-surface-census-sep14`, `deviation-2712-is-apple-...`,
   `shared-so-overwritten-in-place-sigkill`, `no-oversized-blobs-...`. They carry the day's rules.

## 2. State at handoff (origin/main 862ab7691)

| surface | state |
|---|---|
| harness `tools/identity_break.py` | 136 lanes on main; 151 on `origin/lane/claim-surface-lanes` (e33f457bc, 7 commits ahead: the 15 workstream-D lanes plus the six `_NOT_YET` rows they close). NOT pushed to main because that branch carries `lane/expose-d`, which is not mergeable yet (see the exposures row) |
| three-vendor record | `2026-09-14_120-lanes-2712fix`: Apple M4, H100, MI300X, MI325X each 1080/1080, gate columns IDENTICAL=1080 + 1476. `2026-09-14_136-lanes` at 4048e1b51: Apple and H100 complete (1224 each, identical), MI300X finishing its last seven lanes on the box, plus the 2xH100 par column (144/144 IDENTICAL x3 with Apple and the one-device H100) |
| CPU gate columns | still the 47-lane record on purpose; the 136-lane record's commit switches `python/mojolearn/host_surface.py` TRAINING_GPU_COLUMNS and the workflow greps to it (the peer does this in the record's commit) |
| CPU training | 25 lanes identical x4 on seven free runners (E batches 1 and 2 on main); batch 3 (forests, then GBDT host trainers) on `lane/cpu-training-e3`; NO_CPU_PATH = UMAP, GP, ARIMA, the neural blocks, forest/GBDT training |
| exposures (workstream D) | `lane/expose-d` 2b2f568b0 built Cholesky, KernelRidge/Nystroem/RBFSampler, GaussianMixture, HDBSCAN, resample, the six training primitives, the KMeans arms; IVF and embedding prepared, not exposed. NOT mergeable: HDBSCAN crashes the Mojo compiler on gfx942 (exit 139, AMDGPU isel on the stability kernel), mixture RED (random init), kernel_methods RED (square Nystroem off by 2.85; RBFSampler n_components=0 refuses from the buffer layer). `lane/expose-d-fixes` is the peer's worker on it |
| multi-GPU drivers (Codex, on main) | 13 families; 16 `par-*` harness lanes with `MOJOLEARN_PAR_DEVICES` (default "0", recorded as `package.par_devices`); two H100s equal to one H100 and an Apple M4 on 288 cells; AMD two-device column owed (Hot Aisle 2gpu spec); throughput never measured, memory not pooled |
| outsider verification | `tools/verify_external.sh` + `docs/VERIFY_EXTERNALLY.md`; two H100 runs from the PyPI 0.8.5 wheel under `bench/results/verify_external/`; the verdict is positive (column clean AND diff identical); the record and the wheel must come from the same commit, no override switch by design; 0.8.6 ships a record at its own commit |
| packaging (workstream F) | `lane/packaging-f`: all ten host bindings and `python -m mojolearn identity` in the wheels, 0.8.6 prep without the bump, unmeasured until the freeze |
| deviations | 2711 (sm_120a split-K miscompile) fixed; 2712 + 2713 (one over-read in `m2_ydiag_kernel`) fixed and closed on four boxes with the poison-and-band gate `pixi run check-mamba-poison`; 2714 = the k-means round seed sign-extension, PINNED by name, never a fix |

## 3. File ownership (agreed with the peer; keep it or renegotiate by message)

Yours (the harness owner): `tools/identity_break.py`, `tools/mamba2_step_probe.py`,
`tools/verify_external.sh`, `docs/VERIFY_EXTERNALLY.md`, `python/mojolearn/ensemble.py` and
`model_selection.py` (the two refusals), `python/mojolearn/__init__.py`'s `_NOT_YET` block,
`python/mojolearn/tests/test_refuse_ignored_knobs.py`, the two TEMP files and the two briefs above.
The peer's workers send you lane bodies as `docs/lanes/LANE_BODY_<family>.py` and never edit the
harness; you merge them (adapt only what the harness's semantics require, say so in the commit).

The peer's: everything that needs a rented box or a Mojo build (legs, records, gates, bindings,
host oracles, packaging, the `host_surface.py` manifest and the workflow).

## 4. Next actions, in order

1. Message `mojolearn-ea`: you are the new harness owner; ask for the D merge signal and the
   AMD two-device column, and confirm nothing of yours is running on the Mac.
2. When the peer says `lane/expose-d` (with its fixes) is on main: in the worktree
   `.claude/worktrees/claim-surface` (branch `lane/claim-surface-lanes`), `git fetch`,
   `git merge origin/main`, check `python3 -c "import ast; ast.parse(open('tools/identity_break.py').read())"`,
   confirm 151 lanes load, confirm `_NOT_YET` lists only IVFIndex and Embedding, run
   `cd python && python3 -m mojolearn.tests.test_refuse_ignored_knobs`, then
   `git push origin lane/claim-surface-lanes HEAD:main` (fast-forward only; if refused, merge
   again). Tell the peer; the next three-column run carries 151 lanes.
3. When the AMD two-device column lands: diff it against the one-device MI300X column and the
   Apple column with `python3 tools/identity_break.py --diff <apple> <mi300x one-device> <amd two-device> --require-columns 3 --lanes par-forest,...,par-iforest`
   (the sixteen `par-*` names are in the harness docstring). IDENTICAL on all 288 cells closes the
   drivers' cross-vendor two-device claim; anything else is a lane brief, not a docs edit.
4. Watch for new `LANE_BODY_*.py` files from the peer's workers (E batch 3 will not need any;
   D-fixes may revise the mixture and kernel-methods bodies). Merge the same way as step 2.
5. Owed, small, yours: the 13 D lanes have NO real cell yet on any box (only training-primitives
   was smoked; kmeans-cosine's Mac hash was the stale binding's param-count error, not the cosine
   refusal). Their first cells come from the next three-column run after step 2; check that
   `kmeans-cosine` hashes the SAME sentence on all three columns and that `gmm/dupes` is a legal
   REFUSED (a collapsed component, DEVIATION 1723) or STABLE, not MOVED.
6. Delete this file and `TEMP_claim_surface_plan_2026-09-14.md` when their owed lists are empty.

## 5. Rules that bit today (all in memory, repeated because they cost hours)

- On any DIVERGENT cell, print EVERY column's hash beside the earlier records BEFORE naming a
  vendor. The diff names a cell, not a culprit; we spent four hours and four AMD legs on the
  wrong vendor because nobody asked which column moved (it was Apple).
- A gate must be seen to FAIL: run its sabotage first; a "no DIVERGENT" pass over a column of
  REFUSED cells is not a pass (two false passes today, one mine, one the peer's).
- `_h` refuses object and text arrays; a dict hashed through `np.asarray` hashes its address and
  reads MOVED forever (byte-lm-resident, my bug, fixed).
- Cold probes cannot see uninitialized reads on an allocator that zeroes. Use the harness's
  warm shape (`--lanes <preceding>,<lane>`) with `MOJOLEARN_IDENTITY_DUMP_DIR`, or the poison
  build.
- Never rebuild a binding INTO the shared checkout while any run uses it (SIGKILL 137 for every
  later loader); build in a worktree. The shared checkout sits on the Codex branch; origin is
  the truth; pull only into clean worktrees.
- One core on this Mac for everything (Andrew's rule): `OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1`,
  `nice -n 10`, one process, no Mojo builds here. Smoke against a worktree build that carries
  the entries you need (`wt-mamba-poison` under the peer's scratchpad had every binding at
  9ade3e2cd).
- `ls -la` every generated file before `git add`; a 10-second regenerable reference is never
  committed (I committed 6.9 MB by mistake; it is in history).
- Commit with `git add <new files>` then `git commit -o <files>` after `git rev-parse --abbrev-ref HEAD`
  (`-o` refuses untracked paths, which is how this very file failed to commit once); never
  `git add -A`; never rewrite history; every commit message ends with the two attribution lines.
- Andrew: American English, no dashes, no prose colons, never say we are faster, report INERT
  not "unchanged", handoffs only when asked (this one was).
