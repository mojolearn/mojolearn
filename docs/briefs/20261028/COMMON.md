# Common rules for every lane (2026-10-28, orchestrator mojolearn-c6)

Read first: /Users/andrewhendel/CascadeProjects/mojolearn/CLAUDE.md, AGENTS.md (sections "Compile and merge",
"Provider migration CPU builds and peer autonomy", "Provider migration delivery", "Logs and context"),
and PROVIDER_MIGRATION_HANDOFF_2026-10-28.txt (root handoff; your scope's section + FINAL ADDENDUM), plus
~/mojolearn-evidence/provider-migration-resume-20261027/ROOT_HANDOFF_LATE_UPDATES_20261028.txt.

Objective of the program: remove Python from the runtime. The whole public operation (validation, fitted
checks, binding selection, preparation, execution, result conversion) lives in Mojo/native providers;
Python is only the thin API surface. Narrow documented exceptions (sklearn tag metadata, documented Mojo-bug
exceptions) stay as they are. Native code must never call back into arbitrary Python hooks/protocols.

LOGS / CONTEXT (Andrew, mandatory; propagate to any nested subagent):
- Never print a whole log, build output or JSON report. Use `rg -n -m 20 -C 3 --max-columns 240`,
  `tail -n 20`, `grep -E 'error|FAIL|refused'`. For a build: `cmd > build.log 2>&1; echo rc=$?; grep -m 5 -B 2 -A 8 'error' build.log`.
- Never read a subagent transcript or tasks/*.output. Long results go to a file under ~/mojolearn-evidence/<lane>/; report the path.
- Keep the real exit status; filtered output is not proof of success. Report failures and skipped work plainly.

COMPILES (allowed for this migration; supersedes "lanes never compile"):
- A small affected compile of the paths you changed is REQUIRED before anything is called green. Model your
  compile on an existing receipt (e.g. ~/mojolearn-evidence/peer1-ivf-radius-pcs-20261028/compile-command.sh:
  `pixi run mojo build -j <jobs> --emit shared-lib -D MOJOLEARN_NUMERIC_IDENTICAL=1
  -D MOJOLEARN_COLUMN_CPU -I . -I bindings <entry.mojo> -o out.so`), and on the experiments/*/compile.json
  receipts of your own scope. Use standard optimized compilation, including for small development
  libraries. Correction (2026-10-29): the former O0 example produced an incorrect zero in a
  reported image despite explicit buffer lifetime protection; normal optimization returned
  the correct result. The compiler internal cause remains unconfirmed. Preserve earlier O0
  records as compile evidence, not proof of correct execution. Save a receipt (source commit, entry, command, compiler version, jobs, exit, log path,
  binary sha256) under experiments/<scope>/ and the full log under ~/mojolearn-evidence/<lane>/.
- On the Mac: wrap every compile in `bash ~/mojolearn-evidence/compile_slot.sh <cmd>` (4 machine-wide slots,
  nice 19) and use -j 10 at most. Check `df -h /` before compiling; stop if under 8 GiB free.
- For big closures rent ONE RunPod CPU box (no GPU) with `bash tools/runpod_cpu_leg.sh --lane <tag> --cmd '...'
  --worktree <wt> --rent` (run without --rent first for a dry run; `list` shows live pods). Use all allocated cores
  (affinity/cgroup), 60-minute deadman, retain logs/binaries on the Mac (and R2 if the tool does) before reaping,
  then `reap` your pod. Never touch a pod you did not create. NEVER create any Apple machine of any kind.
- A compile is not runtime success. Never claim runtime, identity or GPU verification. No runtime/identity campaigns.

GIT:
- Work ONLY in your assigned worktree/branch. Never touch another worktree. Never `git stash`, `reset --hard`,
  `checkout --` or rebase away existing edits; the dirty files in your worktree are a peer's unfinished draft that
  you are finishing: read `git status`, `git diff`, the lane's CHECKPOINT/HANDOFF/CURRENT files under experiments/
  and ~/mojolearn-evidence/ before changing anything.
- Commit after every coherent step; push often with a normal `git push` (hooks run; never --no-verify; never force).
  If the push hook refuses (no_compute_in_shells, no-host-routes), fix the cause (usually driver-loop tags/comments), never bypass.
- Do NOT merge to main. Main integration is serialized by the orchestrator. Deliver a green head.
- Commit messages end with: Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>

FINAL REPLY (short, no narrative, no pasted diffs):
- branch and head sha; what the head contains (one line per change, file:line for the cause);
- compile receipt path + exit + what closure it covers + what is NOT covered;
- rented box id (if any) and confirmation it was reaped; evidence dir;
- what remains pending / held, exactly.
Also write the same as ~/mojolearn-evidence/<lane>/HANDOFF.md.
