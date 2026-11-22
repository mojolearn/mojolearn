# Lane briefs

The orchestrator writes every lane brief before launching the lane, as
`~/mojolearn-evidence/briefs-<YYYYMMDD>/<lane>.md` (plus a `COMMON.md` with the rules every lane of that day shares).
After a crash a lane is relaunched from that file plus `git status` and `git log origin/main..HEAD` in its worktree.

Briefs and lane handoffs are no longer mirrored into the repo (2026-11-21 tree clean-up): they live in
`~/mojolearn-evidence/` and R2 only. The last mirrors (`docs/briefs/20261028/`, `experiments/handoffs_20261028/`)
were moved to `~/mojolearn-evidence/repo-archive-20261125/` (see its `MANIFEST.tsv`) and stay in git history.

Briefs carry no credentials: secrets stay in `~/.mojolearn_r2` and provider config, never in a brief.

On 2026-10-28, ten handoff documents last changed before 2026-10-25 were
removed from the current checkout at the owner's request. Historical
qualification metadata can still name those old paths; the qualification
records themselves were preserved. Read an old handoff with
`git show dacfa820ad64864062d044ecd062f799fedf9572:<path>`.
This cleanup did not remove identity records, model fixtures, benchmark
results, handoff-related executable tools, or the external evidence folder.
