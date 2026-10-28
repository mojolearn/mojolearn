# Lane briefs

The orchestrator writes every lane brief before launching the lane, as
`~/mojolearn-evidence/briefs-<YYYYMMDD>/<lane>.md` (plus a `COMMON.md` with the rules every lane of that day shares).
After a crash a lane is relaunched from that file plus `git status` and `git log origin/main..HEAD` in its worktree.

The briefs are mirrored here as `docs/briefs/<YYYYMMDD>/` so the instructions behind each branch are versioned
next to the code. The evidence copy is the working copy; the repo copy is a snapshot taken when the mirror commit
was made. Edit the evidence copy, then copy it here again.

Lane handoffs (`~/mojolearn-evidence/<lane>-<YYYYMMDD>/HANDOFF.md`, written by each lane at the end) are mirrored as
`experiments/handoffs_<YYYYMMDD>/<lane>.md`. Both are dated directories that accumulate: a new day gets a new
directory, and older ones are never deleted (`docs/EVIDENCE_LAYOUT.md`).

Briefs carry no credentials: secrets stay in `~/.mojolearn_r2` and provider config, never in a brief.
