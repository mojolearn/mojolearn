@AGENTS.md

# Claude sessions in this repo

The project rules are in [AGENTS.md](AGENTS.md), the one rule file for every agent. The first line of this file
imports it, so Claude Code loads it with this file. An agent whose harness does not expand `@` imports must read
`AGENTS.md` in full before doing anything else. This file holds only what is specific to Claude sessions and their
subagents; a rule that also binds Codex belongs in `AGENTS.md`, never here.

## Who does what

- The orchestrator session (Fable) reads the logs with grep and short tails, judges, plans, merges and decides. It
  does not hand its judgment to a lane.
- Implementation lanes are Opus subagents: one lane, one worktree, one brief. They write code and push; they do not
  compile, test, measure or rent (AGENTS.md "Lanes").
- Fable subagents are for work that is judgment: ideas-only and design lanes, audits, and any change to the rule
  files themselves. Andrew asks for a Fable implementation lane by name when he wants one.
- Only the orchestrator compiles, plus the one integration compile-fix lane per integration branch that it names
  in that lane's brief (AGENTS.md "Lanes").

## Briefs

- The orchestrator saves every lane brief as `~/mojolearn-evidence/briefs-<date>/<lane>.md` before launching the
  lane, with one shared `_common.md` the lane reads first.
- After a crash, a lane is relaunched from that file plus `git status` and `git log origin/main..HEAD` in its
  worktree. A lane that goes silent is checked in its worktree (last commit, untracked files) and relaunched in the
  same worktree, never duplicated in a second one.
- Each finished lane's result and leftovers go into a follow-ups file beside the briefs, so every leftover gets an
  owner before a freeze.

## Subagents and context

- Never read a subagent transcript or a `tasks/*.output` file. A lane's final reply and its pushed branch are the
  record.
- Do not poll a running subagent. Check the artifact it is producing (a build log's counts, a branch head).
- Check `ListAgents` for live sessions before resuming a handoff; the author may still be running.

## Memory

- The file memory under `~/.claude/projects/.../memory/` is Claude's. Save there only what changes behavior in
  future sessions and is not already in `AGENTS.md`, the repo or git history.
- A standing rule Andrew gives belongs in `AGENTS.md` (so Codex has it too), with a one-line memory pointer at most.
