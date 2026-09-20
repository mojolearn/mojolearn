# Worktree hygiene audit and conservative cleanup

Date: 2026-09-20, America/New_York. Base: `origin/main` at
`c0c702d6374b75123274cf7e22fd199b53c3f064`.

This cleanup used `git status --porcelain=v1 -uall`, worktree lock records,
and `git cherry origin/main HEAD`. Ancestry alone was never accepted as proof
that content shipped. No branch was deleted, and neither `--force` nor a glob
was used. Live-agent, root integration, dirty/untracked, locked, cloud/evidence,
detached release, and every `git cherry` `+` lane was preserved.

The earlier read-only audit observed 53 worktrees. Other sessions reduced that
set before this cleanup began. After creating this workflow worktree, a fresh
28-worktree snapshot had SHA-256
`943bf7de8130cbda1f37d3a0c5e7db10b1b278f79c87ac0fa483cc95d598bd59`.
It was recaptured immediately before removal and matched byte-for-byte.

## Removed after immediate recheck

| Path | Branch | HEAD | Proof |
|---|---|---|---|
| `/private/tmp/mojolearn-decode-million-next` | `agent/decode-million-next` | `3aac9357132b6a71a6665baaae7ba45e75f0e0e5` | clean, unlocked, no `git cherry +`, completed owner |
| `/private/tmp/mojolearn-token-million-next` | `agent/token-million-next` | `1bf16bb8428d935c673e449db2d03c54b467256a` | clean, unlocked, patch-equivalent `git cherry -`, completed owner |
| `/Users/andrewhendel/mojolearn-wt/probabilistic-million-cpu` | `lane/probabilistic-million-cpu` | `0772896c195ec641278295a4ca05667ebe209bf0` | clean, unlocked, patch-equivalent `git cherry -`, no live owner |

## Preserved

Twenty-five worktrees remained, including this audit worktree. The important
preservation classes were:

- root integration: the shared project checkout and `handoff-sep18`;
- live/ambiguous agent lanes: all neural, GPU/tree, Claude, and active workflow lanes;
- dirty evidence: decomposition-linalg, GPU production/static, gemm fold,
  GPU tree R2, Mamba2 sync, neural GPU, weighted runtime, and the shared root;
- locked Claude worktrees;
- unique patches: classical GPU hotspot, Bayesian calibration, classical
  million sweep, attention corner predicate, both release worktrees, and any
  other lane producing a `git cherry` `+` line;
- cloud/evidence and detached release lanes regardless of apparent cleanliness.

`tools/worktree_prune_check.sh` is a reusable read-only gate. Its
`ELIGIBLE_PENDING_OWNERSHIP_CHECK` result is deliberately not authorization to
delete: live-agent, lease, and evidence ownership still require an external
check immediately before an explicit `git worktree remove PATH`.
