# lane/merged: integration branch progress

Andrew, 2026-09-28: stop per-branch gating. Every finished lane branch is
merged into lane/merged, checked once globally, and merged into main only
if clean. Source branches are never modified; lane/merged only merges FROM
them.

Base: origin/main at 9a8f9e390 (branch created off it).

## Merge log

One `git merge --no-ff` per branch. Where a local lane ref was ahead of its
origin ref (local merged origin/main but was not pushed), the local ref was
merged; every such local ref was a strict superset of its origin ref.

