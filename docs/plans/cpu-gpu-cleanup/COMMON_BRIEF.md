# CPU/GPU cleanup: common brief for every sub-lane

Branch base: main 269ffa57. Integration branch: `claude/lucid-ride-74ce6a`.
Worklist: `rows/<lane>.tsv`, made by `split.py` from the `debt` rows of
`tools/hooks/host_routes_baseline.tsv`. In-flight rows (`inflight@...`) belong
to open PRs, so leave them alone.

## Rules

- **Each operation runs fully on the GPU on a GPU install, or fully on the CPU on a CPU-only install.**
  No host step inside a GPU fit, transform or predict. No size threshold, env
  switch or route table that sends GPU work to the host. Never fall back to
  the CPU to win time.
- **No serial GPU work:** no one-thread, one-block, tid==0 or per-sample default
  loops. Use grid-wide kernels with fixed-order (blocked, then fixed-tree) reductions.
- **Bits:** old bits don't matter. Bits must match across NVIDIA, AMD, Apple and
  the host column. When you change a kernel's fold order, change the CPU
  reference / host column (the CPU-only install path) the same way, in the same
  commit, so the two stay equal.
- CPU-only installs keep their host bindings. Only the routing from GPU code to
  the host goes away.
- No A/B define this round: delete the route, its env switch, threshold, host
  import and route-only tests directly. Leave test files that check results;
  delete only tests whose sole subject is the host route.
- **Run nothing.** No compile, no tests, no pixi, no python, no checker, no network.
  Another session builds and runs. Only read, edit and use git.
- Do not edit `tools/hooks/host_routes_baseline.tsv`. The orchestrator prunes it once after merging.
- Edit only files under your lane's prefixes (listed in `split.py`). If a fix
  needs a change in another lane's file, don't make it. Write it under
  CROSS-LANE in your final reply (file, line, what to change).
- Don't touch `docs/apple-fast/` or files that only the Apple FAST peer owns (`*_fast.mojo`).
- Match the surrounding code style. Commit after every coherent edit, using clear messages that name the lane.

## Hard spots, to do properly rather than skip

The UMAP optimizer, the HDBSCAN dendrogram and labelling, the LU one-block
launches, the neural-pass144 eigensolver, and the serial chains (`serial-launch`, `tid0-loop` rows).

## Final reply (short)

- branch and head;
- one line per change, as `file:line` plus the cause;
- bit changes (which outputs change and on which path);
- rows you could not clear, with the reason;
- CROSS-LANE asks;
- RUN OWED: the compile and identity checks the run session should do (lq ID lines for the lanes and datasets touched).
