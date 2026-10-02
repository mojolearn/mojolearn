# Apple FAST handoff (final state, 2026-10-03 ~03:00Z; cloud code session)

For the next code session or the M3 manager. Rules unchanged (CLAUDE.md, docs/apple-fast/NEXT_PASS.md, NEXT_PASS_TREES.md on
origin/main). Lane briefs, the gap table and the GPU-only audit output are in this directory on branch
`claude/charming-volta-y568xd`. Copy the directory to `~/mojolearn-evidence/briefs-2026-10-02/` (home is /root on the cloud box;
the Write tool resolves `~` to /home/user, so use absolute /root paths) before relaunching lane subagents.

Main was never pushed to. Every lane below has origin/main **2d7eade5b** merged in (the lane/cg-integrate merge), merges cleanly
with it (`git merge-tree --write-tree`), passes `tools/hooks/no_host_routes.py` in diff mode with zero findings, and has a
clean, fully pushed worktree. Every change is FAST + Apple only behind a default-off `-D MOJOLEARN_*` define; IDENTICAL compiles
main's code unchanged (static check stated in the commits; the M3 runs the real identity check). No lane converted a FAST path to
CPU: the audit found and fixed three leftovers (core env switches, trees-ensembles host fold loops, ann one-block scan).

## Lane heads (all `lane/apple-fast-<family>`; request lines = `CMD` lines in docs/apple-fast/ab/<family>.txt)

| family | head | request lines |
|---|---|---|
| ann | 68cc6b6bd | 14 |
| bayes | 1a0bb2b2b | 1 |
| cluster | aaef7b261 | 2 |
| cluster2 | 775d5b7be | 9 |
| core | d22a7494c | 7 |
| dart | 02362668e | 2 |
| decomp-linalg | f231ebab4 | 11 |
| decomp-sparse | 1b2c8f955 | 9 |
| depthwise | bdc8c10b1 | 2 |
| ets | e3369d119 | 1 |
| gram | 47ab9b791 | 6 |
| graph | 1fa36a7ec | 1 |
| isotonic-knn | fd9ebd32a | 6 |
| kapprox | 2157ed10b | 6 |
| kernel | ca0756e50 | 4 |
| linear | 44ec8018d | 15 |
| lle | f2ea1ecb5 | 1 |
| meta | debb5f743 | 2 |
| nb | 9f2b71471 | 4 |
| neighbors2 | 6d9c7b1f1 | 6 |
| ordered | 3fa785efc | 2 |
| pairlogit | 382a1b234 | 3 |
| prep | 387211293 | 5 |
| prep2 | cc3b27d5f | 7 |
| prep3 | ec65873e3 | 5 |
| resample | 50b96e795 | 7 |
| rfet-scan | b272364e4 | 0 |
| robust | cfdb95e48 | 5 |
| select | 4743bb576 | 4 |
| shap | 13343dd51 | 3 |
| trees-depthwise | da8ad083f | 5 |
| trees-ensembles | e9803ee25 | 8 |
| trees-io | dfbd3f61f | 5 |
| trees-scan | 43430ca0f | 2 |
| trees-symmetric | ce517b4b3 | 5 |
| tsa | 9ee4a2ed2 | 3 |
| tsa2 | 22edf8f13 | 3 |
| yetirank | c9eb1ce14 | 2 |

rfet-scan carries its A/B rows in docs/apple-fast/PLAN-trees.md rows 84-86 (already queued). bayes head 1a0bb2b2b includes a
commit pushed by the M3 side after ours.

## Superset branches (overlaps settled; the later one carries the earlier)
x_prep: prep3 > nb > select > prep2; prep3 also carries meta. x_linear: kernel > gram; bayes, linear, core mutually clean.
x_cluster: cluster2 > cluster. x_decomp: decomp-sparse > decomp-linalg. x_neighbors: neighbors2 > kapprox; isotonic-knn,
graph, kapprox, neighbors2 mutually clean. gbdt: yetirank > trees-yeti; trees-scan dropped its scan (trees-depthwise keeps
`GBDT_CTR_FAST_SCAN`); trees-symmetric, depthwise, ordered, pairlogit, dart, trees-depthwise mutually clean.

## Dropped because main already does it (no request line)
kmeans device scale (core), TSNE Z sum (ann), PLS dead columns and EIGH_FAST_RR and both LU switches (decomp-linalg), RP_DIRECT /
MDS diag / Isomap kNN (decomp-sparse), pagerank reduce and MMA kNN (neighbors2; the MMA route lives on isotonic-knn), BAG_SESSION
(trees-ensembles), MINMAX/MULTILABEL env arms (prep). gaussian-rp and ocsvm need a board re-run of main, not a lane.

## Follow-up passes done after the first handoff
- lle (new lane, carries isotonic-knn and nb): sparse LOBPCG for LocallyLinearEmbedding, no dense n x n (`-D MOJOLEARN_LLE_SPARSE_EIG`); if it
  does not settle in 600 iterations the fit falls back to main's dense route (time loss only). Next step if the A/B shows the fallback: a
  CG-based preconditioner or a larger block (lle.md).
- decomp-linalg: main already had the two-launch grid pivot; `-D MOJOLEARN_LU_FAST_PIVOT_GRID` fuses the pivot finish into the swap grid.
- robust: `-D MOJOLEARN_MCD_DEVICE_CSTEPS`, every MCD candidate's C-steps on the device; the Python reweighting tail stays main's.
- nb: `-D MOJOLEARN_NB_TEXT_CSR`, text naive Bayes fit and scoring on the CSR arrays (no densified upload). The bench driver now hands CSR
  to every sparse-capable arm; **the stored sklearn text-NB times were measured on the dense block and must be re-run on CSR** (flagged at
  the top of nb.txt).
- gaussian-rp and ocsvm: baseline-only request lines (arm A, no switch) on kapprox.txt and robust.txt re-measure main's existing device
  routes; the board rows predate them.

## Still open (documented, not written)
Device-resident CTR columns (trees-depthwise.md); perceptron / sgd-ocsvm (robust.md: SGD's serial batch chain); symmetric searcher
`subsets` reuse (trees-symmetric.md); ordered score-std readback and per-permutation partition sorts (ordered.md); kernel-shap's O(q^3)
host elimination (shap.md); MCD's Python reweighting tail (robust.md). Neural-network rows on the board are out of this session's scope.

## Superset branches, updated
prep3 > nb > select > prep2 (+ meta); lle > isotonic-knn (+ nb's x_decomp registrations); kernel > gram; cluster2 > cluster;
decomp-sparse > decomp-linalg; neighbors2 > kapprox; yetirank > trees-yeti. The installed git hook copy in .git/hooks must match main's
tools/hooks (reinstall with tools/hooks/install.sh after main moves; a stale copy refused main itself once).

## After the M3 builds
Read `origin/lane/apple-fast-results:docs/apple-fast/m3/build-errors.txt` first (grep, never cat), then results.txt. Every
lane's docs/apple-fast/ab/<family>.md lists its risky compile sites; the recurring one is a bare `buf.unsafe_ptr()` passed to a
helper typed `MutPointer[T, MutAnyOrigin]` (use `.unsafe_origin_cast[MutAnyOrigin]()`). Several lanes replaced the M3 queue's
older env-form lines with define-form lines (ann, core, trees-scan, resample, linear): the manager should re-read the ab files.
Second-dataset lines are listed in each .md as "after a win".
