# Apple FAST handoff (final state, 2026-10-03 ~01:00Z; cloud code session)

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
| decomp-linalg | 227bf57cc | 9 |
| decomp-sparse | 0f168b374 | 9 |
| depthwise | bdc8c10b1 | 2 |
| ets | e3369d119 | 1 |
| gram | 47ab9b791 | 6 |
| graph | 1fa36a7ec | 1 |
| isotonic-knn | fd9ebd32a | 6 |
| kapprox | 9583fbd26 | 4 |
| kernel | ca0756e50 | 4 |
| linear | 44ec8018d | 15 |
| meta | debb5f743 | 2 |
| nb | 89b9a0694 | 2 |
| neighbors2 | 6d9c7b1f1 | 6 |
| ordered | 3fa785efc | 2 |
| pairlogit | 382a1b234 | 3 |
| prep | 387211293 | 5 |
| prep2 | cc3b27d5f | 7 |
| prep3 | 09e7a7520 | 5 |
| resample | 50b96e795 | 7 |
| rfet-scan | b272364e4 | 0 |
| robust | fdc8259e0 | 2 |
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

## Documented but not written (next passes)
LLE's dense n x n LU in `_lle_smallest` needs a sparse shift-invert/LOBPCG solver (isotonic-knn.md). LU pivot search as a two-launch
grid (decomp-linalg.md). Device-resident CTR columns and `build_ctr_tables` passes (trees-depthwise.md). MCD/elliptic-envelope
resident C-steps, perceptron/sgd-ocsvm (robust.md). Text naive Bayes is upload-bound (nb.md). Symmetric: searcher `subsets`
reuse (trees-symmetric.md). Ordered: score-std readback and per-permutation partition sorts (ordered.md). kernel-shap's O(q^3)
host elimination (shap.md). Neural-network rows on the board are out of this session's scope.

## After the M3 builds
Read `origin/lane/apple-fast-results:docs/apple-fast/m3/build-errors.txt` first (grep, never cat), then results.txt. Every
lane's docs/apple-fast/ab/<family>.md lists its risky compile sites; the recurring one is a bare `buf.unsafe_ptr()` passed to a
helper typed `MutPointer[T, MutAnyOrigin]` (use `.unsafe_origin_cast[MutAnyOrigin]()`). Several lanes replaced the M3 queue's
older env-form lines with define-form lines (ann, core, trees-scan, resample, linear): the manager should re-read the ab files.
Second-dataset lines are listed in each .md as "after a win".
