# Apple FAST: brief for the cloud peer (2026-10-02)

Start from PLAN.md in this folder (PLAN-classical.md and PLAN-trees.md have the details).

## Who does what
- **Local session** owns `lane/apple-fast`, the M3 Ultra and its queue, all builds and runs, and merges to main. It works the **trees** items (PLAN.md 2, 5, 6, 9, 11) and reads the queued classical A/Bs (PLAN.md 0 and 1).
- **Cloud peer** works the **classical** items: PLAN.md 3 (FAST slower than IDENTICAL: theta, rbf-sampler, pca istella), 4 (shared FAST grid Gram for Lars/LassoLars/RidgeClassifier/RidgeCV/LDA/QDA), 7 (PCA one-block Jacobi), 8 (isotonic parallel PAVA, knn-imputer/LLE through fast_mma_knn), 10 (prep host overhead). Take them in that order.

## Branches
- One branch per family: `lane/apple-fast-<family>` (for example `lane/apple-fast-gram`), based on `origin/lane/apple-fast`. The `lane/apple-fast` prefix is required: the M3 queue refuses other branches.
- Never push to `lane/apple-fast` itself.
- Each change sits under the FAST numeric mode with a define or env switch that defaults off. IDENTICAL (default mode) bits must not move. Read the repo CLAUDE.md rules: GPU path pure GPU and parallel, no host steps, no one-thread or one-block defaults.

## Asking for runs
Commit `docs/apple-fast/ab/<family>.txt` on your branch, one queue line per row, in this form:

    CMD lane/apple-fast-<family> <tag> AFC_FAMILY=classical bash tools/afc_ab.sh <tag> <algo> <dataset> 3 3 - <ENV_ARM=1>
    CMD lane/apple-fast-<family> <tag> AFT_OUT=$HOME/aft-ab/<tag> bash tools/aft_ab.sh <binding> <lane> <dataset> 3 "" "-D <DEFINE>"

Datasets: taxi, istella (rows-full); identity copies in rows-small. The local session compiles your branch, queues the lines, and writes results to `docs/apple-fast/ab/<family>.results` on lane/apple-fast. A change is kept only when the M3 A/B is faster and held-out quality stays within FAST run-to-run spread; the switch is then deleted and the change becomes the FAST default.

## Rules
- Never time our CPU, never race `MOJOLEARN_VENDOR=cpu`.
- FAST never lowers quality.
- No new machines of any kind.

## Reading results
Branch `lane/apple-fast-results` holds `docs/apple-fast/m3/results.txt` (every M3 result line, numbered) and
`docs/apple-fast/m3/queue.txt` (the queue with line numbers). It refreshes every 10 minutes:
`git fetch origin lane/apple-fast-results && git show origin/lane/apple-fast-results:docs/apple-fast/m3/results.txt | grep <tag>`.
New `docs/apple-fast/ab/*.txt` files on your `lane/apple-fast-*` branches are picked up automatically on the same cycle.
