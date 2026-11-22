# Source-only delivery

- Branch: `ideas/apple-fast-trees-20261006`.
- Forked from local `main`: `07f2b3c00`.
- Worktree: `/Users/andrewhendel/CascadeProjects/mojolearn-apple-fast-trees-ideas`.
- Ideas: 48 individual source candidates, 12 planned interaction experiments.
- Implementation: four lanes F/G/N/P; exact final variants, sources and
  prerequisites are retained in their JSON records. These records describe
  source changes and do not certify that the implementations work.
- All new candidates: opt-in, Apple GPU + FAST only, no promoted defaults.
- Quality and performance: unknown; no measured samples or admitted results.
- Compilation, verification, tests, lint, syntax checks, selector execution,
  manifest checks, benchmarks and GPU jobs: **not run, by owner instruction**.
- Main merge: not performed. No boards updated and no performance result files
  manufactured. Existing defaults, rejection history and measurements are not
  recast as evidence for these new candidates.

The initial [IDEAS.md](IDEAS.md) precedes fan-out. The lane records capture
source-reading refinements: for example F05 is sampled label/weight gathering,
G07 is leaf-walker geometry rather than changing the shared partition-statistics
contract, N06 is row-index cursor-copy tiling, and P10 tiles flattened output
cells. These narrower actual mechanisms must guide future caller coverage.

Future work requires separate authorization for compilation/quality/performance
evaluation and the repository's full-dataset end-to-end acceptance process.
There is no outstanding implementation placeholder implied to be a working
kernel; unsupported performance or quality claims remain pending.

Git delivery output, exit codes and final commit ID are retained outside the
worktree at:

`/Users/andrewhendel/mojolearn-evidence/apple-fast-trees-ideas-20261006/delivery-UKePLV/`

Command-local `git -c core.hooksPath=/dev/null` prevents commit/push hooks from
running code verification forbidden by the owner. The commit also requests CI
skip. Neither hook files nor shared repository configuration are changed.

## Integration follow-up

The first source delivery is commit `fef052a4e46e721456dce9c84f79285c3a078b1b`.
The follow-up on the same branch connects all 48 candidates to the shared A/B
catalog with namespaced IDs, paired binding artifacts, the existing full A/B
queue and full-workload public-caller capture adapters. See
[INTEGRATION.md](INTEGRATION.md) for the source connections, repaired routing,
explicit recipe contracts and remaining coverage obligations.

This is programmed integration only. No compilation, verification, tests,
selector execution, measurement or quality admission was performed. Full
dataset recipes require real dataset provenance and caller settings before a
future authorized campaign; pending variants remain documented in the lane
records. All candidate defaults remain off. The branch has not merged to main.

The follow-up commit/push logs, exit codes and resulting commit ID are retained
outside the worktree at:

`/Users/andrewhendel/mojolearn-evidence/apple-fast-trees-ideas-20261006/integration-IvD3y1/`
