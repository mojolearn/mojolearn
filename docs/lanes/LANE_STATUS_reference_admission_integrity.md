# Reference admission integrity and CPU TSA controls

2026-09-17. Follow-up to the 246-entry verification audit; user authorized
RunPod and R2 staging. Comprehensive 0.8.7 qualification remains open.

## Closed

- New reference generation requires two identical valid samples, input witnesses,
  held-out witnesses for non-training parts, and matching numerical-property
  protocols. Tests cover absent witnesses, stale protocols and malformed values.
- Probe errors can no longer disappear behind matching saved-model hashes.
  Failed reloads refuse their model part; malformed public results refuse too.
- Legacy tables stay readable and are explicitly identified in public coverage.
  The current bundled table was not silently regenerated. An audit found 708
  historical column references with one recorded sample (177 each for train,
  infer, model and batch). This is a count of column references, not algorithms.
- Native KPSS decision sabotage now reaches select-d. The compile-time flag is
  absent in production and both TSA and forecast bindings expose its readback.
- A clean-source RunPod run at `73a5af1734e24b872274c8830918c155e9534bd6`
  passed all 36 training negative controls: KPSS, select-d, additive and
  multiplicative Holt-Winters; nine fixtures, two repetitions per arm. KPSS
  flips the stationary flag; Holt-Winters uses its arithmetic perturbation.
  This does not prove sensitivity to every operation inside those algorithms.

Strict historical appendix control coverage increases from 44 to 48 of 246
entries (all mapped lanes/parts, at least one qualifying fixture). The new four
lanes each have all nine fixtures. The broader matrix now has 50 lanes with a
seen native build control; its accounting differs from the appendix inventory.

## Execution and evidence

`bench/results/identity_break/2026-09-17_tsa-negative-controls/` retains both
columns, digests, summary, cost and verified teardown receipts. The first run
omitted the core transpose dependency; 18 cells refused and the gate failed.
The diagnostic summary is retained under `reference-admission-probe/` and is
not admitted as numerical evidence. The corrected runner shares production
core with both control arms and fails early if it is missing.

Two-vCPU RunPod instances, at most two compiler workers, single-thread numerical
libraries; local tests use one core. Both pods confirmed deleted. Total measured
cost $0.0063. R2 reused four native artifacts on the corrected run and cached
core. Harness fixtures are deterministic and require no external dataset.

261 focused tests pass. A staging-only reference regeneration (before adding
these new columns) successfully admitted 6,608 parts across 1,651 cells and 103
records under the stricter policy. It was not installed as the release table.

An installed macOS CPU development wheel was tested outside the checkout with
no host, harness or model-path overrides: all 246 inventory entries load; all
36 new controls and the legacy policy notice are present; 18 CTR cells pass
all nine fixtures twice; repeated OLS reports agree; altered held-out provenance
returns INCOMPARABLE. Its receipt and replay script are retained under
`reference-admission-probe/2026-09-17-installed-wheel/`. It reuses previous Mac
native bindings and does not qualify the new native source or final release.
The new native sources were compiled and executed separately on RunPod Linux.

## Still owed

Freeze and qualify the final wheels on every applicable CPU/GPU route; record
missing native controls and batch/other property evidence; finish one-versus-two
GPU coverage (11 of 50 parallel lanes have no historical qualifying pair),
including device-use evidence; then regenerate references, replay installed
final wheels and pass release admission. Historical multi-GPU evidence remains
39 lanes on NVIDIA and nine on AMD. No PyPI release was published here.
