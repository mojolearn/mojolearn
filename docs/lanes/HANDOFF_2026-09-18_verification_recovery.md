# Verification coverage: worktree recovery audit

2026-09-18. Continued at origin/main a4dfa844fde4fbd65c7beb802a1daead0f8e1a27
in ~/mojolearn-wt/verification-coverage-continuation. The user requested checking
existing worktrees for good work not yet merged before continuing coverage.

## Findings

The recent next-wheel-coverage, qualify-ordinary-23, causal-lm-distributed-proof,
classical-distributed, multigpu-cv, cv-wheel-verifier, expanded087-build-proof,
cpu-verification-completion and reference-regen branch tips are ancestors of
main. These coverage implementation worktrees contain no uncommitted changes.
Reference-regen does retain untracked run artifacts; ancestry does not bank those
files. Other optimization worktrees also retain untracked logs and evidence.
They were left untouched; do not blindly add provider files that may contain
sensitive material.

release/087-final is different: 20 commits are absent by ancestry. Some fixes
were already ported, including saved-model fault checking, CPU gate time budgets,
checkpointing and concurrent architecture certification. Its frozen references
and release metadata must not overwrite current expanded-source work.

## Recovered release gates

- cfc476b98: Mamba poison negative controls reject setup failures and require
  an actual changed hash or the specific native NaN refusal, with unchanged
  unaffected controls. Existing main reference selection is preserved.
- ddcdc96ce: exact Hopper sm_90 device / sm_90a selected-binary admission and
  tests, without copying the old release status document.
- The release-only Linux qualification runner and smoke checker from
  release/087-final: tier-specific complete binding inventory, compiled-mode
  readback for newer bindings, binary digest checks, and complete exception
  text before expected-refusal classification. The current supplemental
  source-pin and multi-GPU arguments remain intact. Both shell entry points
  are included in qualification source witnesses.

Validation: 48 tests passed; three Linux-only resource tests skipped on macOS.
Shell syntax and git whitespace checks passed. These are gate/orchestration
checks, not a fresh wheel or hardware qualification.

## Retained work needing separate review

lane/identity-record-next (925d1e6a8) contains a historical 178-lane record
not on main: three single-device columns, 39 NVIDIA two-device lanes and 34 AMD
two-device lanes. Its README documents mixed AMD build digests and five missing
AMD lanes. The final branch leaves the gate switch unapplied. Preserve it;
review strict pairing before admitting any evidence or regenerating snapshots.
It is not qualification of today's artifact.

lane/attention-corner-predicate (50133b7a3) has an unmerged documentation-only
attribution audit. Older CPU-training and certification-snapshot branches also
remain outside ancestry; commit counts alone cannot establish missing behavior.
The active tiled-GEMM work in the original checkout was not changed.

## Next coverage work

The expanded handoff remains authoritative for numerical debt: 23 ordinary
holds have AMD captures; NVIDIA, watched installed replay, rebuilt exact-wheel
qualification and physical two-GPU evidence remain owed. The two historical
Apple gp-sample-y/odd disagreements are unresolved. No reference values or
release_qualified flags changed in this recovery. No build, rental or publication
was started. Continue with retained historical evidence review and installed
CPU replay, then the source-pinned build/hardware campaign under its resource
constraints.
