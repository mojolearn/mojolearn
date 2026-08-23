# Contributing to MojoLearn

Contributions are welcome: bug reproductions, documentation, tests, hardware
cards, estimator work, performance improvements and numerical audits all
matter. You do not need access to all three GPU vendors to contribute.

## Before opening a pull request

1. Open an issue for changes that alter a public API, numerical profile or
   algorithm boundary.
2. Keep the change focused and preserve unrelated work in the tree.
3. Add the smallest test that would have failed before the change.
4. Run the relevant local checks documented by the owning module or `pixi`
   task.
5. State which hardware and numerical modes you actually ran. Unrun columns
   should be marked `cross-vendor-pending`, not inferred.

## Ported and original code

MojoLearn mirrors algorithms from CatBoost, cuML, cuVS, RAFT and FAISS under
their licenses. A contribution derived from upstream code must:

- name the exact upstream file and commit;
- update the section's `PORTED_MAP.tsv` and, when applicable, `UNPORTED.tsv`;
- preserve required copyright and license notices;
- distinguish transliteration, replacement and original `mojo_only` work.

Do not paste code from a source whose license is incompatible or unknown.

## Numerical changes

Any change capable of moving `IDENTICAL` bits must name the numerical-profile
clause or `IDENTITY_PATHS.md` row it affects and provide one of:

- evidence that the change is bit-inert;
- a separating fixture and the required profile-version decision; or
- a named refusal that prevents an unsupported claim.

Changes to reductions, RNG mapping, arithmetic contraction, denormal policy,
tie-breaking, serialization and dispatch never merge solely because ordinary
correctness tests pass. They require explicit review of the numerical DAG.

When a numerical pin is added, the test must first show that its fixture can
distinguish the pinned and unpinned spellings. A passing random hash that
cannot separate them is not evidence.

## Pull-request evidence

The pull-request template asks for:

- upstream provenance or a reason the code is original;
- affected public APIs;
- FAST and IDENTICAL effects;
- tests and adversarial fixtures;
- hardware columns actually exercised;
- performance evidence when a performance claim changes.

Maintainers can run scheduled cross-vendor certification after local review.
A contributor is not expected to rent three GPUs.

## Becoming a maintainer

Maintainership is intentionally transferable. Contributors who repeatedly
demonstrate sound review, preserve the numerical and provenance contracts,
and help other contributors may be nominated as reviewers and then
maintainers. See [GOVERNANCE.md](GOVERNANCE.md).

By contributing, you agree that your contribution is licensed under the
repository's Apache-2.0 license and that you have the right to submit it.
