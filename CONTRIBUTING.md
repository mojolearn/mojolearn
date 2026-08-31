# Contributing to MojoLearn

Contributions are welcome. Bug reproductions, documentation, tests, hardware
cards, estimator work, performance improvements and numerical audits all
matter.

New here? [docs/START_HERE.md](docs/START_HERE.md) is the whole path from a
clone to a merged change, and it is short on purpose. You do not need to read
the rest of this repository's documentation before your first contribution.

## What you actually need

**One GPU. Any vendor. That is the entire hardware requirement.**

This is worth stating plainly because the surrounding code can suggest
otherwise. Certificates in this tree are recorded on an M4, an H100 and an
MI325X, and that is what it takes to CLOSE a cross-vendor identity claim. It
is not what it takes to contribute, and it never was.

The real floor is whatever Mojo and MAX support, which is low:

| vendor | floor | examples that qualify |
|---|---|---|
| NVIDIA | Turing | T4, RTX 20XX, RTX 30XX, RTX 40XX, RTX 50XX. A free Google Colab T4 is enough |
| AMD | RDNA2 | Radeon RX 6900, RX 7600, RX 7900, and the integrated 780M / 880M / 890M in ordinary laptops |
| Apple | M1 | any Apple silicon Mac, M1 through M5 |

There is no CPU path, so you do need a GPU. You do not need a good one, you do
not need to rent one, and you do not need more than one.

### What one GPU lets you close

Everything except a cross-vendor identity claim. Bug reproductions and fixes,
host oracles, separating fixtures, sabotage arms, documentation, performance
work on your own column, and a new estimator in the `fast` tier are all fully
closable on a single machine.

A cross-vendor claim is the one thing that needs three vendors, and running
those legs is a maintainer job. Mark the columns you did not run
`cross-vendor-pending` and stop there. That is a complete contribution, not a
partial one. Never infer a column you did not execute.

### Nothing here is built for the maintainer's machine

If you have read a build script and concluded otherwise, it is worth being
explicit. `bindings/build_linalg.sh` pins `--target-cpu apple-m1`, not the M4
it usually runs on, and the Linux builds pin `x86-64-v3`, which is Haswell
2013 and Zen 1 2017 onward. Both pins exist because targeting the build box
shipped a wheel that crashed on other people's hardware, twice. A source build
with no `MOJOLEARN_GPU_ARCHS` set targets YOUR GPU, which is what you want.

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

Maintainers run cross-vendor certification after local review. A contributor
is not expected to own or rent a second vendor, let alone a third.

## Becoming a maintainer

Maintainership is intentionally transferable. Contributors who repeatedly
demonstrate sound review, preserve the numerical and provenance contracts,
and help other contributors may be nominated as reviewers and then
maintainers. See [GOVERNANCE.md](GOVERNANCE.md).

By contributing, you agree that your contribution is licensed under the
repository's Apache-2.0 license and that you have the right to submit it.
