# Contributing to MojoLearn

Contributions are welcome. Bug reproductions, documentation, tests, hardware
cards, estimator work, performance improvements and numerical audits all
matter.

New here? [docs/START_HERE.md](docs/START_HERE.md) is the whole path from a
clone to a merged change, and it is short on purpose. You do not need to read
the rest of this repository's documentation before your first contribution.

## What you actually need

**One GPU. Any vendor. That is the entire hardware requirement.** The vendor
floors and the examples that qualify are the table in
[docs/START_HERE.md](docs/START_HERE.md) section 1; they are not repeated here.
There is no CPU path, so you do need a GPU. You do not need a good one, you do
not need to rent one, and you do not need more than one.

One GPU closes everything except a cross-vendor identity claim: bug
reproductions and fixes, host oracles, separating fixtures, sabotage arms,
documentation, performance work on your own column, and a new estimator in the
`fast` tier. Certificates in this tree are recorded on an M4, an H100 and an
MI325X because that is what it takes to CLOSE a cross-vendor claim, and running
those legs is a maintainer job. **Mark the columns you did not run
`cross-vendor-pending` and stop there.** That is a complete contribution, not a
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
- update the section's `DERIVATION_MAP.tsv` and, when applicable, `NOT_IMPLEMENTED.tsv`;
- preserve required copyright and license notices;
- distinguish transliteration, replacement and original `original` work.

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

## Automatic checks for external pull requests

External PRs to the default branch receive two separate reports from
[External contribution checks](.github/workflows/external-performance.yml).
There is no push trigger. PR authors GitHub identifies as `OWNER`, `MEMBER`
or `COLLABORATOR` are exempt from this additional workflow; existing maintainer
CI and release work continue as before.

- The **admission report** reads changed-file metadata using the policy from
  the trusted base commit. It does not fetch or execute the PR. Existing
  implementation optimizations on its narrow allowlist receive `GPU_PENDING`.
  Changes to infrastructure, dependencies, tests, contracts, public APIs or
  files outside that allowlist receive `REVIEW_REQUIRED`. PR text, labels,
  uploaded results and changes to the gate cannot approve the PR itself.
- The **hosted CPU report** runs base-version packaging/version/CPU-baseline
  checks against the candidate, parses Python and package TOML, checks binding
  shell syntax, and runs trusted comparator negative controls against the
  candidate helpers. It runs on a disposable GitHub-hosted Ubuntu machine
  without cloud/release credentials, caches or a writable token. It does not
  build Mojo, run GPU algorithms or measure GPU performance.

Each report records the exact base/head commits and policy hash. Reports are
available in the Actions job summary and its JSON artifacts. A green CPU job
does **not** close `GPU_PENDING` or permit automatic merging. GitHub's fork
workflow approval policy can hold a first-time contributor's CPU job; this
repository currently uses `first_time_contributors` approval. Workflow code
cannot override that repository/organization setting. Metadata admission
does not require executing the fork's workflow.

### GPU automation integration and current limits

The admission job also emits `external-gpu-request.json`, a typed integration
request containing exact commits, trusted recipe IDs, required vendor/mode
evidence, bounded worker requirements and explicit missing configuration.
Its `enabled` field is currently **false**. No GPU fleet or automatic merge
service is configured by this change. The repository inspection for this work
found no registered Actions runners and no repository Actions secrets;
credentials on a maintainer's local machine are not CI configuration.

The existing [manual GPU workflow](.github/workflows/gpu-validation.yml)
targets self-hosted labels. It must not receive untrusted external PR code.
The [release runner](tools/release_runner.sh) is also outside this trust zone.

A controller can consume the request without asking the owner to start every
run, once an administrator provides these concrete services and limits:

1. A trusted GitHub App/controller with a configured recurring budget and
   quotas, plus disposable NVIDIA, AMD and Apple GPU capacity. Provider keys
   stay in the controller. Guest workers receive no provider, release or
   repository-write credentials and no shared caches. One bounded worker runs
   at a time, then is destroyed with a retained receipt.
2. A controller that independently reloads the current PR metadata and trusted
   base policy, checks the exact base/head commits and recipe IDs, and refuses
   stale or contributor-supplied commands. The JSON artifact is data, not a
   signed approval or executable job definition. Benchmark drivers and evidence
   validators come from the trusted base. Candidate source runs only inside
   the disposable guest; a trusted collector validates outputs outside it.
3. Retained correctness and negative controls, unchanged legacy IDENTICAL
   bytes, cross-vendor certificates, and at least nine interleaved before/after
   samples on the same device. The controller must apply a reviewed,
   fixture-specific regression/performance policy. No time is a pass merely
   because it is faster, and no passing fixture proves every possible input.
4. Exact-head result publication by that App and repository rules that require
   its checks. Existing-feature optimizations can become eligible for automatic
   acceptance only after this proof path is operational; infrastructure,
   dependencies, test and numerical-contract changes still require review.
   Automatic merge remains disabled until those conditions are configured.

Any future self-hosted runner groups must restrict access to trusted workflows
at the administrator level. A `runs-on` line in this workflow cannot prevent a
fork from proposing a different workflow that requests another runner.
GitHub documents the separation of untrusted PR execution from privileged
events in its [secure-use guidance](https://docs.github.com/en/actions/reference/security/secure-use)
and [workflow event reference](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows).

Maintainers can run the admission negative controls locally with
`python3 -m unittest discover -s tools -p test_external_contribution_gate.py`.
No provider account or GPU is needed for those controls.

## Becoming a maintainer

Maintainership is intentionally transferable. Contributors who repeatedly
demonstrate sound review, preserve the numerical and provenance contracts,
and help other contributors may be nominated as reviewers and then
maintainers. See [GOVERNANCE.md](GOVERNANCE.md).

By contributing, you agree that your contribution is licensed under the
repository's Apache-2.0 license and that you have the right to submit it.
