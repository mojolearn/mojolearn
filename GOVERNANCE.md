# Governance

MojoLearn is designed to outlive a single maintainer while preserving an
accurate record of authorship and contribution.

## Roles

- **Contributor:** submits code, tests, documentation, result cards or review.
- **Reviewer:** has demonstrated judgment in an owned area and may approve
  ordinary changes there.
- **Maintainer:** can merge changes, triage releases and nominate reviewers.
- **Release manager:** coordinates a particular release and its certificate;
  this is a rotating responsibility, not permanent ownership.

Numerical-profile changes require approval from a maintainer familiar with
the identity contract. Release publication requires review of the support
matrix, provenance, licenses and recorded evidence.

## Adding maintainers

A maintainer may nominate a contributor who has a sustained record of:

- technically sound contributions or reviews;
- respectful, useful support for other contributors;
- accurate claims about tested hardware and performance;
- care with upstream provenance and licensing;
- care with the FAST/IDENTICAL boundary.

Existing maintainers decide by lazy consensus: the nomination is announced
publicly and is accepted if no maintainer raises a substantiated objection
within seven days. When only one maintainer exists, the nomination should be
recorded in a public issue before access is granted.

## Credit

Project leadership and code ownership can change without rewriting history.
Release authors and substantial contributors are recorded in release notes;
the repository's `CITATION.cff`, `NOTICE`, commit history and archived release
metadata preserve authorship. Papers and reports list authors according to
their actual scholarly contributions rather than repository permissions.

No maintainer may remove historical attribution or license notices while the
attributed work remains in the project.

## Decision making

Routine decisions use lazy consensus. Public API breaks, numerical-profile
changes, licensing changes and governance changes require an explicit written
decision in an issue or pull request. When consensus cannot be reached, the
maintainers document the alternatives and use a simple majority; ties defer
the change.

## Stepping down and succession

A maintainer may become emeritus at any time and retains historical credit.
Before stepping down, a sole maintainer should, where practical:

1. nominate at least two active successors;
2. transfer repository and release-service access using organization roles;
3. document signing, package publication and hardware-certification steps;
4. rotate personal credentials out of project automation;
5. publish the current release status and unresolved blockers.

The goal is stewardship by a team, not permanent support dependence on the
original author.
