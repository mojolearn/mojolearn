# Support

MojoLearn welcomes bug reports, compatibility results, performance data and
new hardware measurements. Reports from hardware outside the certified matrix
are useful: they are how that matrix grows.

## Start here

Run:

```sh
mojolearn doctor
```

For a bug, reduce the behavior to a small script and create a bundle:

```sh
mojolearn bug-report reproduce.py
```

Review the archive before uploading it. Diagnostics use an allowlist and omit
usernames, home directories, arbitrary environment variables and credentials.
The reproducer and any files supplied with `--attach` are copied verbatim and
may contain private data.

## What the support levels mean

- **Certified** means the exact released configuration has a recorded result
  card on the named hardware and numerical profile.
- **Supported** means the public installation and API are tested and defects
  are release-blocking within the documented boundaries.
- **Experimental** means feedback and patches are welcome, but APIs or bits may
  change and a repair is not promised for a particular release.
- **Compatibility report** means the configuration is outside the current
  matrix. These reports are welcomed and retained; someone willing to test or
  maintain the column can help promote it.

See [SUPPORT_MATRIX.md](SUPPORT_MATRIX.md) for the current evidence.

## Actionable reports

A report can usually be investigated when it contains:

1. a `mojolearn bug-report` bundle;
2. a minimal reproducer using synthetic or publicly shareable data;
3. expected and actual behavior;
4. whether it occurs under `FAST`, `DETERMINISTIC`, `IDENTICAL`, or more than one mode;
5. the last known working release, if it is a regression.

Incomplete reports receive `needs-reproducer`. They remain open for 14 days
for the missing information and may then close automatically. A closed report
can be reopened as soon as a reproducer is available.

Unsupported environments are not rejected. They are labeled
`compatibility-report` so users and prospective hardware-column maintainers
can find them.

## Priorities

1. Security issues, silent wrong results and data corruption.
2. `IDENTICAL` contract violations or crashes on certified configurations.
3. Regressions on supported configurations.
4. Verified performance regressions and incorrect refusals.
5. Experimental configurations, compatibility work and feature requests.

Silent wrong answers outrank crashes. Identity violations outrank performance
regressions because the numerical contract is a public API.

This community project does not promise a response time. That is not a limit
on its ambition: it allows support commitments to grow with the maintainer
team instead of depending permanently on one person.

## Numerical and performance reports

An identity report should attach an identity trace or stage hashes. A
difference between `FAST` runs is not an identity defect unless the operation
documents that guarantee.

A performance report should include the complete benchmark command, shapes,
warm-up policy, synchronization boundary, hardware record, MojoLearn version
and comparison version. Results without these facts may be discussed but do
not change a published performance claim.

## Security

Do not file suspected vulnerabilities or exposed credentials publicly. Follow
[SECURITY.md](SECURITY.md).
