# NearestCentroid fixture lifetime repair

Root's `wave8r2` CPU source compiled successfully. Its execution then reported
fixture 14 (`quarter bits`) and fixtures 17/18 (prior refusal text/order), and
the process terminated with exit `-11`. The complete original output remains
at `~/mojolearn-evidence/python-mojo-migration-20261110/wave8r2-cpu-run.log`.
This is a failed runtime batch, not a compile failure or a partial pass.

Source review found missing ownership pins in
`checks/native_tests/nearest_centroid_plan_suite.mojo`. Fixture 14 converted its
source List pointer to an integer and never used that owner afterward.
Refusal fixtures 16–18 did the same. Uniformity fixtures 8–11 likewise had no
owner use after their second raw-address check. Mojo ASAP destruction can end
these List lifetimes before the native call reads their untracked addresses.

The test-only repair names each source owner after the native call and retains
the existing numerical/error oracles. Quarter narrowing is exactly `0x3e800000`;
the product checks negative priors before fsum, and positive fsum overflow maps
to the existing math-range error. No evidence warrants changing these expected
values or any product kernel/plan arithmetic. Refusal failures now print their
expected and actual text (or that no refusal occurred).

The dangling sources can explain the observed fixture failures and can cause
a crash, but the process exit cause has not been established by this source
review. Root is isolating that crash separately. `git diff --check` passed;
this lane performed no compilation or test execution. Root must rebuild/run
the repaired native fixture suite before claiming the failures are resolved.
