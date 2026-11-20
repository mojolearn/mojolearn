# Native label descriptor admission audit

The native-only v2 label descriptor changes were reviewed independently against
their parent implementation. Numerical requests, descriptor slots, device switches,
kernel dispatch and class/weight preparation retain the existing algorithms.
The v2 capsule identity is admitted while holding the GIL; a copied provider
reference pins the compiled image across each GIL-released operation. The v1
compatibility factory remains callable under its old capsule identity, while
new native clients require v2 before releasing the GIL.

The audit found and repairs these concrete wrapper regressions:

- `unique_inverse` converted buffer objects before its count/kind refusal and
  converted later buffers after an earlier null pointer. Shared native shape
  admission and ordered boundary address admission restore the original order.
- `prepare_targets` eagerly converted unused fields and weight metadata before
  label errors. Its legacy boundary now admits and executes labels first, then
  weights, through the same typed preparation body. A no-label/no-weight call
  skips the unused fields. Internal descriptor clients retain their one typed
  request and one native execution; only the compatibility entry needs staged
  UNPACK to retain its original observable errors.
- Device 64-bit gathers, row argmax and class ratios converted addresses before
  shape/empty-result decisions. Shared native shape checks restore lazy UNPACK
  and short-circuit null-address errors.
- `reduce_stat` and `cast_elements` retain their old scalar conversion order
  when the integer-sum or cast device switch is disabled. Their native dispatch,
  device eligibility and host refusal replay are unchanged.

`gather_i32` already eagerly converted its five scalar arguments in both parent
routes; no change was needed there. Typed `HpStat` consumers were checked against
their admitted dtypes: float32/float64 min/max use `.f`, integral predicates and
int32/int64 min/max use `.i`; tuple sum words preserve the existing ABI.

Removed: duplicate host/device `unique_inverse` and preparation UNPACK bodies,
and duplicate scalar shape/refusal definitions replaced by shared helpers.
No tuned kernel, switch, arithmetic operation, numerical allocation or data
copy was removed or changed. No Python callback or fallback was introduced.

Added boundary coverage: `test_labels_legacy_admission.py`, 27 cases each for
host and device (54 expanded cases). These test Python conversion ordering,
ignored arguments and a NaN label error before weight admission. Pure numeric
contracts remain in the native `labels_driver_suite.mojo` (100 fixtures).

No compilation or test execution was performed in this lane. The orchestrator
must compile the shared-label affected product closure together with the v2
descriptor batch. Boundary execution and NVIDIA/AMD measurements remain owed;
this audit makes no runtime, speed, or cross-vendor identity claim.
