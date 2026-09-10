# Sub-byte layout gate audit — September 10, 2026

Decision: retain and wire `sub_byte_layout_gate.mojo`. It protects live layout
helpers; it is not an obsolete production implementation.

The greedy histogram dispatcher still launches binary and half-byte kernels
in `greedy_search_helper.mojo`. `greedy_sub_byte_excluded_for` in
`checks/kernel_matrix.mojo` returns false. The production half-byte template,
one-byte ladder and hist_2 base obtain their logical replica width from
`replication_lanes_for`; their layouts remain distinct from hardware wave
width. Production comments already reference this gate, but it previously
had no named Pixi task or umbrella invocation.

Run the focused check with:

```sh
pixi run check-sub-byte-layout
```

The task holds the shared build lock. The existing standalone matrix task
also invokes it:

```sh
tools/with_build_lock.sh pixi run check-hardware-matrix
```

Both checks execute host arithmetic only. Adding the invocation to the
matrix file's `main()` preserves the workload of other suites that import
and call `check_hardware_matrix()` directly.

The gate compares shipped layout helpers with arithmetic models and then
checks three deliberately hardware-coupled models:

- Half-byte/binary folding loses replica contributions.
- One-byte private replicas acquire colliding thread pairs.
- Hist_2 shared storage no longer covers the keyed slices.

Those are internal expected violations. The program exits successfully only
when the shipped cases pass and each modeled negative control violates its
invariant; an unexpectedly passing negative control raises an error. There
is no separate expected-failure subprocess. This gate does not demonstrate
that production compile-time assertions reject a sabotaged build, nor does
it qualify GPU execution on AMD or any other accelerator.

Task validation logs are stored under
`bench/results/sub_byte_layout_gate_2026-09-10/`.
