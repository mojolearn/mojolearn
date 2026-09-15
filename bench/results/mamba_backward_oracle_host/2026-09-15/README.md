# The Mamba-1 host backward oracle against the device VJP (2026-09-15)

`mamba/checks/mamba_backward_host_oracle_check.mojo` (pixi task
`check-mamba1-backward-oracle-host`), on the Apple M4, one core, shared
machine, no GPU. It runs the host backward oracle
(`mamba/checks/mamba_backward_oracle.mojo`) and the device prefill VJP as
`tools/mamba_host_gen.py` writes it for the host (`mamba/host/gen/`, the pass
that reads IDENTICAL x4 against the 166-lane record, see
`bench/results/identity_break/2026-09-15_cpu-mamba/`) on the 16 corpus cases
and compares 29 tensors per case bit for bit (four forward stages, eleven
backward stages, thirteen public gradients and the B and C stage gradients).

| file | oracle | result |
|---|---|---|
| `check.unfixed-oracle.txt` | as on main before this change | `FAIL: 151 tensors differ`, exit 1; first seams `bwd.dz` and `bwd.ddelta` |
| `check.fixed-oracle.txt` | this change | `PASS`, exit 0, 464 of 464 tensors equal |

Which side was wrong: the oracle, not the device. The device pass is the one
the three GPU columns and the CPU column agree on, and it follows the backward
plan's amended rows (the plan was archived as
`archive/plans/mamba/IDENTICAL_BACKWARD_PLAN.md` and removed in e08cda5bc; its
rows B7 and B18 and DEVIATIONS 1082, 1083 and 1085 are read from e08cda5bc^).
The oracle carried three readings the plan had rejected:

- B7, `silu'`: the oracle fused `1 + v*(1 - sig)` into one multiply-add (three
  roundings, its free choice FC1, which named "the plan amending the count" as
  its falsifier). The plan's DEVIATION 1085 says four roundings, the
  `transformer_backward.mojo` chain, which the device's `_silu_prime`
  transcribes. Moves `bwd.dz` and `bwd.dconv`.
- B18, `ddelta`: the oracle folded the B path and the A path as two chains
  joined by an add (FC2). DEVIATION 1083 pins ONE chain, per n the B term then
  the A term; the two-fold reading is the device's `SAB_BWD_DDELTA_TWO_FOLDS`
  sabotage arm.
- T1, the reverse recurrence: the oracle folded a stored `+0.0` seed at the
  walk's first step, which turns a `-0.0` contribution into `+0.0`
  (`bwd.dh`, 186 of 4096 cells on `adv_gate_saturation_b1_l8_d16`, no public
  gradient moved). DEVIATION 1082 makes the seed an omitted operation; the
  stored-seed reading is the device's `SAB_BWD_T1_SEED_ADD` arm.

What relied on the oracle: only `mamba_check.mojo`'s opt-in
`MOJOLEARN_MAMBA_GRADIENT_DUMP` export, which `tools/mamba_gradient_oracle.py`
compares to a float64 reference at a tolerance; no bitwise gate read it, so
the mismatch had never been seen. `mamba/README.md` said IDENTICAL validation
matches a device card to "the pinned Mojo host oracle bit for bit"; that
sentence is corrected in the same change.
