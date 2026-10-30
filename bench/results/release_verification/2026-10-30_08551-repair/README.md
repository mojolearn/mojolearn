# 0.8.55.1 maintenance repair candidate

Source freeze: `e7f7417a6abdb51f62bf81e2e601e8c9b1a391f6`, based on the actual published 0.8.55 source `a8255828597a3bd19ac7bb5f657a38d09b8c732f`. Not yet published to PyPI.

## Changes

- Export the existing CPU stratified-group planner and implement its missing integer group/class-count helper.
- Read IVF-PQ output addresses under the GIL, then use native copies during GIL-released CPU work. This repairs the two reproduced x86 ANN crashes.
- Confine the additions to CPU modules; original shared GPU sources are unchanged.
- Admit 36 corrected reference values in Adamax, NAdam and learning-rate scheduling through the supported verifier/identity-admission guard. Retained Apple, NVIDIA, AMD and CPU observations agree; PTX observations are included. Original reports and failures remain retained.
- Accept four-component maintenance versions in release, overlay, verifier-patch and documentation tooling.

## Validation and scope

Both Arm/macOS and x86/Linux compiled both repaired host bindings successfully, including the final algorithm-module layout required by the source guard. On each CPU, the three affected regular lanes pass all nine selected fixtures: 197 applicable cell parts, zero divergent/refused/owed, 162 not applicable. The three selected neural-training lanes pass 54 applicable parts, zero divergent/refused/owed, 189 not applicable. Seven installed regression/optimizer tests pass on each CPU. Verifier self-test and reference coverage checks pass. Reference admission checks 243 parts (54 numerical), with no problems.

These are installed development overlays, not final 0.8.55.1 wheels. Existing GPU evidence and binaries are retained, not described as new executions. No performance measurements were rerun. Complete logs, original failures, source and binary bytes are retained locally and in R2; see qualification.json for hashes and locations.

## Main and remaining release work

Main already contained the splitter repair (`dec6dac9fc`). The ANN GIL fix and maintenance-version support are on main (`3bc7df4a40`, `296dd477b6`), with the public-Array regression correction integrated in `b66b80b52d`. The historical reference table is not copied wholesale onto current main: its evolving numerical implementations require the separately requested frozen reference campaign.

Final wheel assembly, installed-wheel checks and PyPI publication remain pending. The historical reuse planner confirms 276 unchanged bindings, plus the two repaired CPU bindings per CPU target. Its legacy PTX rule requests recompilation merely because the source commit changed; PTX metadata/provenance must be rebound through the supported modern reuse tooling without treating a commit-name change as numerical invalidation. The latest planner also assumes native modules absent from this historical tree. Preserve these findings rather than claiming a final package pass.

The MLSys paper was not changed by this maintenance patch.
