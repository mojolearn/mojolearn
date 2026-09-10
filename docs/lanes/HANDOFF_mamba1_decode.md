# Mamba-1 decode wiring, 2026-09-09

The public Mamba1Block.step binding now calls the device overload of
`mamba_simple.mamba_step`. It delegates to the existing certified block
at L=1, so decode and prefill retain one GPU arithmetic implementation.
The module also exposes device cache allocation; the host overload remains
the independent reference. The old executable entry was renamed to
`check_reference_decode` so the module can be imported.

Run `tools/with_identical_mode.sh pixi run check-mamba-decode`. This new
task runs eight host reference cases, five device decode/prefill cases,
and a B=2/L=16 prefill-to-step continuation. Every stage and final cache
compares bitwise; zero cache initialization and invalid input extents and
batch sizes are checked. `probe` exercises all 18 stage comparisons;
`sabotage` and `sabotage-window` succeed only when the intentional
arithmetic/state perturbations are detected on the required cases.

Apple: task, probe, both negative controls PASS; rebuilt IDENTICAL Mamba
binding passes the Python surface (102 checks, zero failures). Logs in
`bench/results/mamba1_decode_wiring_2026-09-09/`. NVIDIA L40S: the combined
Mamba1/Mamba3 extension compiled, decode task and full Python surface
passed in the Mamba3 performance lane. This is wiring/correctness work,
with no new Mamba-1 performance claim.
