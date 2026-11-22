# Mamba2 dt clamp CPU completion

The complete Mac column at `b74de8f54` passes all nine fixtures twice:
training, inference, batch, step/full and sampler/replay are stable; model
serialization is explicitly N/A. It resumed seven completed cells after the
600-second deadline under the exact same source, package, native binding,
environment and protocol signature. The incomplete checkpoint is not admitted.
Mac bindings are reused except the fresh training binding.

The Linux pair at `dcddb8345` detects all nine native training perturbations.
Its clean training hashes match the complete Mac column exactly. Its missing
neural binding caused inference/property refusals: the stricter batch evaluator
correctly reports passed=false for that incomplete validation. Those failures
are retained, not erased or counted as full Linux qualification. Clean full
property evidence comes from the Mac record. The originating Linux pod was
still running other lanes when these completed files were fetched; its final
teardown will be recorded with the full neural batch. Production algorithms and
fixtures are unchanged between these sources; subsequent harness changes only
preserve explicit numerical-oracle failure diagnostics.
