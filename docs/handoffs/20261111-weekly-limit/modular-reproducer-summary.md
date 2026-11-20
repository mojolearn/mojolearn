# Modular asks re-checked on Mojo 1.2.0 / MAX 26.7.0 (2026-11-11, Claude session mojolearn-c6)

Project pins Mojo 1.0.0 / MAX 26.5.0. Scratch env: env/ (pixi, Mojo 1.2.0 2016d0cd, MAX 26.7.0). Sources in src/,
build and run logs in out/, compile table in results.tsv. Laptop compile slot used (tools/compile_slot.sh).

| Blocker | 1.0.0 | 1.2.0 | Action |
|---|---|---|---|
| IPCA nested List pointer captured by sync_parallelize closure (13_immutable_nested_capture) | SIGSEGV 139 | compiles | FIXED upstream; do not file. Upgrade removes the IPCA test-storage exception. |
| DBSCAN tracked mutable field pointer capture "cannot be converted ... to ref" (01_mutable_field_capture) | error | compiles | FIXED upstream; do not file. |
| DeadArgumentElimination surveyUse (dbscan/checks/compiler_repro_dead_arg_elim.mojo) | abort 134 | compiles | Already modular/modular#7164 (closed completed 2026-10-14). |
| OPTICS nested result pointer capture (optics_01) | confusing ref-conversion error | clear rule: "cannot derive an interior origin from a captured container while an interior reference to the container is also captured" | NOT a Modular bug on 1.2.0: our code violates the rule; fix in our code (take the pointer outside the capture / capture one reference only). |
| PythonObject = PythonObject(None) runtime default evaluated at compile time | fails | fails | FILED modular/modular#7543 (18-line repro + 1-line control). |
| Shared DeviceContext concurrent submission on Metal | 2/3 runs crash | 2/3 crash; per-stream select_stream 4/6 crash (num_streams()==1 on Apple) | FILED modular/modular#7544. |
| Compiler deadlock (mutual recursion + default) | seen once | not rechecked | No small repro; not filed. |
| "compile offload result missing kernelId" | seen | not rechecked | No small repro; not filed (AGENTS.md rule already avoids it). |
| 11 older asks in docs/MODULAR_ASKS.md (AMD/NVIDIA/Linux) | - | not rechecked | Need NVIDIA/AMD/Linux to recheck; not filed. |

Recommendation for Andrew: moving the pin to Mojo 1.2.0 / MAX 26.7 removes three blockers (IPCA, DBSCAN, dead-arg)
and turns OPTICS into an ordinary code fix. It is a toolchain change (API moves, e.g. `std.gpu` -> `max.gpu`, closure
changes; bits may change, all columns together per AGENTS.md); Andrew decides.
