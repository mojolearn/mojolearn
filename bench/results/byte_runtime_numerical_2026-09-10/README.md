# Actual native runtime shape numerical checks

Root command, under nice19/shared build lock and OMP/OPENBLAS threads2:
`MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python
/tmp/mojolearn-root-validation-20260906/bin/python tools/byte_lm_runtime_numerical_check.py
--run --native-vendor metal --oracle-device cpu --out <fresh directory>`.
Exit0; verdict passes both shapes, four native train steps and four evaluations.
Independent FP64 autograd and actual-gradient AdamW use preset tolerances;
negative gradient/nonlinearity controls must fail. Full native/reference arrays,
state witnesses, runtime binding hashes and checks are retained. This tests
small synthetic configurations, not training quality, old/new binary trajectory,
large runtime shape admission or cross-vendor identity. Torch was installed in
the pre-existing temporary validation environment; install log retained.

Host regression command uses the same Python/env: pytest byte surface/runtime
shape/Samba wiring plus tools byte capture portability/state compare tests.
85 tests and11 subtests pass. The oracle's header was annotated after execution;
run metadata's source hashes identify its pre-annotation source correctly.
No native arithmetic changed during this continuation.
