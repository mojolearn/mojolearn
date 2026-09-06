# Corrective backward attempt: Mamba3 regression compile failure

AMD MI325X, source `6cbebbbdd67c3b27e8c113a09d7a41ec41e3e48a`.
The new regression used an unsupported variadic List constructor. Both Mamba3
cases failed compilation before producing native outputs; this is
`INFRA_FAILURE`, not evidence about the chain-rule fix. Other completed
backward rows and the full compiler diagnostics are retained here.

The regression was corrected to typed list literals at source `f6e55193`.
A reuse attempt did not execute: deletion was already requested before the
controller pause. The controller resumed and droplet 598113981 was confirmed
absent by HTTP 404. The next run uses a fresh guarded machine.
