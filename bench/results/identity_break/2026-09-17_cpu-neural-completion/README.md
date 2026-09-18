# CPU neural and low-bit controls, 2026-09-17

Source `dcddb8345`, RunPod `4018mzhxmp3prb`, two vCPUs and at most two
compiler workers. Every lane ran all nine fixtures twice with step/full and
sampler/replay where declared. The pod is VERIFIED DELETED (HTTP 404 and absent
from listing). Cost $0.0433; 362 seconds building, 2126 seconds executing.

Six lanes pass complete clean checks and all-nine native training controls:
IVF, both low-bit GEMMs, both byte-LM lanes, and the corrected H/C/V metric.
Five add CPU public references; the metric was already restored separately.

The original runner checked training controls alone. Its `summary.json` and
per-lane `negative-controls.json` are retained unchanged. The authoritative
follow-up `reevaluated-controls.json` additionally rejects failed clean
properties: eighteen neural/low-bit lanes lacked the neural host binding,
causing inference and property refusals. They are not full successful records.
Ordered gradient sum's original fault was inert (0/9); a new native fault and
all-nine control are recorded separately. The other 24 training controls each
move all nine fixtures. These are combined-family sabotage sets, not isolated
proof of every native operation.

Mamba2's complete Mac record fills its clean property gap separately. Concurrent
reference work is filling the other ordinary neural lanes. The next scoped job
includes the neural dependency for the twelve low-bit lanes and CPU controls
for the twelve candidates still awaiting additional GPU vendor records.
