# GPT-3-small optimizer and pointwise audit

Base: `1543defbc`. No production source change is recommended from this
local-only audit.

## Already fused or intentionally pinned

- Adam/AdamW is one elementwise launch. Its clean kernel performs the pinned
  O1--O14 sequence and writes parameters and both moments in place. Denominator
  and quotient scratch are one cell unless an explicit intermediate-recording
  build is requested.
- Byte-LM refuses gradient clipping, SGD, and mixed-precision optimizer paths;
  the general optimizer's clip reduction tree and scalar readback therefore do
  not occur in the GPT-3-small production step.
- Biased GELU is already fused (`92f06eef3`). Backward scratch initialization
  and intermediate RMSNorm waits were already consolidated (`030328401`). A
  residual-plus-RMSNorm fusion was previously screened and rejected because it
  could not preserve the pinned reduction behavior.

## Rollback traffic

At 162,147,840 parameters, the pooled optimizer owns three rollback shadows.
Across all owners, a successful step copies exactly 1,945,774,080 bytes of
old parameter/moment payload into them. Counting source reads and destination
writes, that is 3,891,548,160 bytes of device memory traffic. It is required
because a failure after an in-place update must restore every owner before the
group is usable again.

A plausible scheduling-only candidate is to enqueue each owner's three shadow
copies without the final per-owner wait, then enter the existing owner update
loop. The first-moment refusal scan at the start of `pool_update` drains that
same context before the update, so snapshot completion and error precedence
remain intact. This would overlap snapshot copies across physical devices and
would not alter arithmetic or storage.

It is not committed: on one device there is no independent context to overlap,
and current NVIDIA/AMD capacity is owned by other qualification lanes. Thus no
production-shaped timing establishes a benefit. Fusing snapshot and AdamW in
one kernel was also rejected: although it could remove three old-state reads,
it duplicates the optimizer's highly audited O1--O14 implementation and
creates a second arithmetic spelling for a small whole-step upper bound.

The remaining validation scans are not redundant. Before snapshot they pin
the public refusal order for parameters, first moments, second moments, and
negative second moments. After update they prevent a partially invalid group
from being published. Removing or combining them would change error precedence
or the fixed scan reduction and was not attempted.
