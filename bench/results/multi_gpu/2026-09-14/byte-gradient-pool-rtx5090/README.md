# Distributed byte-LM reduction scratch: two RTX 5090s

RunPod `cvisjmryfcmz6g`, two 32 GB RTX 5090 cards, IDENTICAL, sm_120.
Source `2d4f3952a` is applied to the frozen qualified H100 tree in a separate
`/root/gradient-pool` checkout. The frozen replay checkout and binaries remain
unchanged. The base extension is reused from that same-source RTX 5090 build;
the byte-LM production and fault bindings are rebuilt here. No local execution.

The job exits zero. Eight production fixtures cover one- and three-layer
models at 2, 3, 5 and 8 logical shards on two GPUs, compared bit for bit with
complete optimizer replicas and one-GPU ordered replay. They verify all state,
losses, gradients, checkpoint migration, malformed-result rollback, invalid
input atomicity and native allocation counts. The original byte-LM receipt and
the three-shard pooled final-state hashes remain exactly unchanged.

The native ownership reports verify reduction scratch totals eight bytes per
parameter across the group, partitioned with optimizer ownership. With two
GPUs the first GPU holds half the former root-only reduction allocation.
Moments and rollback state retain their previously qualified ownership. Total
model weight and gradient replicas remain; this is not full model capacity.

The separate fault build passes bad summed gradients, refused moments,
nonfinite updated moments and negative updated moments on the second owner,
with both owners restored exactly. Production is restored after the gate.
Source archive, binary hashes, logs, allocation reports and job script are kept.

This change is qualified on RTX 5090s. The preceding frozen-source H100/5090
receipt covers the prior root-scratch implementation; matching its retained
outputs is replay evidence, not a new same-source H100/AMD/Apple qualification
of this changed driver. No throughput or beyond-single-GPU model claim.
The pod remains leased for the next model-memory pooling work.
