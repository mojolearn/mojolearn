# Transformer backward recorded-stage alias — rejected

An exact-memory prototype made `d_gate_proj_out` a view of the allocation
holding `d_silu_out`.  The SiLU derivative is elementwise and loads each
`d_silu_out[i]` before writing the corresponding gate gradient, so the runtime
dataflow itself is in-place safe.  At GPT-3-small B1/L2048/intermediate3072 it
would remove 6,291,456 floats (25,165,824 bytes) per backward-stage owner with
no new arithmetic or launch.

The full IDENTICAL transformer backward gate rejected it.  All downstream
stages remained exact, but recorded stage 4 (`bwd.d_silu_out`) moved in every
fixture because `IdentityTrace` retains the device view and consumes it after
the later overwrite.  Representative failures were 256/256 moved cells for
`base_b1_l4_nrep1`, 512/512 for `base_b2_l4_nrep2`, and 3,072/3,072 for
`base_b3_l16_nrep2`; the gate exited nonzero at clause (a).

The source prototype was fully reverted.  Do not retry this alias without
either proving tracing is disabled for the allocation's entire lifetime or
changing trace ownership to take an immediate snapshot; either is a separate
contract/API change.  No cloud resource was used.
