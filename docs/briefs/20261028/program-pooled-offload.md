# Lane: program-pooled-offload (Program pooled/offloaded training import-time admission; distributed draft)
Read COMMON.md. Worktrees: ~/mojolearn-wt/program-distributed-boundary-20261027 (lane/program-distributed-boundary-20261027 @ 654531acce
green, being integrated) and ~/mojolearn-wt/distributed-public-boundary-20261027 (6 dirty, preserved earlier child draft: do NOT reset it).
Read experiments/owned_program_boundary_20261028/, experiments/program_public_boundary_20261027/HANDOFF.txt and the root handoff
"Local /root/program_dispatch_finish" section. Evidence: ~/mojolearn-evidence/program-pooled-offload-20261028/.
Goal: (1) assess the 6-file draft against merged 654531acce: if it is superseded, record that in the evidence dir and leave the files
(do not delete); if it carries unmerged value, commit it onto a child branch lane/distributed-draft-20261028 and say what it would need.
(2) New scope, in a NEW branch lane/program-pooled-offload-20261028 from origin/main (new worktree ~/mojolearn-wt/program-pooled-offload-20261028,
lean if disk is tight): bindings/parallel_byte_native.mojo, parallel_training/model_pool_training/offload_training import-time admission,
and the NEW private native ByteLM schedule descriptor/factory in _mojolearn_byte_lm.mojo, so public methods delegate to one native call
with no Python compute. Peer3 owns existing ByteLM generation/catalog/codec exports: do not duplicate codec edits. Compile the
HostExec/Program root (receipt pattern in experiments/owned_program_boundary_20261028; ~250 s on 10 workers, rent a CPU box if the Mac is
busy), push green.
