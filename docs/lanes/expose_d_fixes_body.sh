# Workstream D fixes on a GPU box, the WHOLE leg: the successor of
# docs/lanes/handoff_subagents_2026-09-14/expose_d_body.sh, with
# bindings/build_training.sh in the build list, the base binding built last,
# an HDBSCAN crash control and bisect, and every check bounded.
#
# IT DOES NOT FIT A 60-MINUTE LEASE in the worst case (the bounds sum to more
# than an hour), and both runners cap a lease at 60 minutes and refuse more.
# On Hot Aisle and RunPod launch the two halves as two legs instead, each as
# its own extra body: docs/lanes/expose_d_fixes_body_a.sh (control, builds,
# bisect, seven surface tests, check-hdbscan, check-resample) and
# docs/lanes/expose_d_fixes_body_b.sh (check-mixture, check-kernel-methods,
# check-cholesky). This file runs the two in-tree halves in order, for a box
# with no lease cap.
set -u
cd /root/mojolearn 2>/dev/null || cd "$(pwd)"
sh docs/lanes/expose_d_fixes_body_a.sh
sh docs/lanes/expose_d_fixes_body_b.sh
exit 0
