# Pooled optimizer buffers on the Apple column only (PR #38, lane/neural-pass34 bdda533d1), 2026-10-01

Follow-up to PR #31: NVIDIA and AMD return to the pre-#31 per-step transport by construction (MOJOLEARN_OPT_POOL=1
forces the pool). L40S, optimizer_resident_check PASS (32 steps, every byte equal) for adam / sgd / adamw on main and
the branch; resident step medians main -> branch: adam 22.4 -> 21.6, sgd 21.7 -> 35.2, adamw 21.2 -> 22.0 ms (this pod
swings 10-60% run to run). Board cells sgd / adam / adamw: released 0.8.32 196.6 / 183.3 / 185.4, branch 184.5 /
203.6 / 196.6, branch with POOL=1 183.5 / 199.3 / 182.3 ms; digests identical in every run. Apple keeps #31's path.
