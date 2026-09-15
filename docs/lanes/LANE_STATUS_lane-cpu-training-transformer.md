# Lane status: lane/cpu-training-transformer (2026-09-15)

## Done
- CPU host path for TransformerBlock (the `transformer` and `transformer-window` lanes): new host family `transformer`
  (`bindings/_mojolearn_transformer_host.mojo`, `bindings/build_transformer_host.sh`,
  `transformer/host/transformer_block_host.mojo` over the existing transformer host oracles), declared in
  `python/mojolearn/host_surface.py`; no-CPU-path sentence now says "the Mamba and Samba blocks"; spans regenerated;
  gate workflow triggers added; new test `python/mojolearn/tests/test_cpu_training_transformer.py`. Commit 82ffcb388.
- M4, one core: both lanes, 9 fixtures, IDENTICAL=18 train / 18 infer / 18 batch x4 against the 166-lane record;
  sabotage build DIVERGENT=18 on all three. Evidence committed at 6839b4b12 in
  `bench/results/identity_break/2026-09-15_cpu-transformer/`; scratch copies in `~/mojolearn-evidence/cpu-training-transformer/`.
- No rented boxes, nothing running remotely.

## Gate result so far
- CPU identity gate run 34969598898 at 6839b4b12: NOT green. On all seven runners the covered-lanes step passed with
  `require-columns 4 ... : OK` and all 18 transformer and transformer-window train cells IDENTICAL x4. Apple M1, x86 draw a
  and draw d finished green (sabotage caught). ARM64 and x86 draws b, c, e were cancelled at the job's 60-minute limit
  inside the sabotage step, the same cancellation main's own runs 34968255704 and 34956243867 hit. The fix is
  lane/cpu-training-gate-budget (sharded identity_break), gating as run 34971932337.
- Not merged to main. Plan: once gate-budget is on origin/main, merge origin/main into this branch, push (the gate
  queues), and merge to main when that run is green.

## Next commands (from a worktree on this branch)
```
gh run view 34969598898 --json jobs -q '.jobs[] | .name+" "+.conclusion'
git fetch origin && git merge origin/main      # keep both sides of host_surface.py lists
python3 tools/docs_facts.py --write && python3 tools/docs_facts.py --check
python3 packaging/wheel_ci.py pins . && python3 packaging/wheel_ci.py inventory python/mojolearn
cd python && /Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/test/bin/python -m pytest -q mojolearn/tests/test_host_surface.py
git push origin HEAD:lane/cpu-training-transformer && git push origin HEAD:main
```
Local rebuild/rerun, one core: `sh bindings/build_transformer_host.sh` with `MOJOLEARN_HOST_OUTDIR=<fresh dir>`, then
`MOJOLEARN_HOST_DIR=<dir> PYTHONPATH=python python3 tools/identity_break.py --lanes transformer,transformer-window --json out.json`
and `--diff $(python3 python/mojolearn/host_surface.py --training-gpu-columns) out.json --require-columns 4 --lanes transformer,transformer-window`.
