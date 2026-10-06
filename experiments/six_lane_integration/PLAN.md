# Six-lane source integration

Scope: the seven source commits in inputs.json, six logical lanes. Neural r3 and v2 require explicit reconciliation. Historical neural v1 and all other worktrees are excluded.

Authorization: source/harness integration and supported local compile-only work. Owner steering additionally authorizes renting a fast NVIDIA RunPod worker for compilation only. No estimators, runtime tests, performance, identity, quality, AMD remote jobs, promotions, or merge into main.

1. Freeze source heads, preserve dirty-source gaps, inventory ideas, aliases, sub-arms, interactions and production reach.
2. Consolidate on latest committed main, retaining incumbent fixes/defaults and benchmark contracts. Resolve shared code semantically. Keep new switches off.
3. Build a master discovery/planning/build/evidence interface around retained workload runners; freeze benchmark source and configuration; make missing coverage explicit.
4. Program the future matrix, including candidate A against incumbent B, sub-arms, justified interactions and combined configurations. No execution in this task.
5. Freeze integrated source; compile distinct locally supported binding/mode/define configurations serially. Fix real compiler errors, retain failed logs and exact artifact provenance. Never import produced modules.
6. Deliver catalogs, ledger, benchmark snapshot, future matrix, build coverage and later-phase instructions.

Every new integrated source entry starts uncompiled, untested, unmeasured, quality unverified and identity unverified. Source reach and compile coverage are not runtime reach.

Logs: complete output stays under the external evidence directory recorded in inputs.json. Inspect exit statuses and bounded diagnostics; never infer full success from filtered logs.
