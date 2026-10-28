# Lane: triage (classify remaining ahead-of-main branches)  -- read-only on git; writes only to the evidence dir

Read ~/mojolearn-evidence/briefs-20261028/COMMON.md first. Repo: /Users/andrewhendel/CascadeProjects/mojolearn.
Evidence/output: ~/mojolearn-evidence/triage-20261028/REPORT.md (one section per branch) + a one-line-per-branch TSV.

For each branch below, determine: (a) commits not in main by patch id (`git cherry main <sha>`), (b) files changed vs
main, (c) whether the content is already in main or in one of the GREEN heads being integrated now
(aefc4545aa 0456ee3e6a 0f0b336d87 a3a8150111 74c1e58ebe 4b9999bee4 654531acce 7cacec39ba 8e67e1bce1 c7694b3277
9a0f043190 36a66a6885 2f3e60c2ca 77ed135648) -- compare `git diff main...<sha> -- <file>` against those heads,
(d) whether a compile receipt exists for its exact source (search experiments/*/ and ~/mojolearn-evidence/ for the sha
and the branch name; read the lane's HANDOFF/CURRENT/CHECKPOINT files, never transcripts), (e) verdict:
SUPERSEDED (content already landed elsewhere; safe to leave), READY (green receipt for this exact source, should be
queued for main; say what it depends on), NEEDS-COMPILE (coherent source, no receipt; name the likely entry point),
DRAFT (incomplete; say what is missing), or STALE (abandoned/duplicate; say why).

Branches (worktree path -> branch@head):
 ~/mojolearn-wt/hdbscan-public-lifecycle-20261027 -> lane/native-label-protocol-20261028 @ b395d0f541 (16 unmerged commits, 40 files)
 ~/mojolearn-wt/peer1-radius-native-20261027 -> lane/peer1-radius-native-20261027 @ 3c33231bdd (10; likely superseded by aefc4545aa)
 ~/mojolearn-wt/peer2-decomposition-20261027 -> @ 3bd993dec5 (10)
 ~/mojolearn-wt/peer2-manifold-20261027 -> @ 8c25f09b2c (4, 3 dirty)
 ~/mojolearn-wt/peer2-preprocessing-decomposition-manifold-20261027 -> @ 9d89fb64ad (41 unmerged, 118 files; which are batch 8 / prep hold / other)
 /Users/andrewhendel/CascadeProjects/mojolearn-gp-lifecycle -> lane/peer5-gp-lifecycle-20261027 @ 3ae8472e55 (5)
 /Users/andrewhendel/CascadeProjects/mojolearn-gp-kernels -> lane/peer5-gp-kernels-20261027 @ c9389f3f5b (2)
 ~/mojolearn-wt/native-dispatch-20261027 -> lane/classical-shared-provider-20261027 @ b12b022062 (2 commits, 96 files!)
 ~/mojolearn-wt/native-neural-integration-20261027 -> integration/native-neural-boundary-20261027 @ b9d325d034 (2)
 ~/mojolearn-wt/gbdt-cpu-verifier-integration-20261028 -> @ 33fa14748f (2) and ~/mojolearn-wt/gbdt-cpu-ranking-20261027 @ b16c6374a0 (1): handoff says ranking is in main as cherry-picks; confirm nothing is missing
 ~/mojolearn-wt/peer1-hierarchy-cluster-20261028 @ f51678fa09 (1); ~/mojolearn-wt/peer1-hierarchy-audit-20261027 @ bb8a62c3a7 (1)
 ~/mojolearn-wt/peer1-native-next-20261027: the 7 commits aefc4545aa..4ad607eff5 (descriptor continuation) + 3 dirty files
 ~/mojolearn-wt/classical-frontdoor-admission-20261027 @ 120816632d (1); ~/mojolearn-wt/provider-boundary-audit-20261027 @ 2318bc53af (1)
 /Users/andrewhendel/CascadeProjects/mojolearn-peer4-embedding @ 2ee7273ffe (1); mojolearn-peer4-cnn-conv (2 dirty), mojolearn-peer4-cnn-graph (2 dirty)
 /Users/andrewhendel/CascadeProjects/mojolearn-peer5-iforest -> lane/peer5-iforest-lifecycle-20261028 @ 115b5eb8bb (1)
 ~/mojolearn-wt/peer6-metrics-20261027 tip de7fea9506 (1 beyond green 74c1e58ebe, 5 dirty)
 /Users/andrewhendel/CascadeProjects/mojolearn-peer3-compile-20261027 (detached 788b45d6f3) vs mojolearn-peer3-20261027 @ 360e0fa81a: what differs
 ~/mojolearn-wt/distributed-public-boundary-20261027 (6 dirty, 0 ahead): what the draft is, vs merged 654531acce
 ~/mojolearn-wt/apple-release-0857, release0856-native (detached, dirty): what the dirt is; are they release leftovers
Also: run `python3 tools/public_runtime_inventory.py --summary` (read its usage first; it is a source scan, no product import)
on main and report the prohibited_source count and the 24 flagged files with their owners from
experiments/provider_boundary_audit_20261028/COVERAGE_OWNERS.tsv, so the orchestrator can see which Python-in-runtime
files have no active lane. Do not modify any worktree. Final reply: the TSV plus the top 5 recommendations.
