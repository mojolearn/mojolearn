# Lane: peer3-neural-tokenizer (finish Transformer/Mamba/tokenizer/corpus/catalog/generation native boundary)
Read COMMON.md. Worktree /Users/andrewhendel/CascadeProjects/mojolearn-peer3-20261027, branch lane/peer3-neural-tokenizer-20261027
(head 360e0fa81a, 2 dirty files, 28 commits not in main). Evidence: ~/mojolearn-evidence/peer3-neural-tokenizer-20261028/.
Start by reading experiments/peer3_neural_tokenizer_20261027/COMPACTION_HANDOFF_20261028.txt and the other handoffs in that
dir, ~/mojolearn-evidence/peer3-corpus-tape-moe-20261027/compaction-handoff-push.log (grep, 25 no_compute_in_shells refusals
over metadata-loop comment formats), and the root handoff's Peer3 section + addendum (codec root passed at ed35e27cd9; neural LLVM
compile failed only for ENOSPC; constructor 60594f0c11 / tokenizer 7d435efeba source complete; parallel snapshot reread correction
and the final coherent compile still needed; constructor post-GIL field publication and canonical UTF8 handler identity open).
Goal: a coherent, compiled, pushed head for the whole lane: finish the open items, fix the guard-comment formats so the push hook
accepts (tag host loops `# driver-loop: over <what>` per docs/plans/GPU_PATH_HOST_LOOPS.md), merge current origin/main into the lane
(preserve Peer1/Peer4/Program shared-file edits; no native_codec duplication), then compile the public/parallel/codec/tokenizer roots
named in the handoff (rent a RunPod CPU box for the neural closure; it is large) and push. If part cannot be made green, split it onto a
clearly named child branch and deliver the green remainder. Ownership limits: do not edit Program lane's ByteLM schedule
descriptor/factory or Peer1's kernel files beyond conflict resolution.
