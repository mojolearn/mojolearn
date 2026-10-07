# SPDX-License-Identifier: Apache-2.0
"""Source-only NI19--NI36 experiments, 2026-10-06; all new arms default OFF.

These are scheduling/storage candidates with the existing arithmetic graph,
except NI20's attention, NI34's chunked head and NI35's CE token-total graph.
No build, identity, quality or performance evidence has been collected for
this revision. Full-workload NVIDIA/AMD A/B and host/Apple identity remain
required before any promotion. MOJOLEARN_IDN_ALL_OFF suppresses every arm.
"""
from std.sys.compile import is_defined, get_defined_int
from checks.numerics import GLOBAL_NUMERIC_MODE, NUMERIC_IDENTICAL

comptime _ENABLED = (
    GLOBAL_NUMERIC_MODE == NUMERIC_IDENTICAL
    and not is_defined["MOJOLEARN_IDN_ALL_OFF"]()
)

# NI23: rotate K while writing the packed attention layout. S; same products
# and add, with all saved stages retained. Full prefill only by cache semantics.
# Arm 2 of the ONE RoPE switch MOJOLEARN_IDN_ROPE (arm 1 is NN27).
comptime IDN_ROPE_CACHE = _ENABLED and get_defined_int["MOJOLEARN_IDN_ROPE", 0]() == 2
# NI24 (+ NN28, merged 2026-10-07; NN28 was the same capability on fewer
# callers): only owned training-prefill callers may omit the persistent cache
# copy. One switch for byte_lm, its layer pool and modeling_llama.
comptime IDN_TRAIN_NO_DECODE_CACHE = _ENABLED and is_defined["MOJOLEARN_IDN_TRAIN_NO_DECODE_CACHE"]()
# NI25 S sub-arm: serial original statistics, cooperative output scaling.
# Arm 3 of the ONE norm switch MOJOLEARN_IDN_NORM (norm_profile_contract.mojo).
comptime IDN_RMS_ROW_BLOCK = _ENABLED and get_defined_int["MOJOLEARN_IDN_NORM", 0]() == 3
# NI26 (+ NN26, merged 2026-10-07; both enabled the same training_swiglu
# launch): preserve both silu_out and gated for tracing and backward.
comptime IDN_TRAIN_SWIGLU = _ENABLED and is_defined["MOJOLEARN_IDN_TRAIN_SWIGLU"]()
# NI27: views are rebound to the current owned arena after optimizer swaps.
# L11 (2026-10-07): NN60 and NI27 are arms of ONE switch,
# -D MOJOLEARN_IDN_LM_VIEWS=0|1|2|3: 1 = block weight views (NN60 block),
# 2 = embedding/head views (NN60 emb/head), 3 = all views incl. the layer pool
# (NI27). Aliases only; no arithmetic changes.
comptime IDN_LM_VIEWS_ARM = get_defined_int["MOJOLEARN_IDN_LM_VIEWS", 0]()
comptime IDN_LM_PARAM_VIEWS = _ENABLED and IDN_LM_VIEWS_ARM == 3
# NI32: reuse validation only inside one owned, immutable byte-training call.
# L11 (2026-10-07): NN51 and NI32 are arms of ONE switch,
# -D MOJOLEARN_IDN_LM_RESIDENT_TOKENS=0|1|2: 1 = NI32 (the forward's ID check
# covers the embedding gather/backward), 2 = NN51 (arm 1 plus prerefused
# targets in the CE forward).
comptime IDN_LM_RESIDENT_TOKENS_ARM = get_defined_int["MOJOLEARN_IDN_LM_RESIDENT_TOKENS", 0]()
comptime IDN_LM_OWNED_TOKENS = _ENABLED and IDN_LM_RESIDENT_TOKENS_ARM >= 1
# NI33: keep the original per-cell divides and the same ignored-row stores.
# NN52 (retired MOJOLEARN_NN52_CE_WEIGHT_GRAD) launched the same fused kernel
# in identical_ce_backward_into; one switch (L11, 2026-10-07).
comptime IDN_CE_GRAD_FUSED = _ENABLED and is_defined["MOJOLEARN_IDN_CE_GRAD_FUSED"]()
# R6 (lane/neural-ce-denom, 2026-10-07): the CE softmax denominator (loss L4,
# and L9 under smoothing) computed by one block per row that walks the GEMM
# contract's own leaf chains and fold tree (training/checks/loss.mojo
# ce_denom_rowfold_kernel) instead of the routed n=1 ones-GEMV. Execution plan
# only: same bits on every column. Refused together with the neural GEMM
# profile (MOJOLEARN_IDN_GEMM_LEAF=1|2, MOJOLEARN_IDN_NEURAL_CHAINS), whose
# chain spelling the row fold does not reproduce.
comptime IDN_CE_DENOM_ROWFOLD = _ENABLED and is_defined["MOJOLEARN_IDN_CE_DENOM_ROWFOLD"]()
# NI36 narrow sub-arm: dA and dW GEMMs run serially on the same queue, so
# their disjoint scratch lifetimes can share one allocation. No tape alias.
# L11 (2026-10-07): NI36 (now only this scratch sharing; its tape half is
# NI48 below) and NN62 are arms of ONE switch,
# -D MOJOLEARN_IDN_TRAIN_SCRATCH=0|1|2|3: 1 = shared dA/dW scratch,
# 2 = NN62 byte-LM lifetime arena (training/neural_ab_lifetime.mojo), 3 = both.
comptime IDN_TRAIN_SCRATCH_ARM = get_defined_int["MOJOLEARN_IDN_TRAIN_SCRATCH", 0]()
comptime IDN_TRAIN_BACKWARD_SCRATCH = _ENABLED and (IDN_TRAIN_SCRATCH_ARM == 1 or IDN_TRAIN_SCRATCH_ARM == 3)
# NI35: explicit V arithmetic profile for the CE token-total fold only.
# Vocabulary folds, objective divisor, dlogits and optimizer are unchanged.
# Arm 2 of MOJOLEARN_IDN_CE_TOKEN_FOLD (training/neural_ab_profile_contract.mojo).
comptime IDN_LOSS_TOKEN_TREE_V2 = _ENABLED and get_defined_int["MOJOLEARN_IDN_CE_TOKEN_FOLD", 0]() == 2
# NI34: the existing explicit config stays supported; this opt-in chooses it
# for default-constructed byte configs and enables compatible Samba callers.
comptime IDN_CHUNKED_LM_HEAD_V2 = _ENABLED and is_defined["MOJOLEARN_IDN_CHUNKED_LM_HEAD_V2"]()
# NI36/NI48: immutable native forward tapes, consumed exactly once.
# Arm 3 of MOJOLEARN_IDN_ACT_RETAIN (transformer/experiments/checkpoint_contract.mojo).
comptime IDN_SAMBA_FORWARD_TAPE = _ENABLED and get_defined_int["MOJOLEARN_IDN_ACT_RETAIN", 0]() == 3

# NI20: fixed tile32 online attention numerical graph on every column. Arm 2
# of the ONE softmax switch MOJOLEARN_IDN_ATTN_SOFTMAX (arm 1 is NN20,
# transformer/experiments/attention_summary_contract.mojo).
comptime IDN_ATTENTION_V2 = _ENABLED and get_defined_int["MOJOLEARN_IDN_ATTN_SOFTMAX", 0]() == 2
# S1 (lane/samba-resident, 2026-10-07): the Samba stack's forward and train
# step as ONE device-resident binding call each (training/samba_resident.mojo):
# the registry, the gradient, every block's activations and the backward
# stages stay on the device; the per-op route (python/mojolearn/_samba_impl.py
# driving training/samba_ops.mojo and the block bindings layer by layer, a
# PCIe round trip and a device drain at every op) is arm B. Same kernels on
# the same operands in the same order: no bit change on any column.
comptime IDN_SAMBA_RESIDENT_STEP = _ENABLED and is_defined["MOJOLEARN_IDN_SAMBA_RESIDENT_STEP"]()
