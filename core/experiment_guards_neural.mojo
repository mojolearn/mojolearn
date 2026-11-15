# SPDX-License-Identifier: Apache-2.0
"""Opt-in strategy admission for the neural bindings (training, transformer, Mamba, embedding, byte LM, sequence, CNN, tokenizer).

Each binding of this family imports NEURAL_GUARDS_OK and asserts it in its build function, so every compile of one of
those bindings (per-binding or merged) evaluates these guards. A guard whose define is read in code by bindings
of more than one family belongs in core/six_lane_experiment_guards.mojo, which every binding reaches through
checks/numerics.mojo.
"""
from std.sys.compile import is_defined, get_defined_int
# The entries below moved verbatim, in their original order, from core/six_lane_experiment_guards.mojo on
# 2026-11-04 (lane split-hot-groups; the placement of each entry is docs/build/guard_split.json). A header comment
# that headed lines of several families is carried with each of them.


def _check_configuration() -> Bool:
    comptime assert not (is_defined["MOJOLEARN_NN14_BOUNDED_IM2COL"]() and is_defined["MOJOLEARN_NI12_IMPLICIT_CONV"]()), "incompatible integrated strategies: MOJOLEARN_NN14_BOUNDED_IM2COL / MOJOLEARN_NI12_IMPLICIT_CONV"
    comptime assert not (is_defined["MOJOLEARN_IDN_NEURAL_NN12"]() and is_defined["MOJOLEARN_NI01_TRAINING_WORKSPACE"]()), "incompatible integrated strategies: MOJOLEARN_IDN_NEURAL_NN12 / MOJOLEARN_NI01_TRAINING_WORKSPACE"
    comptime assert not (is_defined["MOJOLEARN_NN48_CSR_TILES"]() and is_defined["MOJOLEARN_NI55_GRAPH_FEATURE4"]()), "incompatible integrated strategies: MOJOLEARN_NN48_CSR_TILES / MOJOLEARN_NI55_GRAPH_FEATURE4"
    comptime assert not (is_defined["MOJOLEARN_NI59_DROPOUT_CHANNEL"]() and is_defined["MOJOLEARN_NI60_DROPOUT_APPLY4"]()), "incompatible integrated strategies: MOJOLEARN_NI59_DROPOUT_CHANNEL / MOJOLEARN_NI60_DROPOUT_APPLY4"
    comptime assert not (is_defined["MOJOLEARN_NN53_HEAD_CHUNK512"]() and is_defined["MOJOLEARN_IDN_CHUNKED_LM_HEAD_V2"]()), "incompatible integrated strategies: MOJOLEARN_NN53_HEAD_CHUNK512 / MOJOLEARN_IDN_CHUNKED_LM_HEAD_V2"
    comptime assert not (is_defined["MOJOLEARN_NN53_HEAD_CHUNK512"]() and is_defined["MOJOLEARN_NN53_HEAD_CHUNK2048"]()), "incompatible integrated strategies: MOJOLEARN_NN53_HEAD_CHUNK512 / MOJOLEARN_NN53_HEAD_CHUNK2048"
    comptime assert not (is_defined["MOJOLEARN_AFN26_MAMBA1_CHUNKS16"]() and is_defined["MOJOLEARN_AFN26_MAMBA1_CHUNKS64"]()), "incompatible integrated strategies: MOJOLEARN_AFN26_MAMBA1_CHUNKS16 / MOJOLEARN_AFN26_MAMBA1_CHUNKS64"
    comptime assert not (is_defined["MOJOLEARN_AFN26_MAMBA3_THREADS64"]() and is_defined["MOJOLEARN_AFN26_MAMBA3_THREADS256"]()), "incompatible integrated strategies: MOJOLEARN_AFN26_MAMBA3_THREADS64 / MOJOLEARN_AFN26_MAMBA3_THREADS256"
    comptime assert not (is_defined["MOJOLEARN_AFN26_EMB_THREADS64"]() and is_defined["MOJOLEARN_AFN26_EMB_THREADS128"]()), "incompatible integrated strategies: MOJOLEARN_AFN26_EMB_THREADS64 / MOJOLEARN_AFN26_EMB_THREADS128"
    comptime assert not (is_defined["MOJOLEARN_AFN26_ATTN_NORM_TPB128"]() and is_defined["MOJOLEARN_AFN26_ATTN_NORM_TPB512"]()), "incompatible integrated strategies: MOJOLEARN_AFN26_ATTN_NORM_TPB128 / MOJOLEARN_AFN26_ATTN_NORM_TPB512"
    # Deleted 2026-10-18 (feature/native-autograd-20261018): noise. Recoverable at 91ebd1cb4 (never on main).
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_MLP_SEED_SCALE"](), "removed: MOJOLEARN_AUTOGRAD_MLP_SEED_SCALE (1/B folded into the MLP backward seed) was noise: NV 0.388 -> 0.387 s, AMD 0.602 -> 0.599 s on HIGGS 1M (256,) b512, same digest; deleted 2026-10-18, recoverable at feature/native-autograd-20261018@91ebd1cb4"
    # Deleted 2026-10-26 (lane dead-native-entries-0859): dead native entries (docs/TOMBSTONES.md). SCHED_FAST_TABLE only
    # registered sched_exp_block, which nothing called after the schedules went native; PURITY_3_OFF / PURITY_6_OFF only
    # changed the x_neighbors_purity_flags word, which nothing read. Recoverable at 485ba9cc1.
    comptime assert not (is_defined["MOJOLEARN_SCHED_FAST_TABLE"]() or is_defined["MOJOLEARN_SCHED_FAST_TABLE_OFF"]()), "removed: MOJOLEARN_SCHED_FAST_TABLE (sched_exp_block) had no caller; the schedules run lr_schedule_at / lr_schedule_series"
    # Restored 2026-11-09 (lane r1-restore-measured-winners): MOJOLEARN_TRANSFORMER_FRESH_PREFILL /
    # MOJOLEARN_TRANSFORMER_LEGACY_FRESH_PREFILL (the GPU binding's per-call transformer_forward_fresh export)
    # are live again, with their pre-x4 meaning; lane x4's refusal (deleted at 256e5e0c16) is lifted.
    # Deleted 2026-11-09 (lane x7-byte-lm-on-executor): the byte LM trainer's hand step, whose arms these were
    # (the step is the trainer's captured program on Program.fit now). Never measured as defaults; docs/TOMBSTONES.md
    # "BYTE_LM_HAND_STEP". Recoverable at the deletion commit's parent on lane/x7-byte-lm-on-executor.
    comptime assert not (is_defined["MOJOLEARN_IDN_TRAIN_ONE_WAIT"]() or is_defined["MOJOLEARN_AFN_LM_NOSYNC"]() or is_defined["MOJOLEARN_AFN26_LM_NOSYNC"]() or is_defined["MOJOLEARN_AFN_LM_HEAD_FUSE"]() or is_defined["MOJOLEARN_AFN26_LM_HEAD_FUSE"]()), "removed: MOJOLEARN_IDN_TRAIN_ONE_WAIT / MOJOLEARN_AFN_LM_NOSYNC / MOJOLEARN_AFN_LM_HEAD_FUSE (and the AFN26 spellings): arms of the byte LM's hand step, deleted 2026-11-09 with it (lane x7-byte-lm-on-executor: the trainer runs one captured program on Program.fit); see docs/TOMBSTONES.md BYTE_LM_HAND_STEP"
    # Deleted 2026-11-09 (lane x7b-step-ops-host-arm): the hand train step's gradient views (docs/TOMBSTONES.md
    # "BYTE_LM_X7_LEFTOVERS"). Recoverable at the deletion commit's parent on lane/x7b-step-ops-host-arm.
    comptime assert not is_defined["MOJOLEARN_IDN_PARAM_GRAD_VIEWS"](), "removed 2026-11-09 (lane x7b-step-ops-host-arm): MOJOLEARN_IDN_PARAM_GRAD_VIEWS (every parameter and gradient of the hand byte LM train step a view of the flat owners) went with that step (lane x7: the step is the trainer's captured program); default off, never promoted; see docs/TOMBSTONES.md BYTE_LM_X7_LEFTOVERS"
    comptime assert get_defined_int["MOJOLEARN_IDN_NEURAL_STAGE_DEPTH",2]() == 1 or get_defined_int["MOJOLEARN_IDN_NEURAL_STAGE_DEPTH",2]() == 2, "invalid neural STAGE_DEPTH configuration"
    comptime assert get_defined_int["MOJOLEARN_IDN_NEURAL_STAGE_PAD",0]() == 0 or get_defined_int["MOJOLEARN_IDN_NEURAL_STAGE_PAD",0]() == 1, "invalid neural STAGE_PAD configuration"
    comptime assert get_defined_int["MOJOLEARN_IDN_NEURAL_FILL_BLOCKS",1]() > 0, "positive neural resource budget required"
    comptime assert get_defined_int["MOJOLEARN_IDN_NEURAL_STREAM_GROUP",8]() > 0, "positive neural resource budget required"
    comptime assert get_defined_int["MOJOLEARN_IDN_NEURAL_RETAINED_FLOATS",16777216]() > 0, "positive neural resource budget required"
    # Lane neural-gemm-attn-dedupe (2026-10-07): each merged idea is ONE define
    # with named arms, so its old mutual exclusions hold by construction.
    # Retired defines are refused so a stale catalog arm cannot build as a
    # silent incumbent.
    comptime R = get_defined_int["MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE_ROLES",7]()
    comptime assert R >= 1 and R <= 7, "MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE_ROLES is a mask: 1 projection, 2 head, 4 weight-grad"
    comptime CH = get_defined_int["MOJOLEARN_IDN_NEURAL_CHAINS",1]()
    comptime assert CH == 1 or CH == 2 or CH == 4, "invalid neural CHAINS configuration"
    comptime EPI = get_defined_int["MOJOLEARN_IDN_NEURAL_GEMM_EPILOGUE",0]()
    comptime assert EPI >= 0 and EPI <= 3, "MOJOLEARN_IDN_NEURAL_GEMM_EPILOGUE is a mask: 1 mlp, 2 cnn"
    comptime HS = get_defined_int["MOJOLEARN_IDN_ATTN_HEAD_SHARE",1]()
    comptime assert HS == 1 or HS == 4, "MOJOLEARN_IDN_ATTN_HEAD_SHARE legal set {4} (two-head I06/NI19 deleted as a loser)"
    # TOMBSTONE: MOJOLEARN_IDN_ATTN_SOFTMAX=2 (slower) deleted 2026-10-10 by c1bf9d832; code recoverable at ca25d9321.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ATTN_SOFTMAX-arm2.patch
    # Both arms are gone, so the define itself is refused, with its forward rows-per-block control MOJOLEARN_IDN_ATTN_V2_TQ.
    comptime assert not is_defined["MOJOLEARN_IDN_ATTN_SOFTMAX"](), "removed: MOJOLEARN_IDN_ATTN_SOFTMAX (=2 online_tile32 attention v2) retired 2026-10-10: slower, avg 1.49x lm-train-step (NV 1.43x / AMD 1.56x), 1.01x lm-forward (NV 0.98x / AMD 1.04x), mean_nll SAME (main ca25d9321, A/B nv2 v1229/v1231/v1232, amd a1555/a1556/a1557, default v1232/a1557); arm 1 summary_tree removed 2026-10-08: slower, NV/AMD lm-forward 4.86x/16.79x, lm-train-step 42.89x/40.18x, samba-forward 1.79x/4.79x, samba-train-step 1.64x/1.81x, transformer-forward 5.48x/21.89x (grid ge123e6f9); see EXPERIMENTS.md; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_IDN_ATTN_V2_TQ"](), "removed: MOJOLEARN_IDN_ATTN_V2_TQ retired 2026-10-10 with the online_tile32 arm it tuned (MOJOLEARN_IDN_ATTN_SOFTMAX=2: slower, avg 1.49x lm-train-step); see docs/TOMBSTONES.md"
    comptime ST = get_defined_int["MOJOLEARN_IDN_ATTN_STASH",0]()
    comptime assert ST >= 0 and ST <= 3, "MOJOLEARN_IDN_ATTN_STASH arms: 1 recompute, 2 packed, 3 alias_y"
    comptime NO = get_defined_int["MOJOLEARN_IDN_NORM",0]()
    comptime assert NO >= 0 and NO <= 4, "MOJOLEARN_IDN_NORM arms: 1 lanes8, 2 split_scale, 3 row_block, 4 lanes8_split_scale"
    comptime RO = get_defined_int["MOJOLEARN_IDN_ROPE",0]()
    comptime assert RO >= 0 and RO <= 2, "MOJOLEARN_IDN_ROPE arms: 1 qk_pair, 2 k_cache"
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_NN01"](), "MOJOLEARN_IDN_NEURAL_NN01 is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_NN02"](), "MOJOLEARN_IDN_NEURAL_NN02 is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_NN03"](), "MOJOLEARN_IDN_NEURAL_NN03 is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_NN04"](), "MOJOLEARN_IDN_NEURAL_NN04 is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_NN08"](), "MOJOLEARN_IDN_NEURAL_NN08 is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_NN09"](), "MOJOLEARN_IDN_NEURAL_NN09 is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_NN10"](), "MOJOLEARN_IDN_NEURAL_NN10 is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_NN11"](), "MOJOLEARN_IDN_NEURAL_NN11 is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_NN15"](), "MOJOLEARN_IDN_NEURAL_NN15 is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_LEAF"](), "MOJOLEARN_IDN_NEURAL_LEAF is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_NN06"](), "MOJOLEARN_IDN_NEURAL_NN06 is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_NI02_GEMM_STREAM_PARTIALS"](), "MOJOLEARN_NI02_GEMM_STREAM_PARTIALS is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_NI08_GEMM_LEAF_256"](), "MOJOLEARN_NI08_GEMM_LEAF_256 is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_NI09_TILED_BIAS"](), "MOJOLEARN_NI09_TILED_BIAS is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_NN17_GQA_FOUR_HEADS"](), "MOJOLEARN_NN17_GQA_FOUR_HEADS is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_IDN_ATTN_GQA_HEAD_REUSE"](), "MOJOLEARN_IDN_ATTN_GQA_HEAD_REUSE is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_NN20_BALANCED_SUMMARY_TREE"](), "MOJOLEARN_NN20_BALANCED_SUMMARY_TREE is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_IDN_ATTENTION_V2"](), "MOJOLEARN_IDN_ATTENTION_V2 is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_NN22_EAGER_DKDV_PAIR"](), "MOJOLEARN_NN22_EAGER_DKDV_PAIR is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_NN23_ROWDOT_DS"](), "MOJOLEARN_NN23_ROWDOT_DS is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_NN24_NORM_LANES8"](), "MOJOLEARN_NN24_NORM_LANES8 is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_NN25_RMS_SPLIT_SCALE"](), "MOJOLEARN_NN25_RMS_SPLIT_SCALE is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_IDN_RMS_ROW_BLOCK"](), "MOJOLEARN_IDN_RMS_ROW_BLOCK is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_NN26_TRAIN_SWIGLU"](), "MOJOLEARN_NN26_TRAIN_SWIGLU is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_NN27_QK_ROPE_PAIR"](), "MOJOLEARN_NN27_QK_ROPE_PAIR is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_IDN_ROPE_CACHE"](), "MOJOLEARN_IDN_ROPE_CACHE is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_NN28_DEAD_TRAINING_CACHE"](), "MOJOLEARN_NN28_DEAD_TRAINING_CACHE is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_ATTN_V1_RECOMPUTE_BACKWARD"](), "MOJOLEARN_ATTN_V1_RECOMPUTE_BACKWARD is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_ATTN_V1_PACKED_ESTASH"](), "MOJOLEARN_ATTN_V1_PACKED_ESTASH is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    comptime assert not is_defined["MOJOLEARN_ATTN_V1_ALIAS_Y_ESTASH"](), "MOJOLEARN_ATTN_V1_ALIAS_Y_ESTASH is retired: see experiments/six_lane_integration/grid_controls/neural-gemm-attn-dedupe.json"
    # L11 (2026-10-07) neural sequence/training dedupe: retired defines refuse
    # (a stale build line must not silently run the B arm), and every merged
    # switch admits only its documented arms. See grid_controls/neural-seq-train-dedupe.json.
    comptime assert not is_defined["MOJOLEARN_NN38_CACHE_SUFFIX_SEEDS"](), "retired define MOJOLEARN_NN38_CACHE_SUFFIX_SEEDS: use IDN_M3_ANGLE_CARRY_CACHE"
    comptime assert not is_defined["MOJOLEARN_NN36_SHARED_DECAY"](), "retired define MOJOLEARN_NN36_SHARED_DECAY: use IDN_M2_YOFF_EXP=1"
    comptime assert not is_defined["MOJOLEARN_IDN_M2_YOFF_EXP_CACHE"](), "retired define MOJOLEARN_IDN_M2_YOFF_EXP_CACHE: use IDN_M2_YOFF_EXP=2"
    comptime assert not is_defined["MOJOLEARN_NN33_YDIAG_ROWS8"](), "retired define MOJOLEARN_NN33_YDIAG_ROWS8: use IDN_M2_YD_ROWS=8"
    comptime assert not is_defined["MOJOLEARN_NN33_CSTATE_P16"](), "retired define MOJOLEARN_NN33_CSTATE_P16: use IDN_M2_CS_PT=16"
    comptime assert not is_defined["MOJOLEARN_NN39_M2_GRAD_TREE"](), "retired define MOJOLEARN_NN39_M2_GRAD_TREE: use IDN_M2_GRAD_FOLD=1"
    comptime assert not is_defined["MOJOLEARN_IDN_M2_GRAD_LEAF128"](), "retired define MOJOLEARN_IDN_M2_GRAD_LEAF128: use IDN_M2_GRAD_FOLD=2"
    comptime assert not is_defined["MOJOLEARN_NN45_CONV_RELU"](), "retired define MOJOLEARN_NN45_CONV_RELU: use IDN_CNN_CONV_RELU=1"
    comptime assert not is_defined["MOJOLEARN_NI16_CONV_RELU_FUSED"](), "retired define MOJOLEARN_NI16_CONV_RELU_FUSED: use IDN_CNN_CONV_RELU=2"
    comptime assert not is_defined["MOJOLEARN_NN43_WGRAD_FIXED128"](), "retired define MOJOLEARN_NN43_WGRAD_FIXED128: use IDN_SEQ_WGRAD=2"
    comptime assert not is_defined["MOJOLEARN_IDN_SEQ_WGRAD_LEAF256"](), "retired define MOJOLEARN_IDN_SEQ_WGRAD_LEAF256: use IDN_SEQ_WGRAD=1"
    comptime assert not is_defined["MOJOLEARN_IDN_SEQ_LN_LEAF32"](), "retired define MOJOLEARN_IDN_SEQ_LN_LEAF32: use IDN_SEQ_LN_LEAF=32"
    comptime assert not is_defined["MOJOLEARN_IDN_SEQ_ROW_SERIAL_SCAN"](), "retired define MOJOLEARN_IDN_SEQ_ROW_SERIAL_SCAN: use deleted (serial per-row scan)"
    comptime assert not is_defined["MOJOLEARN_NN44_STABLE_GROUP"](), "retired define MOJOLEARN_NN44_STABLE_GROUP: use IDN_MOE_STABLE_PACK"
    comptime assert not is_defined["MOJOLEARN_NN52_CE_WEIGHT_GRAD"](), "retired define MOJOLEARN_NN52_CE_WEIGHT_GRAD: use IDN_CE_GRAD_FUSED"
    comptime assert not is_defined["MOJOLEARN_NN54_LOSS_PROFILE"](), "retired define MOJOLEARN_NN54_LOSS_PROFILE: use IDN_CE_TOKEN_FOLD=1"
    comptime assert not is_defined["MOJOLEARN_IDN_LOSS_TOKEN_TREE_V2"](), "retired define MOJOLEARN_IDN_LOSS_TOKEN_TREE_V2: its successor MOJOLEARN_IDN_CE_TOKEN_FOLD=2 was removed 2026-10-10 (noise); see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_NN51_RESIDENT_TOKEN_VALIDATION"](), "retired define MOJOLEARN_NN51_RESIDENT_TOKEN_VALIDATION: use IDN_LM_RESIDENT_TOKENS=2"
    comptime assert not is_defined["MOJOLEARN_IDN_LM_OWNED_TOKENS"](), "retired define MOJOLEARN_IDN_LM_OWNED_TOKENS: use IDN_LM_RESIDENT_TOKENS=1"
    comptime assert not is_defined["MOJOLEARN_NN60_BLOCK_VIEWS"](), "retired define MOJOLEARN_NN60_BLOCK_VIEWS: use IDN_LM_VIEWS=1"
    comptime assert not is_defined["MOJOLEARN_NN60_EMB_HEAD_VIEWS"](), "retired define MOJOLEARN_NN60_EMB_HEAD_VIEWS: use IDN_LM_VIEWS=2"
    comptime assert not is_defined["MOJOLEARN_IDN_LM_PARAM_VIEWS"](), "retired define MOJOLEARN_IDN_LM_PARAM_VIEWS: use IDN_LM_VIEWS=3"
    comptime assert not is_defined["MOJOLEARN_NN56_GROUPED_ADAM"](), "retired define MOJOLEARN_NN56_GROUPED_ADAM: its successor MOJOLEARN_IDN_LM_GROUPED_ADAM was removed 2026-10-11 (avg 1.03x slower)"
    # TOMBSTONE: MOJOLEARN_IDN_M3_S17_TAIL_DBUF (LOSER avg 1.33x samba-train-step, NV 0.935 / AMD 1.726) deleted 2026-10-11 by b9e2282b2; code recoverable at b9e2282b2^.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_M3_S17_TAIL_DBUF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_M3_S17_TAIL_DBUF"](), "removed: MOJOLEARN_IDN_M3_S17_TAIL_DBUF (double-buffered Mamba-3 S17 tail) deleted 2026-10-11: LOSER, samba-train-step NV 42.36 -> 39.60 ms (0.935x), AMD 69.04 -> 119.20 ms (1.726x), avg 1.33x; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_NN55_BLOCK_STATUS"](), "removed define MOJOLEARN_NN55_BLOCK_STATUS (arm 2 deleted 2026-10-11, avg 1.15x slower)"
    comptime assert not is_defined["MOJOLEARN_NN32_RETAIN_FORWARD"](), "retired define MOJOLEARN_NN32_RETAIN_FORWARD: use IDN_ACT_RETAIN=1"
    comptime assert not is_defined["MOJOLEARN_NN31_BOUNDED_CHECKPOINTS"](), "retired define MOJOLEARN_NN31_BOUNDED_CHECKPOINTS: use IDN_ACT_RETAIN=2"
    comptime assert not is_defined["MOJOLEARN_IDN_SAMBA_FORWARD_TAPE"](), "retired define MOJOLEARN_IDN_SAMBA_FORWARD_TAPE: use IDN_ACT_RETAIN=3"
    comptime assert not is_defined["MOJOLEARN_IDN_TRAIN_BACKWARD_SCRATCH"](), "retired define MOJOLEARN_IDN_TRAIN_BACKWARD_SCRATCH: use IDN_TRAIN_SCRATCH=1"
    comptime assert not is_defined["MOJOLEARN_NN62_LIFETIME_ARENA"](), "retired define MOJOLEARN_NN62_LIFETIME_ARENA: use IDN_TRAIN_SCRATCH=2"
    comptime assert get_defined_int["MOJOLEARN_IDN_M2_YOFF_EXP",0]() == 0 or get_defined_int["MOJOLEARN_IDN_M2_YOFF_EXP",0]() == 1 or get_defined_int["MOJOLEARN_IDN_M2_YOFF_EXP",0]() == 2, "invalid MOJOLEARN_IDN_M2_YOFF_EXP arm (legal: 0|1|2)"
    comptime assert get_defined_int["MOJOLEARN_IDN_M2_YD_ROWS",8]() == 4 or get_defined_int["MOJOLEARN_IDN_M2_YD_ROWS",8]() == 8, "invalid MOJOLEARN_IDN_M2_YD_ROWS arm (legal: 4|8)"
    comptime assert get_defined_int["MOJOLEARN_IDN_M2_CS_PT",8]() == 8 or get_defined_int["MOJOLEARN_IDN_M2_CS_PT",8]() == 16, "invalid MOJOLEARN_IDN_M2_CS_PT arm (legal: 8|16)"
    comptime assert get_defined_int["MOJOLEARN_IDN_M2_GRAD_FOLD",0]() == 0 or get_defined_int["MOJOLEARN_IDN_M2_GRAD_FOLD",0]() == 1 or get_defined_int["MOJOLEARN_IDN_M2_GRAD_FOLD",0]() == 2, "invalid MOJOLEARN_IDN_M2_GRAD_FOLD arm (legal: 0|1|2)"
    comptime assert get_defined_int["MOJOLEARN_IDN_CNN_CONV_RELU",0]() == 0 or get_defined_int["MOJOLEARN_IDN_CNN_CONV_RELU",0]() == 1 or get_defined_int["MOJOLEARN_IDN_CNN_CONV_RELU",0]() == 2, "invalid MOJOLEARN_IDN_CNN_CONV_RELU arm (legal: 0|1|2)"
    comptime assert get_defined_int["MOJOLEARN_IDN_SEQ_WGRAD",0]() == 0 or get_defined_int["MOJOLEARN_IDN_SEQ_WGRAD",0]() == 1 or get_defined_int["MOJOLEARN_IDN_SEQ_WGRAD",0]() == 2, "invalid MOJOLEARN_IDN_SEQ_WGRAD arm (legal: 0|1|2)"
    comptime assert get_defined_int["MOJOLEARN_IDN_SEQ_LN_LEAF",64]() == 32 or get_defined_int["MOJOLEARN_IDN_SEQ_LN_LEAF",64]() == 64, "invalid MOJOLEARN_IDN_SEQ_LN_LEAF arm (legal: 32|64)"
    # TOMBSTONE: MOJOLEARN_IDN_CE_TOKEN_FOLD=2 (noise) deleted 2026-10-10 by 3034789a2; code recoverable at ca25d9321.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_CE_TOKEN_FOLD-arm2.patch
    comptime assert get_defined_int["MOJOLEARN_IDN_CE_TOKEN_FOLD",0]() != 2, "removed: MOJOLEARN_IDN_CE_TOKEN_FOLD=2 (NI35 token-tree256 v2) retired 2026-10-10: noise, avg 1.00x lm-forward (NV 0.99x / AMD 1.01x), 1.01x lm-train-step (NV 1.01x / AMD 1.00x), mean_nll SAME (main ca25d9321, A/B nv2 v1229/v1231/v1232, amd a1555/a1556/a1557, default v1232/a1557); see docs/TOMBSTONES.md"
    comptime assert get_defined_int["MOJOLEARN_IDN_CE_TOKEN_FOLD",0]() == 0 or get_defined_int["MOJOLEARN_IDN_CE_TOKEN_FOLD",0]() == 1, "invalid MOJOLEARN_IDN_CE_TOKEN_FOLD arm (legal: 0|1; 2 removed 2026-10-10)"
    comptime assert get_defined_int["MOJOLEARN_IDN_LM_RESIDENT_TOKENS",0]() == 0 or get_defined_int["MOJOLEARN_IDN_LM_RESIDENT_TOKENS",0]() == 1 or get_defined_int["MOJOLEARN_IDN_LM_RESIDENT_TOKENS",0]() == 2, "invalid MOJOLEARN_IDN_LM_RESIDENT_TOKENS arm (legal: 0|1|2)"
    comptime assert get_defined_int["MOJOLEARN_IDN_LM_RESIDENT_TOKENS",0]() == 0, "removed 2026-11-09 (lane x7b-step-ops-host-arm): MOJOLEARN_IDN_LM_RESIDENT_TOKENS (arms 1 NI32, 2 NN51), resident token validation of the hand byte LM training step, had no reader after lane x7 moved the step onto the executor; never promoted; see docs/TOMBSTONES.md BYTE_LM_X7_LEFTOVERS"
    comptime assert get_defined_int["MOJOLEARN_IDN_LM_VIEWS",0]() == 0 or get_defined_int["MOJOLEARN_IDN_LM_VIEWS",0]() == 1 or get_defined_int["MOJOLEARN_IDN_LM_VIEWS",0]() == 2 or get_defined_int["MOJOLEARN_IDN_LM_VIEWS",0]() == 3, "invalid MOJOLEARN_IDN_LM_VIEWS arm (legal: 0|1|2|3)"
    # TOMBSTONE: MOJOLEARN_IDN_LM_GROUPED_ADAM (arm 1, LOSER avg 1.03x lm-train-step, NV 1.019 / AMD 1.043) deleted 2026-10-11 by cdb22ee93; code recoverable at cdb22ee93^.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_LM_GROUPED_ADAM.patch
    comptime assert not is_defined["MOJOLEARN_IDN_LM_GROUPED_ADAM"](), "removed: MOJOLEARN_IDN_LM_GROUPED_ADAM (NN56 grouped OOP Adam) deleted 2026-10-11: LOSER, lm-train-step NV 35.51 -> 36.18 ms (1.019x), AMD 42.76 -> 44.61 ms (1.043x), avg 1.03x; arm 2 removed earlier; see docs/TOMBSTONES.md"
    comptime assert get_defined_int["MOJOLEARN_IDN_TRAIN_SCRATCH",0]() == 0 or get_defined_int["MOJOLEARN_IDN_TRAIN_SCRATCH",0]() == 1 or get_defined_int["MOJOLEARN_IDN_TRAIN_SCRATCH",0]() == 2 or get_defined_int["MOJOLEARN_IDN_TRAIN_SCRATCH",0]() == 3, "invalid MOJOLEARN_IDN_TRAIN_SCRATCH arm (legal: 0|1|2|3)"
    # Lane grid-prune (2026-10-07): deleted losers and dead arms (rows in docs/apple-fast/EXPERIMENTS.md
    # "IDENTICAL grid prune"); recoverable at main ab554bb4a. A stale build line must not run the incumbent silently.
    comptime assert not (is_defined["MOJOLEARN_NN34_AFFINE_PREFIX"]() or is_defined["MOJOLEARN_IDN_M1_STATE_WINDOW"]() or is_defined["MOJOLEARN_IDN_M1_PERSISTENT_SCAN"]()), "merged: use -D MOJOLEARN_IDN_M1_SCAN=2 (state_window, NI38) | 3 (persistent); 1 (affine_prefix, NN34) removed 2026-10-11"
    # Lane grid-act-3 (2026-10-08): IDENTICAL grid ge123e6f9 promotions and losers (docs/apple-fast/EXPERIMENTS.md). Deleted code recoverable at main bc10b8b56.
    comptime assert not is_defined["MOJOLEARN_C58_FORECAST4"](), "removed: MOJOLEARN_C58_FORECAST4 retired 2026-10-08: slower, theta/ets family NV 2.53-3.15x / AMD 1.81-3.08x on synthetic and taxi-hourly (2.72x combined), quality SAME (grid ge123e6f9); see EXPERIMENTS.md"
    comptime assert not (is_defined["MOJOLEARN_IDN_NN20_SPLIT_KV"]() or is_defined["MOJOLEARN_IDN_NN20_SPLIT_KV_LEAVES"]()), "removed: MOJOLEARN_IDN_NN20_SPLIT_KV (and _LEAVES) retired 2026-10-08 with the summary_tree arm it split (MOJOLEARN_IDN_ATTN_SOFTMAX=1, grid ge123e6f9 loser); see EXPERIMENTS.md"
    # Lane grid-act-4 (2026-10-08): IDENTICAL grid ge123e6f9 neural promotions and losers (docs/apple-fast/EXPERIMENTS.md). Deleted code recoverable at main 4e3da4282.
    comptime assert not is_defined["MOJOLEARN_IDN_SAMBA_RESIDENT_STEP"](), "promoted 2026-10-08 (grid ge123e6f9, lane/grid-act-4): the device-resident Samba forward and train step are the IDENTICAL default (samba-forward 0.539x, samba-train-step 0.563x, same bits); drop -D MOJOLEARN_IDN_SAMBA_RESIDENT_STEP, use -D MOJOLEARN_IDN_SAMBA_RESIDENT_STEP_OFF for the per-op route (a MOJOLEARN_IDN_CHUNKED_LM_HEAD_V2 build takes the per-op route by itself)"
    comptime assert not is_defined["MOJOLEARN_IDN_M3_ANGLE_CARRY_CACHE"](), "promoted 2026-10-08 (grid ge123e6f9, lane/grid-act-4): the Mamba-3 angle carry cache (NI43 chunk seeds) is the IDENTICAL default (samba-train-step 0.883x, same bits); drop -D MOJOLEARN_IDN_M3_ANGLE_CARRY_CACHE, use -D MOJOLEARN_IDN_M3_ANGLE_CARRY_CACHE_OFF for the per-chunk suffix re-walk"
    # ACT_RETAIN arm 2 promoted 2026-10-08 (lane/grid-act-4, grid ge123e6f9, samba-train-step 0.890x, same bits): absent = 2; =1 and =3 stay arms; _OFF = arm 0 replay.
    comptime assert get_defined_int["MOJOLEARN_IDN_ACT_RETAIN",2]() == 1 or get_defined_int["MOJOLEARN_IDN_ACT_RETAIN",2]() == 2 or get_defined_int["MOJOLEARN_IDN_ACT_RETAIN",2]() == 3, "invalid MOJOLEARN_IDN_ACT_RETAIN arm (legal: 1|2|3; replay is -D MOJOLEARN_IDN_ACT_RETAIN_OFF)"
    comptime assert not (is_defined["MOJOLEARN_IDN_ACT_RETAIN_OFF"]() and is_defined["MOJOLEARN_IDN_ACT_RETAIN"]()), "MOJOLEARN_IDN_ACT_RETAIN_OFF (replay) and MOJOLEARN_IDN_ACT_RETAIN=N are exclusive arms"
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_NN05"](), "removed 2026-10-08 (grid ge123e6f9, lane/grid-act-4): neural_gemm_pair=nn05 (fused gate/up pair kernel) was SLOWER on both vendors: lm-forward NV 2.19x / AMD 2.80x, lm-train-step 3.72x / 2.44x, transformer-forward 3.85x / 3.47x (3.0x combined); same bits; code at main 4e3da4282, see docs/apple-fast/EXPERIMENTS.md"
    # Lane grid-act-5 (2026-10-08): IDENTICAL grid ge123e6f9 gemm_leaf split and neural_gemm_epilogue loser (docs/apple-fast/EXPERIMENTS.md). Deleted code recoverable at main 8ed94710a.
    comptime assert (get_defined_int["MOJOLEARN_IDN_NEURAL_GEMM_EPILOGUE",0]() & 1) == 0, "removed 2026-10-08 (grid ge123e6f9, lane/grid-act-5): neural_gemm_epilogue=mlp (MOJOLEARN_IDN_NEURAL_GEMM_EPILOGUE bit 1, NN06 fused MLP GEMM + bias/ReLU) was SLOWER: mlp-train-step NV 1.9x (2.0 -> 3.8 ms) / AMD 0.97x (3.5 -> 3.4 ms), 1.367x combined; same bits; only =2 (cnn) remains; code at main 8ed94710a, see docs/apple-fast/EXPERIMENTS.md"
    # Lane postmerge-act-5 (2026-10-09): fg2 board-bridge A/B on main 0a7b206f1, one run per arm (nv2 L40S v1021-v1050, amd MI325X a1161-a1190; ratio = arm / fg2.default-linear; neural nv2 v1010/v1012, amd a1141/a1143; docs/apple-fast/EXPERIMENTS.md). Two promoted (ridge bits change, equal on both vendors), two removed; deleted code recoverable at main 0a7b206f1.
    comptime assert not is_defined["MOJOLEARN_IDN_AF_VEC_FUSED"](), "removed 2026-10-09 (neural A/B nv2 v1010/v1012, amd a1141/a1143, lane/postmerge-act-5): neural-io-2 Adafactor fused vector step (sequence/af_fused.mojo, OP_AF_VFUSE/OP_AF_VFIN) was NOISE: adafactor synthetic NV 13.671 -> 13.738 ms (1.00x), AMD 8.620 -> 8.186 ms (0.95x), avg 0.98x, digest aa99a3dc unchanged; code at main 0a7b206f1; see EXPERIMENTS.md"
    # Lane owed-deletions-D2 (2026-10-09): owed deletions of DROPPED Apple FAST / neural defines (docs/TOMBSTONES.md, docs/apple-fast/EXPERIMENTS.md). Deleted code recoverable at main b639a2bd2.
    comptime assert not (is_defined["MOJOLEARN_AFN_OPT_FUSE_SCAN"]() or is_defined["MOJOLEARN_AFN26_OPT_FUSE_SCAN"]()), "removed 2026-10-09 (lane/owed-deletions-D2): MOJOLEARN_AFN_OPT_FUSE_SCAN (and alias MOJOLEARN_AFN26_OPT_FUSE_SCAN), the fused optimizer refusal scan, was DROPPED-noise: M3 rab19 adam within +-2%, digest moves, no quality metric; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not (is_defined["MOJOLEARN_AFN_OPT_VEC4"]() or is_defined["MOJOLEARN_AFN26_OPT_VEC4"]()), "removed 2026-10-09 (lane/owed-deletions-D2): MOJOLEARN_AFN_OPT_VEC4 (and alias MOJOLEARN_AFN26_OPT_VEC4), the 4-wide FAST Adam update, was DROPPED-noise: M3 rab19 adam within +-2%, digest moves, no quality metric; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not (is_defined["MOJOLEARN_AFN_OPT_RESIDENT_STATE"]() or is_defined["MOJOLEARN_AFN26_OPT_RESIDENT_STATE"]()), "removed 2026-10-09 (lane/owed-deletions-D2): MOJOLEARN_AFN_OPT_RESIDENT_STATE (and alias MOJOLEARN_AFN26_OPT_RESIDENT_STATE), the FAST pooled optimizer/loss scratch, was DROPPED-noise: M3 rab19 adam within +-2%, digest moves, no quality metric; IDENTICAL keeps its scratch pool; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not (is_defined["MOJOLEARN_MOE_FAST_MMA_KB32"]()), "removed 2026-10-09 (lane/owed-deletions-D2): MOJOLEARN_MOE_FAST_MMA_KB32, the MoE 32-word slab, was DROPPED-slower as the KB32 + WIDE + PF bundle: M3 rab10-moemmaall moe synthetic 71.3 -> 146.8 ms; MM_KB is 16; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not (is_defined["MOJOLEARN_OPT_FAST_MAP_DOWN"]()), "removed 2026-10-09 (lane/owed-deletions-D2): MOJOLEARN_OPT_FAST_MAP_DOWN, the map_to_host optimizer read-back, was DROPPED-slower: M3 rab7-optfastmapdo rmsprop/adagrad/adamax/nadam +81% .. +86%; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not (is_defined["MOJOLEARN_OPT_FAST_RAW_DOWN"]()), "removed 2026-10-09 (lane/owed-deletions-D2): MOJOLEARN_OPT_FAST_RAW_DOWN, the raw chunked optimizer read-back, was DROPPED-slower: M3 rab7-optfastrawdo rmsprop/adagrad/adamax/nadam +76% .. +80%; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not (is_defined["MOJOLEARN_OPT_FAST_PIPE_CH"]()), "removed 2026-10-09 (lane/owed-deletions-D2): MOJOLEARN_OPT_FAST_PIPE_CH, the optimizer transfer-chunk override (=524288 arm), was dropped with the gap-optim read-back batch (DROPPED-slower, rab7 MAP_DOWN +81-86%, RAW_DOWN +76-80%; this arm untried); OPT_PIPE_CH stays 1 << 21; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not (is_defined["MOJOLEARN_SEQ_FAST_LSTM_SCAN"]() or is_defined["MOJOLEARN_SEQ_FAST_LSTM_SCAN_WIDE"]()), "removed 2026-10-09 (lane/owed-deletions-D2): MOJOLEARN_SEQ_FAST_LSTM_SCAN (and its _SCAN_WIDE arm), the FAST one-launch LSTM scan, was DROPPED-quality (BROKEN): M3 rab10 lstm-clf accuracy 0.9608 -> 0.5002, lstm-reg r2 0.9804 -> -0.1043; the IDENTICAL MOJOLEARN_IDN_SEQ_LSTM_SCAN keeps the kernels; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not (is_defined["MOJOLEARN_SEQ_FAST_LSTM_SCAN_SMEM"]()), "removed 2026-10-09 (lane/owed-deletions-D2): MOJOLEARN_SEQ_FAST_LSTM_SCAN_SMEM, the threadgroup-staged LSTM scan, was DROPPED-quality with its bundle: M3 rab10 SCAN + SMEM lstm-clf accuracy 0.9608 -> 0.5002, r2 0.9804 -> -0.1043; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not (is_defined["MOJOLEARN_SEQ_FAST_LSTM_WGRAD"]()), "removed 2026-10-09 (lane/owed-deletions-D2): MOJOLEARN_SEQ_FAST_LSTM_WGRAD, the FAST split-K LSTM weight-gradient folds, was DROPPED-quality with the scan bundle: M3 lstm-clf accuracy 0.9608 -> 0.5002, lstm-reg r2 0.9804 -> -0.1043; never A/B-ed alone; code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not (is_defined["MOJOLEARN_SEQ_FAST_MAP_DOWN"]()), "removed 2026-10-09 (lane/owed-deletions-D2): MOJOLEARN_SEQ_FAST_MAP_DOWN, the map_to_host sequence read-back, was DROPPED-slower: M3 layernorm 50.2 -> 79.8 ms (+59%); code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not (is_defined["MOJOLEARN_SEQ_FAST_RAW_DOWN"]()), "removed 2026-10-09 (lane/owed-deletions-D2): MOJOLEARN_SEQ_FAST_RAW_DOWN, the raw chunked sequence read-back, was DROPPED-slower: M3 rab7-seqrawdown layernorm 48.99 -> 73.68 ms (+50.4%); code at main b639a2bd2; see docs/TOMBSTONES.md"
    comptime assert not (is_defined["MOJOLEARN_SEQ_FAST_VAR_NODRAIN"]()), "removed 2026-10-09 (lane/owed-deletions-D2): MOJOLEARN_SEQ_FAST_VAR_NODRAIN, the VAR no-second-drain executor flag, was DROPPED-slower: M3 verdicts batch 6 var taxi-hourly +10.6%; code at main b639a2bd2; see docs/TOMBSTONES.md"
    # Lane postmerge-act-6 (2026-10-10), promoted: post-merge A/B on main, one run per arm (neural: nv2 L40S v1052-v1062 on 9f83ea479, amd MI325X a1148-a1159 on 0a7b206f1; classical fg2: nv2 v1021-v1050, amd a1161-a1190; docs/apple-fast/EXPERIMENTS.md). The old on-defines are refused; each _OFF restores the old path.
    comptime assert not is_defined["MOJOLEARN_NI12_IMPLICIT_CONV"](), "promoted 2026-10-10 (post-merge A/B nv2 v1052 -> v1053, amd a1152 -> a1153, lane/postmerge-act-6): the implicit conv (cols gathered from x, never materialized) is the IDENTICAL default (conv2d NV 0.74x / AMD 0.69x, resnet-block NV 0.74x / AMD 0.70x, same bits); drop -D MOJOLEARN_NI12_IMPLICIT_CONV, use -D MOJOLEARN_NI12_IMPLICIT_CONV_OFF for the materialized im2col path"
    comptime assert not is_defined["MOJOLEARN_IDN_MOE_GROUPED_GEMM"](), "promoted 2026-10-10 (post-merge A/B nv2 v1052 -> v1055, amd a1152 -> a1155, lane/postmerge-act-6): the MoE expert products on identical_gemm_into are the IDENTICAL default (moe NV 0.61x / AMD 0.81x; digest 393fefe9 -> c7c1ee2c on both vendors, host twin with them); drop -D MOJOLEARN_IDN_MOE_GROUPED_GEMM, use -D MOJOLEARN_IDN_MOE_GROUPED_GEMM_OFF for the moe_reg one-chain tile"
    comptime assert not is_defined["MOJOLEARN_IDN_LN_ROW_WARP"](), "promoted 2026-10-10 (post-merge A/B nv2 v1056 -> v1057, amd a1156 -> a1157, lane/postmerge-act-6): the 32-lane LayerNorm row fold is the IDENTICAL default (layernorm NV 0.77x / AMD 0.79x; digest 5fe9165c -> 2a8100a0 on both vendors, host twin with them); drop -D MOJOLEARN_IDN_LN_ROW_WARP, use -D MOJOLEARN_IDN_LN_ROW_WARP_OFF for the one-chain row"
    # Lane postmerge-act-6 (2026-10-10), removed: losers of the same A/B (docs/TOMBSTONES.md, docs/apple-fast/EXPERIMENTS.md). Deleted code recoverable at main 9f83ea479.
    comptime assert not is_defined["MOJOLEARN_IDN_CNN_BN_FUSE"](), "removed 2026-10-10 (lane/postmerge-act-6): IDN_CNN_BN_FUSE (the ResNet BasicBlock BatchNorm epilogues fused, x_cnn_idn_flags bit 6) was NOISE/slower: resnet-block NV 0.94x / AMD 1.02x, conv2d NV 1.14x / AMD 1.03x (nv2 v1052 -> v1054, amd a1152 -> a1154); code at main 9f83ea479; see docs/TOMBSTONES.md"
    comptime assert not (is_defined["MOJOLEARN_IDN_EMB_SCAN_DEFERRED"]() or is_defined["MOJOLEARN_IDN_EMB_SCAN_DEFERRED_OFF"]()), "removed 2026-10-10 (lane/postmerge-act-6): IDN_EMB_SCAN_DEFERRED (the default-on deferred dY non-finite scan of the resident embedding backward) LOST to its _OFF arm: embedding NV 2.70 -> 2.28 ms (0.84x) / AMD 2.09 -> 1.94 ms (0.92x) with _OFF (nv2 v1056 -> v1059, amd a1156 -> a1159), digest 2753e61a unchanged; the eager scans are the only path, drop both defines; code at main 9f83ea479; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_MOE_FAST_MMA_WIDE"](), "removed 2026-10-10 (lane/postmerge-act-6): MOE_FAST_MMA_WIDE (the MoE FAST hidden product as 64 x 64 blocks) was DROPPED-slower as the bundle KB32 + WIDE + PF: M3 rab10-moemmaall moe synthetic 71.3 -> 146.8 ms (2x slower), never A/B-ed alone; MOE_FAST_MMA itself stays the FAST + Apple default; code at main 9f83ea479; see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_MOE_FAST_MMA_PF"](), "removed 2026-10-10 (lane/postmerge-act-6): MOE_FAST_MMA_PF (the MoE FAST next-slab register prefetch) was DROPPED-slower as the bundle KB32 + WIDE + PF: M3 rab10-moemmaall moe synthetic 71.3 -> 146.8 ms (2x slower), never A/B-ed alone; MOE_FAST_MMA itself stays the FAST + Apple default; code at main 9f83ea479; see docs/TOMBSTONES.md"
    # Lane i17-decide-linalg-seq (2026-10-11), removed: losers of the A/B on main b2657cc89 / 2afa942e3, one run per arm (docs/TOMBSTONES.md,
    # docs/apple-fast/EXPERIMENTS.md, experiments/removed/*.patch).
    comptime assert not is_defined["MOJOLEARN_IDN_SEQ_FMA"](), "removed 2026-10-11 (lane/i17-decide-linalg-seq, 0c8ea9eeb): IDN_SEQ_FMA (pinned fma in the theta/ETS recurrences) was NOISE with bits changed: theta 1.00x / 1.00x, damped-ets 1.00x / 1.08x, v0039/v0040, a1622/a1623; code at 0c8ea9eeb^, patch experiments/removed/MOJOLEARN_IDN_SEQ_FMA.patch"
    comptime assert not is_defined["MOJOLEARN_IDN_MAMBA1_ELEMENTWISE_FUSED"](), "removed 2026-10-11 (lane/grid-act-16): IDN_MAMBA1_ELEMENTWISE_FUSED (NI46) was SLOWER: mamba1-forward NV 2.31x / AMD 1.00x (avg 1.52x), grid g50ebe26a5"
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_GEMM_OZAKI_SLICES"](), "removed 2026-10-11 (lane/grid-act-16): IDN_NEURAL_GEMM_OZAKI_SLICES (NN-OZ int8 Ozaki neural GEMM) was SLOWER on every neural workload: slices 4/5 1.2x-30x, grid g50ebe26a5"
    # wave-2 w2-decide-neural (2026-10-11): A/B on main c8ef6c66d, one run per arm, NV L40S nv2 / AMD MI325X, digests NV==AMD
    # on every cell (docs/TOMBSTONES.md, docs/apple-fast/EXPERIMENTS.md, experiments/removed/*.patch).
    # Promoted: the old positive define is refused; -D <NAME>_OFF restores the old path.
    comptime assert not is_defined["MOJOLEARN_IDN_CNN_RESIDENT_GRADS"](), "promoted 2026-10-11 (lane/w2-decide-neural): CNN resident grads is the IDENTICAL default (avg 0.754 over 9 layer lanes, same bits, nv2 v0085 / amd a1667); drop -D MOJOLEARN_IDN_CNN_RESIDENT_GRADS, -D MOJOLEARN_IDN_CNN_RESIDENT_GRADS_OFF restores the old glue"
    comptime assert not is_defined["MOJOLEARN_IDN_BN_GRID_STATS"](), "promoted 2026-10-11 (lane/w2-decide-neural): BN grid statistics are the IDENTICAL default (batchnorm1d 0.471, batchnorm2d 0.486, bits change NV==AMD, nv2 v0087 / amd a1669; quality re-race v0137/v0138, a1719/a1720); drop -D MOJOLEARN_IDN_BN_GRID_STATS, -D MOJOLEARN_IDN_BN_GRID_STATS_OFF restores BN_FOLD_BLOCK"
    comptime assert not is_defined["MOJOLEARN_IDN_NM_SPEC"](), "promoted 2026-10-11 (lane/w2-decide-neural): the speculative Nelder-Mead (theta + no-season ETS) is the IDENTICAL default (0.10-0.34x, same bits, nv2 v0098 / amd a1680); drop -D MOJOLEARN_IDN_NM_SPEC, -D MOJOLEARN_IDN_NM_SPEC_OFF restores op_theta / op_ets"
    # TOMBSTONE: MOJOLEARN_IDN_CNN_DX_IMPLICIT (noise 0.980) deleted 2026-10-11 by 499997a56; code recoverable at 499997a56^.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_CNN_DX_IMPLICIT.patch
    comptime assert not is_defined["MOJOLEARN_IDN_CNN_DX_IMPLICIT"](), "removed 2026-10-11 (lane/w2-decide-neural, 499997a56): IDN_CNN_DX_IMPLICIT (implicit transposed-conv dx) was NOISE: conv2d 0.931, resnet-block 0.955, conv1d 1.055, mean 0.980 (nv2 v0083, amd a1665)"
    # TOMBSTONE: MOJOLEARN_IDN_CNN_WGRAD_LEAF1024 (noise 0.972) deleted 2026-10-11 by 499997a56; code recoverable at 499997a56^.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_CNN_WGRAD_LEAF1024.patch
    comptime assert not is_defined["MOJOLEARN_IDN_CNN_WGRAD_LEAF1024"](), "removed 2026-10-11 (lane/w2-decide-neural, 499997a56): IDN_CNN_WGRAD_LEAF1024 (1024-row dW leaves) was NOISE: conv1d 0.954, conv2d 0.976, resnet-block 0.985, mean 0.972 (nv2 v0084, amd a1666)"
    # TOMBSTONE: MOJOLEARN_IDN_GPOOL_WARP (noise/NV spike, bits changed on global-avgpool) deleted 2026-10-11 by f5aafae7f; code recoverable at f5aafae7f^.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GPOOL_WARP.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GPOOL_WARP"](), "removed 2026-10-11 (lane/w2-decide-neural, f5aafae7f): IDN_GPOOL_WARP (block-per-plane global pools) was NOISE/unusable: global-avgpool 0.868 with bits changed, global-maxpool AMD 1.065, NV 216 ms spike; CNN resident grads already takes both pools to ~0.1 ms (nv2 v0086, amd a1668)"
    # TOMBSTONE: MOJOLEARN_IDN_MOE_RESIDENT (LOSER 4.64x moe synthetic) deleted 2026-10-11 by 978e7f43b; code recoverable at 978e7f43b^.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_MOE_RESIDENT.patch
    comptime assert not is_defined["MOJOLEARN_IDN_MOE_RESIDENT"](), "removed 2026-10-11 (lane/w2-decide-neural, 978e7f43b): IDN_MOE_RESIDENT (grouped device-offset MoE GEMM, no fills) was SLOWER: moe synthetic NV 5.94x / AMD 3.34x, avg 4.64x (nv2 v0101, amd a1683)"
    # wave-2 w2-decide-lm (2026-10-11): A/B on main c8ef6c66d, same box, one run per arm, NV==AMD digests (docs/TOMBSTONES.md,
    # docs/apple-fast/EXPERIMENTS.md, experiments/removed/*.patch). Five deleted, none promoted.
    # TOMBSTONE: MOJOLEARN_IDN_GEMM_FILL_SPLITK (NOISE/LOSER) deleted 2026-10-11 by 026a811d5; code recoverable at 026a811d5^.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GEMM_FILL_SPLITK.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GEMM_FILL_SPLITK"](), "removed 2026-10-11 (lane/w2-decide-lm, 026a811d5): IDN_GEMM_FILL_SPLITK was NOISE/LOSER: gemm 1.045, mamba2 1.115 (NV 1.248), mlp 0.941, transformer 1.39 (NV), moe 1.010; AMD all 0.98-1.02; nv2 v0116/v0121, amd a1698/a1703. Restore with experiments/removed/MOJOLEARN_IDN_GEMM_FILL_SPLITK.patch."
    # TOMBSTONE: MOJOLEARN_IDN_LM_LEAN_STEP (NOISE) deleted 2026-10-11 by 8cc78c8ae; code recoverable at 8cc78c8ae^.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_LM_LEAN_STEP.patch
    comptime assert not is_defined["MOJOLEARN_IDN_LM_LEAN_STEP"](), "removed 2026-10-11 (lane/w2-decide-lm, 8cc78c8ae): IDN_LM_LEAN_STEP was NOISE: lm-train-step 0.982 (NV 0.975, AMD 0.990), lm-forward 0.942; nv2 v0119, amd a1701. Restore with experiments/removed/MOJOLEARN_IDN_LM_LEAN_STEP.patch."
    # TOMBSTONE: MOJOLEARN_IDN_LM_LOGITS_LEAN (NOISE) deleted 2026-10-11 by 066722146; code recoverable at 066722146^.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_LM_LOGITS_LEAN.patch
    comptime assert not is_defined["MOJOLEARN_IDN_LM_LOGITS_LEAN"](), "removed 2026-10-11 (lane/w2-decide-lm, 066722146): IDN_LM_LOGITS_LEAN was NOISE: lm-forward 0.925 (NV 0.829, AMD 1.021; the NV gain is common to all B arms), lm-train-step 0.999; nv2 v0120, amd a1702. Restore with experiments/removed/MOJOLEARN_IDN_LM_LOGITS_LEAN.patch."
    # TOMBSTONE: MOJOLEARN_IDN_ATTN_DEFER_REGIME (LOSER) deleted 2026-10-11 by 68d9654bb; code recoverable at 68d9654bb^.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ATTN_DEFER_REGIME.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ATTN_DEFER_REGIME"](), "removed 2026-10-11 (lane/w2-decide-lm, 68d9654bb): IDN_ATTN_DEFER_REGIME was LOSER: transformer-forward 1.376 (NV 1.647, AMD 1.104; NV confirmed by v0139), lm-forward 0.914, lm-train-step 0.989; nv2 v0117, amd a1699. Restore with experiments/removed/MOJOLEARN_IDN_ATTN_DEFER_REGIME.patch."
    # TOMBSTONE: MOJOLEARN_IDN_TF_SESSION_LEAN (LOSER) deleted 2026-10-11 by 68d9654bb; code recoverable at 68d9654bb^.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_TF_SESSION_LEAN.patch
    comptime assert not is_defined["MOJOLEARN_IDN_TF_SESSION_LEAN"](), "removed 2026-10-11 (lane/w2-decide-lm, 68d9654bb): IDN_TF_SESSION_LEAN was LOSER: transformer-forward 1.350 (NV 1.689, AMD 1.011); nv2 v0118, amd a1700. Restore with experiments/removed/MOJOLEARN_IDN_TF_SESSION_LEAN.patch."
    # Lane del-recurrent (2026-10-19), removed: the recurrent trainer's hand-written backward glue (recurrent.backward,
    # HandStep); rnn_fit runs the tape step (AutogradRnnStep) only. docs/TOMBSTONES.md, docs/apple-fast/EXPERIMENTS.md.
    # TOMBSTONE: MOJOLEARN_AUTOGRAD_RNN_OFF (hand glue, superseded by the tape) deleted 2026-10-19 by e88ae4e38; code recoverable at e88ae4e38^.
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_RNN_OFF.patch
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_RNN_OFF"](), "removed 2026-10-19 (lane/del-recurrent, e88ae4e38): MOJOLEARN_AUTOGRAD_RNN_OFF (the hand-chained recurrent backward, HandStep) is gone; the tape step gave the same bits on Apple, NVIDIA and AMD (NV lstm 0.1424 -> 0.1418 s, AMD 0.1206 -> 0.1207 s, per-step chain, pass 5); recoverable at e88ae4e38^"
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_RNN"](), "removed 2026-10-19 (lane/del-recurrent, e88ae4e38): MOJOLEARN_AUTOGRAD_RNN (the old opt-in spelling of the recurrent tape step) is gone: the tape is rnn_fit's only backward; drop the define"
    # end lane del-recurrent
    # ---- lane del-mamba (2026-10-19), removed: the hand Mamba-1/2 prefill backward glue; the tape is the only GPU backward
    # (docs/TOMBSTONES.md, docs/apple-fast/EXPERIMENTS.md "Hand-path deletions (2026-10-19)").
    # TOMBSTONE: MOJOLEARN_AUTOGRAD_MAMBA_OFF (superseded by the tape) deleted 2026-10-19 by 47582a7fb; code recoverable at e378999c7^.
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_MAMBA_OFF.patch
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_MAMBA_OFF"](), "removed 2026-10-19 (lane/del-mamba, 47582a7fb): the hand Mamba-1/2 prefill backward arm is deleted; the tape (autograd/rules_mamba.mojo) is the only GPU backward, same bits on Apple, NVIDIA and AMD. Restore with experiments/removed/MOJOLEARN_AUTOGRAD_MAMBA_OFF.patch"
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_MAMBA"](), "removed 2026-10-19 (lane/del-mamba, 47582a7fb): MOJOLEARN_AUTOGRAD_MAMBA was the opt-in spelling of the tape backward, now the only path; drop the define"
    # ---- end lane del-mamba
    # ---- orchestrator (2026-10-19): the other hand-path deletions whose lanes kept their refusal beside the code
    # (del-mlp, del-transformer, del-lm, del-cnn, del-smallmlp); repeated here so every removed define is refused centrally.
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_MLP_OFF"](), "removed 2026-10-19 (lane/del-mlp, 6784245b2): the hand MLP backward is deleted; the tape is the only backward (same 48 digests on every column). Restore with experiments/removed/MOJOLEARN_AUTOGRAD_MLP_OFF.patch"
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_TRANSFORMER_OFF"]() and not is_defined["MOJOLEARN_AUTOGRAD_TRANSFORMER"](), "removed 2026-10-19 (lane/del-transformer, dd4194a23): the transformer bindings call the tape only; drop the define. Restore with experiments/removed/MOJOLEARN_AUTOGRAD_TRANSFORMER.patch"
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_LM_OFF"]() and not is_defined["MOJOLEARN_AUTOGRAD_LM"](), "removed 2026-10-19 (lane/del-lm, a54a5fb2e): the hand LM backward walk is deleted; the tape is the only backward. Restore with experiments/removed/MOJOLEARN_AUTOGRAD_LM_OFF.patch"
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_CNN_OFF"]() and not is_defined["MOJOLEARN_AUTOGRAD_CNN"](), "removed 2026-10-19 (lane/del-cnn, ddd4fa509): the hand CNN fit-epoch backward is deleted; the tape is the only backward. Restore with experiments/removed/MOJOLEARN_AUTOGRAD_CNN_OFF.patch"
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_SMALL_MLP_OFF"]() and not is_defined["MOJOLEARN_AUTOGRAD_SMALL_MLP"](), "removed 2026-10-19 (lane/del-smallmlp, 3cd0063d7): the hand small-MLP backward arm is deleted; the tape is the only backward. Restore with experiments/removed/MOJOLEARN_AUTOGRAD_SMALL_MLP_OFF.patch"
    # ---- end orchestrator block
    # ---- lane del-samba (2026-10-19): the hand Samba train-step backward walk, deleted; the tape is the only backward
    # (word for word equal on Metal, NVIDIA and AMD; docs/TOMBSTONES.md, docs/apple-fast/EXPERIMENTS.md).
    # TOMBSTONE: MOJOLEARN_AUTOGRAD_SAMBA_OFF (DELETED) deleted 2026-10-19 by 70ce81192; code recoverable at 70ce81192^.
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_SAMBA_OFF.patch
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_SAMBA_OFF"](), "removed 2026-10-19 (lane/del-samba, 70ce81192): MOJOLEARN_AUTOGRAD_SAMBA_OFF (the hand Samba train-step backward walk) was deleted: the tape backward is the only one and gave the same words on Metal, NVIDIA and AMD. Restore with experiments/removed/MOJOLEARN_AUTOGRAD_SAMBA_OFF.patch."
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_SAMBA"](), "removed 2026-10-19 (lane/del-samba, 70ce81192): MOJOLEARN_AUTOGRAD_SAMBA (the Samba tape opt-in) is gone: the tape is always on. Restore the switch with experiments/removed/MOJOLEARN_AUTOGRAD_SAMBA_OFF.patch."
    # ---- end lane del-samba
    # ---- lane x9-moe-gcn-rules (2026-11-09): the MoE backward's refusal arm, deleted; the tape is the only MoE backward
    # TOMBSTONE: MOJOLEARN_AUTOGRAD_MOE_OFF (removed) deleted 2026-11-09 by lane/x9-moe-gcn-rules; code recoverable at origin/integ/pre-0867-w2 f59d913b0b.
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_MOE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_MOE_OFF"](), "removed 2026-11-09 (lane/x9-moe-gcn-rules): MOJOLEARN_AUTOGRAD_MOE_OFF (the refusal of the MoE backward) is deleted; the tape (autograd/route_ops.mojo) is the only MoE backward. Restore with experiments/removed/MOJOLEARN_AUTOGRAD_MOE_OFF.patch"
    # ---- end lane x9-moe-gcn-rules
    # ---- lane x2-small-mlp-on-executor (2026-11-09): the hand small-MLP step and its switches, deleted
    # (docs/TOMBSTONES.md, docs/apple-fast/EXPERIMENTS.md).
    # TOMBSTONE: MOJOLEARN_AFN_MLP_FUSED_STEP / _RESIDENT / _MULTISTEP / _ALL, MOJOLEARN_IDN_MLP_GEMM_LEAF128_OFF deleted 2026-11-09 by lane/x2-small-mlp-on-executor; code recoverable at f59d913b0b.
    # Restore: git apply experiments/removed/MOJOLEARN_X2_SMALL_MLP_HAND_ROUTES.patch
    comptime assert not (is_defined["MOJOLEARN_AFN_MLP_FUSED_STEP"]() or is_defined["MOJOLEARN_AFN_MLP_RESIDENT"]() or is_defined["MOJOLEARN_AFN_MLP_MULTISTEP"]() or is_defined["MOJOLEARN_AFN_MLP_ALL"]()), "removed 2026-11-09 (lane/x2-small-mlp-on-executor, plan row X2): SmallMLPTrainer runs as one captured program on the executor (autograd/small_mlp_program.mojo); the Apple FAST fused step and resident session (merged-unmeasured opt-ins) are deleted with the hand step; drop the define. Restore with experiments/removed/MOJOLEARN_X2_SMALL_MLP_HAND_ROUTES.patch"
    # TOMBSTONE restored 2026-11-09 by lane/r1-restore-measured-winners: MOJOLEARN_IDN_MLP_GEMM_LEAF128_OFF is live again (autograd/small_mlp_op.mojo SMALL_MLP_GEMM_LEAF128: the small_mlp op's GEMMs take the incumbent identical_gemm_into); no refusal. MOJOLEARN_SMALL_MLP_GENERIC_OPS (the generic-ops A/B arm) needs none either.
    # ---- end lane x2-small-mlp-on-executor
    # ---- lane x3-mlp-rnn-predict-on-tape (2026-11-08, docs/plans/EXECUTOR_CONSOLIDATION_AND_CLEANUP.md X3 and 5.6): one
    # path per decision in the MLP and recurrent trainers (docs/TOMBSTONES.md, docs/apple-fast/EXPERIMENTS.md).
    # TOMBSTONE: MOJOLEARN_AUTOGRAD_PREDICT (folded in: the tape forward is the one predict path) deleted 2026-11-08 by c3b7dc42e6; code recoverable at c3b7dc42e6^.
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_PREDICT.patch
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_PREDICT"](), "removed 2026-11-08 (lane/x3-mlp-rnn-predict-on-tape, c3b7dc42e6 + 9a9a8e3148): MLP and GRU/LSTM/RNN predict always run the tape forward in inference mode; the hand mlp_predict / mlp_forward and the recurrent hand forward + head are deleted. Drop the define. Restore with experiments/removed/MOJOLEARN_AUTOGRAD_PREDICT.patch"
    # TOMBSTONE: MOJOLEARN_AUTOGRAD_LOSS (rnn_fit: promoted, the blocked fold replaces the serial OP_SUM) deleted 2026-11-08 by dbab55bea7; code recoverable at c3b7dc42e6^.
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_LOSS.patch
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_LOSS"](), "removed 2026-11-08 (lane/x3-mlp-rnn-predict-on-tape, dbab55bea7): rnn_fit's reported loss always uses the blocked fold (loss_value_into; the serial OP_SUM is deleted); mlp_fit keeps its loss block (already blocked, the arm added launches). Drop the define. Restore with experiments/removed/MOJOLEARN_AUTOGRAD_LOSS.patch"
    # TOMBSTONE: MOJOLEARN_AUTOGRAD_OPTIM (promoted: the executor optimizer is the trainers' one update; the hand arm deleted) deleted 2026-11-08 by 75825f479c; code recoverable at c3b7dc42e6^.
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_OPTIM.patch
    # Restored 2026-11-09 by lane/r2-restore-unmeasured-defaults: the hand OP_OPT launch is the default again and
    # -D MOJOLEARN_AUTOGRAD_OPTIM is the A/B arm (the executor optimizer), so the define is live: no refusal.
    # TOMBSTONE: MOJOLEARN_AUTOGRAD_DATA (promoted: autograd/data.mojo epoch_order is the one epoch order; the hand launches deleted) deleted 2026-11-08 by 75825f479c; code recoverable at c3b7dc42e6^.
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_DATA.patch
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_DATA"](), "removed 2026-11-08 (lane/x3-mlp-rnn-predict-on-tape, 75825f479c): rnn_fit always takes its epoch order from autograd/data.mojo epoch_order (same launches, same key); the inline launches are deleted. Drop the define. Restore with experiments/removed/MOJOLEARN_AUTOGRAD_DATA.patch"
    # ---- end lane x3-mlp-rnn-predict-on-tape
    # TOMBSTONE: MOJOLEARN_NI18_FUSED_EPOCH_GATHER (removed with its route) deleted 2026-11-09 by lane/x8-cnn-on-executor; code recoverable at f59d913b0b.
    # Restore: git apply experiments/removed/X8_CNN_HAND_EPOCH.patch
    # TOMBSTONE: MOJOLEARN_IDN_CNN_EPOCH_DEV_OFF (removed with its route) deleted 2026-11-09 by lane/x8-cnn-on-executor; code recoverable at f59d913b0b.
    comptime assert not is_defined["MOJOLEARN_IDN_CNN_EPOCH_DEV_OFF"](), "removed 2026-11-09 (lane/x8-cnn-on-executor): MOJOLEARN_IDN_CNN_EPOCH_DEV_OFF chose the CNNClassifier epoch's row order and Adam scalars between host and device; that epoch is deleted (the classifier trains through Program.fit); recoverable at f59d913b0b, see docs/TOMBSTONES.md"
    comptime assert not is_defined["MOJOLEARN_NI18_FUSED_EPOCH_GATHER"](), "removed 2026-11-09 (lane/x8-cnn-on-executor): MOJOLEARN_NI18_FUSED_EPOCH_GATHER fused the CNNClassifier device epoch's row gather, which is deleted (the classifier is a captured program on the executor); never measured; recoverable at f59d913b0b, see docs/TOMBSTONES.md"
    # ---- lane m2-neural-leftovers (2026-11-09): the small-MLP primitives and the NN64 session switch, deleted
    # TOMBSTONE: MOJOLEARN_NN64_SESSION_PACK (removed: its mlp_sessions entry is deleted) deleted 2026-11-09 by lane/m2-neural-leftovers; code recoverable at 0ce88411d3.
    # Restore: git apply experiments/removed/M2_SMALL_MLP_PRIMITIVES.patch
    comptime assert not is_defined["MOJOLEARN_NN64_SESSION_PACK"](), "removed 2026-11-09 (lane/m2-neural-leftovers): MOJOLEARN_NN64_SESSION_PACK chose per-session vs packed forwards of the training binding's mlp_sessions entry, which is deleted; SmallMLPTrainer.predict_logits_sessions and training.mlp_inference_sessions run their sessions packed as one forward of the small-MLP program (autograd/small_mlp_program.mojo). Restore with experiments/removed/M2_SMALL_MLP_PRIMITIVES.patch"
    # TOMBSTONE: MOJOLEARN_AFN_ATTN_ARENA (removed: dead; no caller built TransformerWorkspace after lanes x45c and x6) deleted 2026-11-09 by lane/m2-neural-leftovers; code recoverable at 0ce88411d3.
    # Restore: git apply experiments/removed/MOJOLEARN_AFN_ATTN_ARENA.patch
    comptime assert not is_defined["MOJOLEARN_AFN_ATTN_ARENA"](), "removed 2026-11-09 (lane/m2-neural-leftovers): MOJOLEARN_AFN_ATTN_ARENA (A07, Apple FAST, never measured) put TransformerWorkspace's buffers in one device arena; nothing built that workspace after the block forward moved onto the attention step op (lanes x45c, x6), so the workspace, its key and the arena arm are deleted. Restore with experiments/removed/MOJOLEARN_AFN_ATTN_ARENA.patch"
    # ---- end lane m2-neural-leftovers
    # ---- lane expdel-04-neural (2026-11-15): experiment deletions (docs/plans/EXPERIMENT_DELETIONS_2026-11-15.md)
    # TOMBSTONE: MOJOLEARN_AFN_OPTIM_ALL (NEUTRAL) deleted 2026-11-15 by lane/expdel-04-neural; code recoverable at 54b6a264fc.
    # Restore: git apply experiments/removed/MOJOLEARN_AFN_OPTIM_ALL.patch
    comptime assert not is_defined["MOJOLEARN_AFN_OPTIM_ALL"](), "removed 2026-11-15 (lane/expdel-04-neural): MOJOLEARN_AFN_OPTIM_ALL (bundle of the Apple FAST optimizer candidates) was NEUTRAL: M3 rab2-combo equals TRAIN_OPT_FAST_PIPE_DOWN alone, digest moves; set the candidate defines one by one; recoverable at 54b6a264fc; see docs/TOMBSTONES.md"
    # TOMBSTONE: MOJOLEARN_AF_FAST_NOFILL_OFF (superseded (AF_FAST_NOFILL default won)) deleted 2026-11-15 by lane/expdel-04-neural; code recoverable at 4ebe87a042.
    # Restore: git apply experiments/removed/MOJOLEARN_AF_FAST_NOFILL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_AF_FAST_NOFILL_OFF"](), "removed 2026-11-15 (lane/expdel-04-neural): MOJOLEARN_AF_FAST_NOFILL_OFF rolled back AF_FAST_NOFILL, the FAST + Apple default that won M3 rab10-afnofill (adafactor 393.2 -> 380.1 ms, digest identical); drop the define; recoverable at 4ebe87a042; see docs/TOMBSTONES.md"
    # TOMBSTONE: MOJOLEARN_LN_FAST_NOFILL_OFF (superseded (LN_FAST_NOFILL default won)) deleted 2026-11-15 by lane/expdel-04-neural; code recoverable at 3004379441.
    # Restore: git apply experiments/removed/MOJOLEARN_LN_FAST_NOFILL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_LN_FAST_NOFILL_OFF"](), "removed 2026-11-15 (lane/expdel-04-neural): MOJOLEARN_LN_FAST_NOFILL_OFF rolled back LN_FAST_NOFILL, the FAST + Apple default that won M3 afc_ab_def (layernorm 48.2 -> 45.5 ms, digest identical); drop the define; recoverable at 3004379441; see docs/TOMBSTONES.md"
    # TOMBSTONE: MOJOLEARN_SEQ_FAST_PIPE_UP_OFF (superseded (SEQ_FAST_PIPE_UP default won)) deleted 2026-11-15 by lane/expdel-04-neural; code recoverable at eff4ddd2d8.
    # Restore: git apply experiments/removed/MOJOLEARN_SEQ_FAST_PIPE_UP_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SEQ_FAST_PIPE_UP_OFF"](), "removed 2026-11-15 (lane/expdel-04-neural): MOJOLEARN_SEQ_FAST_PIPE_UP_OFF rolled back SEQ_FAST_PIPE_UP, the FAST + Apple default KEPT f6a9e7c04 (M3 layernorm 76.0 -> 52.2 ms with PIPE_DOWN, digests identical); drop the define; recoverable at eff4ddd2d8; see docs/TOMBSTONES.md"
    # TOMBSTONE: MOJOLEARN_SEQ_FAST_PIPE_DOWN_OFF (superseded (SEQ_FAST_PIPE_DOWN default won)) deleted 2026-11-15 by lane/expdel-04-neural; code recoverable at 2e31a412c9.
    # Restore: git apply experiments/removed/MOJOLEARN_SEQ_FAST_PIPE_DOWN_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SEQ_FAST_PIPE_DOWN_OFF"](), "removed 2026-11-15 (lane/expdel-04-neural): MOJOLEARN_SEQ_FAST_PIPE_DOWN_OFF rolled back SEQ_FAST_PIPE_DOWN, the FAST + Apple default KEPT f6a9e7c04 (M3 adafactor 684 -> 390 ms, digests identical); drop the define; recoverable at 2e31a412c9; see docs/TOMBSTONES.md"
    # TOMBSTONE: MOJOLEARN_SEQ_FAST_THETA_SNAP_OFF (superseded (SEQ_FAST_THETA_SNAP default won)) deleted 2026-11-15 by lane/expdel-04-neural; code recoverable at e382ce0c14.
    # Restore: git apply experiments/removed/MOJOLEARN_SEQ_FAST_THETA_SNAP_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SEQ_FAST_THETA_SNAP_OFF"](), "removed 2026-11-15 (lane/expdel-04-neural): MOJOLEARN_SEQ_FAST_THETA_SNAP_OFF rolled back SEQ_FAST_THETA_SNAP, the FAST + Apple default KEPT f6a9e7c04 (M3 dynamic-optimized-theta taxi-hourly 2,088 -> 562 ms, same bits); drop the define; recoverable at e382ce0c14; see docs/TOMBSTONES.md"
    # TOMBSTONE: MOJOLEARN_SEQ_FAST_THETA_SPEC_OFF (superseded (SEQ_FAST_THETA_SPEC default won)) deleted 2026-11-15 by lane/expdel-04-neural; code recoverable at c5925a0e35.
    # Restore: git apply experiments/removed/MOJOLEARN_SEQ_FAST_THETA_SPEC_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SEQ_FAST_THETA_SPEC_OFF"](), "removed 2026-11-15 (lane/expdel-04-neural): MOJOLEARN_SEQ_FAST_THETA_SPEC_OFF rolled back SEQ_FAST_THETA_SPEC, the FAST + Apple default KEPT 6c0ba4379 (M3 theta taxi-hourly 218 -> 20.3 ms, quality identical); drop the define; recoverable at c5925a0e35; see docs/TOMBSTONES.md"
    # TOMBSTONE: MOJOLEARN_IDN_SEQ_REG_OFF (superseded (IDN_SEQ_REG promoted)) deleted 2026-11-15 by lane/expdel-04-neural; code recoverable at 38664dbf45.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SEQ_REG_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SEQ_REG_OFF"](), "removed 2026-11-15 (lane/expdel-04-neural): MOJOLEARN_IDN_SEQ_REG_OFF rolled back the IDENTICAL theta register form, PROMOTED 2026-10-11 on NVIDIA and AMD (theta 0.62x, dynamic-optimized-theta 0.71x-0.78x, digests NV == AMD); drop the define; recoverable at 38664dbf45; see docs/TOMBSTONES.md"
    # TOMBSTONE: MOJOLEARN_SEQ_FAST_VAR_COOP_OFF (superseded (SEQ_FAST_VAR_COOP default won)) deleted 2026-11-15 by lane/expdel-04-neural; code recoverable at 72b582962f.
    # Restore: git apply experiments/removed/MOJOLEARN_SEQ_FAST_VAR_COOP_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SEQ_FAST_VAR_COOP_OFF"](), "removed 2026-11-15 (lane/expdel-04-neural): MOJOLEARN_SEQ_FAST_VAR_COOP_OFF rolled back SEQ_FAST_VAR_COOP, the FAST + Apple default (M3 rab13-varcoop var synthetic 4.96 -> 2.73 ms, taxi-hourly 5.01 -> 3.55 ms, forecast_rmse identical); drop the define; recoverable at 72b582962f; see docs/TOMBSTONES.md"
    # TOMBSTONE: MOJOLEARN_IDN_SAMBA_RESIDENT_STEP_OFF (superseded (IDN_SAMBA_RESIDENT_STEP promoted)) deleted 2026-11-15 by lane/expdel-04-neural; code recoverable at 5bfce25001.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SAMBA_RESIDENT_STEP_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SAMBA_RESIDENT_STEP_OFF"](), "removed 2026-11-15 (lane/expdel-04-neural): MOJOLEARN_IDN_SAMBA_RESIDENT_STEP_OFF rolled back the device-resident Samba step, PROMOTED 2026-10-08 on NVIDIA and AMD (grid ge123e6f9: samba-forward 0.44x / 0.66x, samba-train-step 0.41x / 0.78x, hashes equal); drop the define; recoverable at 5bfce25001; see docs/TOMBSTONES.md"
    # ---- end lane expdel-04-neural
    # Restore: git apply experiments/removed/MOJOLEARN_MAMBA3_LEGACY_FRESH_PREFILL.patch
    comptime assert not is_defined["MOJOLEARN_MAMBA3_LEGACY_FRESH_PREFILL"](), "removed 2026-11-15 (lane/expdel2-01-base): MOJOLEARN_MAMBA3_LEGACY_FRESH_PREFILL (a build without the Mamba-3 fresh prefill entries and session capabilities) is gone, no A/B was ever recorded for it; mamba3_forward_fresh and the prefill session are registered on every non-FAST build; restore: experiments/removed/MOJOLEARN_MAMBA3_LEGACY_FRESH_PREFILL.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TRANSFORMER_FRESH_PREFILL.expdel2-01-base.patch
    comptime assert not is_defined["MOJOLEARN_TRANSFORMER_FRESH_PREFILL"](), "removed 2026-11-15 (lane/expdel2-01-base): MOJOLEARN_TRANSFORMER_FRESH_PREFILL (the opt-in export of the per-call transformer_forward_fresh entry off NVIDIA) is gone, no A/B was ever recorded for that arm; the entry is exported on the NVIDIA IDENTICAL build only; restore: experiments/removed/MOJOLEARN_TRANSFORMER_FRESH_PREFILL.expdel2-01-base.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TRANSFORMER_LEGACY_FRESH_PREFILL.patch
    comptime assert not is_defined["MOJOLEARN_TRANSFORMER_LEGACY_FRESH_PREFILL"](), "removed 2026-11-15 (lane/expdel2-01-base): MOJOLEARN_TRANSFORMER_LEGACY_FRESH_PREFILL (the NVIDIA opt-out of the per-call transformer_forward_fresh export) is gone; NVIDIA IDENTICAL builds always export it (NVIDIA's full public A/B gate admitted it); restore: experiments/removed/MOJOLEARN_TRANSFORMER_LEGACY_FRESH_PREFILL.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SAMBA_WEIGHT_EPOCH_GLOBAL.patch
    comptime assert not is_defined["MOJOLEARN_SAMBA_WEIGHT_EPOCH_GLOBAL"](), "removed 2026-11-15 (lane/expdel2-05-base): SAMBA_WEIGHT_EPOCH_GLOBAL (Samba resident copies keyed by the process weight epoch, never measured) is gone; the per-registry key is the only one"
    # Lane expdel2-04-base (2026-11-15): unmeasured arms and old paths deleted under "An unmeasured arm is deleted" (docs/plans/EXPERIMENT_DELETIONS_V2_2026-11-15.md, docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_MAMBA3_FORWARD_DOOR_OLD.patch
    comptime assert not is_defined["MOJOLEARN_IDN_MAMBA3_FORWARD_DOOR_OLD"](), "removed 2026-11-15 (lane/expdel2-04-base): IDN_MAMBA3_FORWARD_DOOR_OLD (old path, Mamba-3 prefill forward door (IDENTICAL); EXPERIMENTS.md PENDING row (owed row fix-mamba-forward-doors, never run)) is gone; the default stays; code recoverable at 39aa5d2695; restore: experiments/removed/MOJOLEARN_IDN_MAMBA3_FORWARD_DOOR_OLD.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_STEP_GLUE_TRIAL.patch
    comptime assert not is_defined["MOJOLEARN_STEP_GLUE_TRIAL"](), "removed 2026-11-15 (lane/expdel2-04-base): STEP_GLUE_TRIAL (opt-in arm, Byte LM step glue (trial build); no A/B recorded for the opt-in arm (rule a, 'NEVER RUN, PENDING MEASUREMENT')) is gone; the default stays; code recoverable at 6027e14ec1; restore: experiments/removed/MOJOLEARN_STEP_GLUE_TRIAL.patch"
    # ---- end lane expdel2-04-base
    # Restore: git apply experiments/removed/MOJOLEARN_C58_TEAM_MIB.patch
    comptime assert not is_defined["MOJOLEARN_C58_TEAM_MIB"](), "removed 2026-11-15 (lane/expdel2-07-base): C58_TEAM_MIB (Sequence (classical series team state budget 64 MiB, IDENTICAL)) was unmeasured opt-in arm; the opt-in arm is gone and the default path is unchanged; code recoverable at d6045cbd0a; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ATTN_STASH.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ATTN_STASH"](), "removed 2026-11-15 (lane/expdel2-07-base): IDN_ATTN_STASH (Neural (fused attention exp-stash memory profile arms 1 recompute / 2 packed / 3 alias_y, IDENTICAL)) was opt-in arms; EXP:1757 split the Samba winner into its own define, lm-train-step 1.24x slower; the opt-in arm is gone and the default path is unchanged; code recoverable at dc10f7b50f; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ATTN_HEAD_SHARE.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ATTN_HEAD_SHARE"](), "removed 2026-11-15 (lane/expdel2-07-base): IDN_ATTN_HEAD_SHARE (Neural (attention forward: one K/V page for four query heads, IDENTICAL)) was opt-in arm; EXP:1483 DROP for the two-head form, four-head arm never A/B'd; the opt-in arm is gone and the default path is unchanged; code recoverable at 7605a821c7; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_MEASURE_I06_GQA.patch
    comptime assert not is_defined["MOJOLEARN_MEASURE_I06_GQA"](), "removed 2026-11-15 (lane/expdel2-07-base): MEASURE_I06_GQA (Neural (I06 forward GQA reuse timing arm of the attention driver)) was unmeasured opt-in arm; the opt-in arm is gone and the default path is unchanged; code recoverable at dbdaaf2c5e; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_MEASURE_I07_LIFETIME.patch
    comptime assert not is_defined["MOJOLEARN_MEASURE_I07_LIFETIME"](), "removed 2026-11-15 (lane/expdel2-07-base): MEASURE_I07_LIFETIME (Neural (I07 retained-estash lifetime timing arm of the attention driver)) was unmeasured opt-in arm; the opt-in arm is gone and the default path is unchanged; code recoverable at 6ada715def; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_TRAIN_LIVE_STATUS.patch
    comptime assert not is_defined["MOJOLEARN_TRAIN_LIVE_STATUS"](), "removed 2026-11-15 (lane/expdel2-07-base): TRAIN_LIVE_STATUS (Neural (byte LM train step live-status candidate, I10)) was unmeasured opt-in arm; the opt-in arm is gone and the default path is unchanged; code recoverable at c38a3b0613; see docs/TOMBSTONES.md"
    # Lane expdel2-08-base (2026-11-15): unmeasured arms and superseded old paths deleted, docs/plans/EXPERIMENT_DELETIONS_V2_2026-11-15.md (docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_PROGRAM_RUN_ENTRY.patch
    comptime assert not is_defined["MOJOLEARN_PROGRAM_RUN_ENTRY"](), "removed 2026-11-15 (lane/expdel2-08-base): PROGRAM_RUN_ENTRY (unmeasured B arm) is gone; the default is kept; code at a435c824c4; restore: experiments/removed/MOJOLEARN_PROGRAM_RUN_ENTRY.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SESSION_CALL_DESCRIPTOR.patch
    comptime assert not is_defined["MOJOLEARN_SESSION_CALL_DESCRIPTOR"](), "removed 2026-11-15 (lane/expdel2-08-base): SESSION_CALL_DESCRIPTOR (unmeasured B arm) is gone; the default is kept; code at 8fe6eaaa01; restore: experiments/removed/MOJOLEARN_SESSION_CALL_DESCRIPTOR.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TRANSFORMER_DECODE_PER_TOKEN_ADMISSION.patch
    comptime assert not is_defined["MOJOLEARN_TRANSFORMER_DECODE_PER_TOKEN_ADMISSION"](), "removed 2026-11-15 (lane/expdel2-08-base): TRANSFORMER_DECODE_PER_TOKEN_ADMISSION (unmeasured B arm) is gone; the default is kept; code at c33bc3f5d0; restore: experiments/removed/MOJOLEARN_TRANSFORMER_DECODE_PER_TOKEN_ADMISSION.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TRANSFORMER_DECODE_PER_TOKEN_DOOR.patch
    comptime assert not is_defined["MOJOLEARN_TRANSFORMER_DECODE_PER_TOKEN_DOOR"](), "removed 2026-11-15 (lane/expdel2-08-base): TRANSFORMER_DECODE_PER_TOKEN_DOOR (unmeasured B arm) is gone; the default is kept; code at b891c32944; restore: experiments/removed/MOJOLEARN_TRANSFORMER_DECODE_PER_TOKEN_DOOR.patch"
    # Lane expdel2-23-neural (2026-11-15): unmeasured arms and old paths deleted, docs/plans/EXPERIMENT_DELETIONS_V2_2026-11-15.md (docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_AFN_MAMBA3_BWD_ARENA.patch
    comptime assert not is_defined["MOJOLEARN_AFN_MAMBA3_BWD_ARENA"](), "removed 2026-11-15 (lane/expdel2-23-neural): AFN_MAMBA3_BWD_ARENA (arm) is gone; the default path is the only one; code at cca09a09f6; restore: experiments/removed/MOJOLEARN_AFN_MAMBA3_BWD_ARENA.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_AFN_MAMBA3_BWD_CHUNK.patch
    comptime assert not is_defined["MOJOLEARN_AFN_MAMBA3_BWD_CHUNK"](), "removed 2026-11-15 (lane/expdel2-23-neural): AFN_MAMBA3_BWD_CHUNK (arm) is gone; the default path is the only one; code at 4f8466c7ca; restore: experiments/removed/MOJOLEARN_AFN_MAMBA3_BWD_CHUNK.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_AFN_SAMBA_ALL.patch
    comptime assert not is_defined["MOJOLEARN_AFN_SAMBA_ALL"](), "removed 2026-11-15 (lane/expdel2-23-neural): AFN_SAMBA_ALL (arm) is gone; the default path is the only one; code at 4f8466c7ca; restore: experiments/removed/MOJOLEARN_AFN_SAMBA_ALL.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_GATHER_EPOCH_OFF.patch
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_GATHER_EPOCH_OFF"](), "removed 2026-11-15 (lane/expdel2-23-neural): AUTOGRAD_GATHER_EPOCH_OFF (old path) is gone; the default is kept; code at f96a014a39; restore: experiments/removed/MOJOLEARN_AUTOGRAD_GATHER_EPOCH_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_BYTE_LM_GENERIC_OPS.patch
    comptime assert not is_defined["MOJOLEARN_BYTE_LM_GENERIC_OPS"](), "removed 2026-11-15 (lane/expdel2-23-neural): BYTE_LM_GENERIC_OPS (arm) is gone; the default path is the only one; code at b4f0e643f7; restore: experiments/removed/MOJOLEARN_BYTE_LM_GENERIC_OPS.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_CNN_GENERIC_OPS.patch
    comptime assert not is_defined["MOJOLEARN_CNN_GENERIC_OPS"](), "removed 2026-11-15 (lane/expdel2-23-neural): CNN_GENERIC_OPS (arm) is gone; the default path is the only one; code at aefeacb577; restore: experiments/removed/MOJOLEARN_CNN_GENERIC_OPS.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_EMB_AUTO_SORT_MIN_CELLS.patch
    comptime assert not is_defined["MOJOLEARN_EMB_AUTO_SORT_MIN_CELLS"](), "removed 2026-11-15 (lane/expdel2-23-neural): EMB_AUTO_SORT_MIN_CELLS (arm) is gone; the default path is the only one; code at 2da7f516d6; restore: experiments/removed/MOJOLEARN_EMB_AUTO_SORT_MIN_CELLS.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_EMB_SERIAL_RUN_BEGIN.patch
    comptime assert not is_defined["MOJOLEARN_EMB_SERIAL_RUN_BEGIN"](), "removed 2026-11-15 (lane/expdel2-23-neural): EMB_SERIAL_RUN_BEGIN (arm) is gone; the default path is the only one; code at 37ef16948e; restore: experiments/removed/MOJOLEARN_EMB_SERIAL_RUN_BEGIN.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ELEM_VEC4_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ELEM_VEC4_OFF"](), "removed 2026-11-15 (lane/expdel2-23-neural): IDN_ELEM_VEC4_OFF (old path) is gone; the default is kept; code at b57fa1c4dd; restore: experiments/removed/MOJOLEARN_IDN_ELEM_VEC4_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_EMB_RADIX_SORT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_EMB_RADIX_SORT_OFF"](), "removed 2026-11-15 (lane/expdel2-23-neural): IDN_EMB_RADIX_SORT_OFF (old path) is gone; the default is kept; code at 0899483c36; restore: experiments/removed/MOJOLEARN_IDN_EMB_RADIX_SORT_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_FOLD_STAGED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_FOLD_STAGED_OFF"](), "removed 2026-11-15 (lane/expdel2-23-neural): IDN_FOLD_STAGED_OFF (old path) is gone; the default is kept; code at 479d223bd3; restore: experiments/removed/MOJOLEARN_IDN_FOLD_STAGED_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_M2_BWD_CELL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_M2_BWD_CELL_OFF"](), "removed 2026-11-15 (lane/expdel2-23-neural): IDN_M2_BWD_CELL_OFF (old path) is gone; the default is kept; code at a6c819032b; restore: experiments/removed/MOJOLEARN_IDN_M2_BWD_CELL_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_M3_ANGLE_CARRY_CACHE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_M3_ANGLE_CARRY_CACHE_OFF"](), "removed 2026-11-15 (lane/expdel2-23-neural): IDN_M3_ANGLE_CARRY_CACHE_OFF (old path) is gone; the default is kept; code at 41312ea653; restore: experiments/removed/MOJOLEARN_IDN_M3_ANGLE_CARRY_CACHE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_M3_ANGLE_DT_SUFFIX_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_M3_ANGLE_DT_SUFFIX_OFF"](), "removed 2026-11-15 (lane/expdel2-23-neural): IDN_M3_ANGLE_DT_SUFFIX_OFF (old path) is gone; the default is kept; code at 0b1aca903d; restore: experiments/removed/MOJOLEARN_IDN_M3_ANGLE_DT_SUFFIX_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_M3_BWD_SKIP_DEAD_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_M3_BWD_SKIP_DEAD_OFF"](), "removed 2026-11-15 (lane/expdel2-23-neural): IDN_M3_BWD_SKIP_DEAD_OFF (old path) is gone; the default is kept; code at 8e12418241; restore: experiments/removed/MOJOLEARN_IDN_M3_BWD_SKIP_DEAD_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_M3_DECODE_WINDOW_REUSE.patch
    comptime assert not is_defined["MOJOLEARN_IDN_M3_DECODE_WINDOW_REUSE"](), "removed 2026-11-15 (lane/expdel2-23-neural): IDN_M3_DECODE_WINDOW_REUSE (arm) is gone; the default path is the only one; code at 74754fe939; restore: experiments/removed/MOJOLEARN_IDN_M3_DECODE_WINDOW_REUSE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_M3_S17_TAIL_PIPE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_M3_S17_TAIL_PIPE_OFF"](), "removed 2026-11-15 (lane/expdel2-23-neural): IDN_M3_S17_TAIL_PIPE_OFF (old path) is gone; the default is kept; code at 1629ea8265; restore: experiments/removed/MOJOLEARN_IDN_M3_S17_TAIL_PIPE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_MAMBA12_FORWARD_COPIES_OLD.patch
    comptime assert not is_defined["MOJOLEARN_IDN_MAMBA12_FORWARD_COPIES_OLD"](), "removed 2026-11-15 (lane/expdel2-23-neural): IDN_MAMBA12_FORWARD_COPIES_OLD (old path) is gone; the default is kept; code at 449d72a027; restore: experiments/removed/MOJOLEARN_IDN_MAMBA12_FORWARD_COPIES_OLD.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_MAMBA12_FORWARD_DOOR_OLD.patch
    comptime assert not is_defined["MOJOLEARN_IDN_MAMBA12_FORWARD_DOOR_OLD"](), "removed 2026-11-15 (lane/expdel2-23-neural): IDN_MAMBA12_FORWARD_DOOR_OLD (old path) is gone; the default is kept; code at ddf9b022e1; restore: experiments/removed/MOJOLEARN_IDN_MAMBA12_FORWARD_DOOR_OLD.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_MAMBA_BWD_FOLDS.patch
    comptime assert not is_defined["MOJOLEARN_IDN_MAMBA_BWD_FOLDS"](), "removed 2026-11-15 (lane/expdel2-23-neural): IDN_MAMBA_BWD_FOLDS (arm) is gone; the default path is the only one; code at 786965c6c8; restore: experiments/removed/MOJOLEARN_IDN_MAMBA_BWD_FOLDS.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_TRANSPOSE_TILE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_TRANSPOSE_TILE_OFF"](), "removed 2026-11-15 (lane/expdel2-23-neural): IDN_TRANSPOSE_TILE_OFF (old path) is gone; the default is kept; code at 34a89b9e44; restore: experiments/removed/MOJOLEARN_IDN_TRANSPOSE_TILE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_MAMBA3_ANGLE_DT_NAIVE.patch
    comptime assert not is_defined["MOJOLEARN_MAMBA3_ANGLE_DT_NAIVE"](), "removed 2026-11-15 (lane/expdel2-23-neural): MAMBA3_ANGLE_DT_NAIVE (arm) is gone; the default path is the only one; code at c48e90d779; restore: experiments/removed/MOJOLEARN_MAMBA3_ANGLE_DT_NAIVE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_MAMBA3_LEGACY_REFUSAL.patch
    comptime assert not is_defined["MOJOLEARN_MAMBA3_LEGACY_REFUSAL"](), "removed 2026-11-15 (lane/expdel2-23-neural): MAMBA3_LEGACY_REFUSAL (old path) is gone; the default is kept; code at e81ebf2572; restore: experiments/removed/MOJOLEARN_MAMBA3_LEGACY_REFUSAL.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_MAMBA3_S16_QK_NAIVE.patch
    comptime assert not is_defined["MOJOLEARN_MAMBA3_S16_QK_NAIVE"](), "removed 2026-11-15 (lane/expdel2-23-neural): MAMBA3_S16_QK_NAIVE (arm) is gone; the default path is the only one; code at 388a1c5306; restore: experiments/removed/MOJOLEARN_MAMBA3_S16_QK_NAIVE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_MAMBA3_S16_V_NAIVE.patch
    comptime assert not is_defined["MOJOLEARN_MAMBA3_S16_V_NAIVE"](), "removed 2026-11-15 (lane/expdel2-23-neural): MAMBA3_S16_V_NAIVE (arm) is gone; the default path is the only one; code at b720d9dbfb; restore: experiments/removed/MOJOLEARN_MAMBA3_S16_V_NAIVE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_MAMBA3_S17_OPERANDS_NAIVE.patch
    comptime assert not is_defined["MOJOLEARN_MAMBA3_S17_OPERANDS_NAIVE"](), "removed 2026-11-15 (lane/expdel2-23-neural): MAMBA3_S17_OPERANDS_NAIVE (arm) is gone; the default path is the only one; code at 60e95e5888; restore: experiments/removed/MOJOLEARN_MAMBA3_S17_OPERANDS_NAIVE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_MAMBA3_S17_TAIL_NAIVE.patch
    comptime assert not is_defined["MOJOLEARN_MAMBA3_S17_TAIL_NAIVE"](), "removed 2026-11-15 (lane/expdel2-23-neural): MAMBA3_S17_TAIL_NAIVE (arm) is gone; the default path is the only one; code at ccf74402cb; restore: experiments/removed/MOJOLEARN_MAMBA3_S17_TAIL_NAIVE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_MAMBA_CALL_PROGRAM.patch
    comptime assert not is_defined["MOJOLEARN_MAMBA_CALL_PROGRAM"](), "removed 2026-11-15 (lane/expdel2-23-neural): MAMBA_CALL_PROGRAM (arm) is gone; the default path is the only one; code at ea4327182b; restore: experiments/removed/MOJOLEARN_MAMBA_CALL_PROGRAM.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NN50_OWNED_RUNS.patch
    comptime assert not is_defined["MOJOLEARN_NN50_OWNED_RUNS"](), "removed 2026-11-15 (lane/expdel2-23-neural): NN50_OWNED_RUNS (arm) is gone; the default path is the only one; code at 4f9d2d5288; restore: experiments/removed/MOJOLEARN_NN50_OWNED_RUNS.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TRANSFORMER_GENERIC_PREFILL.patch
    comptime assert not is_defined["MOJOLEARN_TRANSFORMER_GENERIC_PREFILL"](), "removed 2026-11-15 (lane/expdel2-23-neural): TRANSFORMER_GENERIC_PREFILL (arm) is gone; the default path is the only one; code at 29b23a6003; restore: experiments/removed/MOJOLEARN_TRANSFORMER_GENERIC_PREFILL.patch"
    return True


comptime NEURAL_GUARDS_OK = _check_configuration()
