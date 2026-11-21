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
    # TOMBSTONE restored 2026-11-17 by lane/fb-neural: MOJOLEARN_AUTOGRAD_PREDICT is live again as the opt-in A/B arm
    # (sequence/mlp_fit.mojo: the hand mlp_predict is the default again, as at 0.8.55; the define takes the tape
    # predict mlp_predict_autograd for MLP only; the recurrent predict stays on the tape). No refusal.
    # TOMBSTONE: MOJOLEARN_AUTOGRAD_PREDICT (DROPPED-slower: tape predict 1.04-1.06x on the four MLP infer cells, NV L40S 2026-11-18) deleted 2026-11-18 by lane/fb15-ab-verdicts; code recoverable at c989eb9222.
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_PREDICT-2026-11-18.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_PREDICT"](), "removed 2026-11-18 (lane/fb15-ab-verdicts, A/B ab1 NVIDIA L40S, wheel d652a4b0a1, one run per arm): the tape MLP predict was slower on all four infer cells (mlp-clf istella 148.3 -> 154.7 ms, mlp-clf taxi 21.48 -> 22.46, mlp-reg taxi 10.49 -> 11.08, mlp-reg istella 74.72 -> 79.21; fits equal; digests equal); the hand mlp_predict is the one MLP predict; recoverable at c989eb9222"
    # TOMBSTONE: MOJOLEARN_NI12_FILL_RULE_OFF (DROPPED-slower: implicit on every layer 1.15x on cnn-clf fit, rest 0.95-1.04, NV L40S 2026-11-18) deleted 2026-11-18 by lane/fb15-ab-verdicts; code recoverable at c989eb9222.
    # Restore: git apply experiments/removed/MOJOLEARN_NI12_FILL_RULE_OFF.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_NI12_FILL_RULE_OFF"](), "removed 2026-11-18 (lane/fb15-ab-verdicts, A/B ab1 NVIDIA L40S, wheel d652a4b0a1, one run per arm): implicit conv on every layer was 1.15x slower on cnn-clf fit (189.9 -> 218.2 ms) and 0.95-1.04 elsewhere, digests equal; the per-layer fill rule (x_cnn/device.mojo ni12_implicit_layer) is the one form; recoverable at c989eb9222"
    # TOMBSTONE: MOJOLEARN_AUTOGRAD_LOSS (rnn_fit: promoted, the blocked fold replaces the serial OP_SUM) deleted 2026-11-08 by dbab55bea7; code recoverable at c3b7dc42e6^.
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_LOSS.patch
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_LOSS"](), "removed 2026-11-08 (lane/x3-mlp-rnn-predict-on-tape, dbab55bea7): rnn_fit's reported loss always uses the blocked fold (loss_value_into; the serial OP_SUM is deleted); mlp_fit keeps its loss block (already blocked, the arm added launches). Drop the define. Restore with experiments/removed/MOJOLEARN_AUTOGRAD_LOSS.patch"
    # TOMBSTONE: MOJOLEARN_AUTOGRAD_OPTIM (promoted: the executor optimizer is the trainers' one update; the hand arm deleted) deleted 2026-11-08 by 75825f479c; code recoverable at c3b7dc42e6^.
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_OPTIM.patch
    # Restored 2026-11-09 by lane/r2-restore-unmeasured-defaults: the hand OP_OPT launch is the default again and
    # -D MOJOLEARN_AUTOGRAD_OPTIM was the A/B arm (the executor optimizer); decided same speed 2026-11-17 and refused below.
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
    # MOJOLEARN_SAMBA_WEIGHT_EPOCH_GLOBAL: refusal removed 2026-11-15 (restored; owed B arm, see docs/TOMBSTONES.md Restored table)
    # Lane expdel2-04-base (2026-11-15): unmeasured arms and old paths deleted under "An unmeasured arm is deleted" (docs/plans/EXPERIMENT_DELETIONS_V2_2026-11-15.md, docs/TOMBSTONES.md).
    # MOJOLEARN_IDN_MAMBA3_FORWARD_DOOR_OLD: refusal removed 2026-11-15 (restored; owed B arm, see docs/TOMBSTONES.md Restored table)
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
    # MOJOLEARN_TRAIN_LIVE_STATUS: refusal removed 2026-11-15 (restored; owed B arm, see docs/TOMBSTONES.md Restored table)
    # Lane expdel2-08-base (2026-11-15): unmeasured arms and superseded old paths deleted, docs/plans/EXPERIMENT_DELETIONS_V2_2026-11-15.md (docs/TOMBSTONES.md).
    # MOJOLEARN_PROGRAM_RUN_ENTRY: refusal removed 2026-11-15 (restored; owed B arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_SESSION_CALL_DESCRIPTOR: refusal removed 2026-11-15 (restored; owed B arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_TRANSFORMER_DECODE_PER_TOKEN_ADMISSION: refusal removed 2026-11-15 (restored; owed B arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_TRANSFORMER_DECODE_PER_TOKEN_DOOR: refusal removed 2026-11-15 (restored; owed B arm, see docs/TOMBSTONES.md Restored table)
    # Lane expdel2-23-neural (2026-11-15): unmeasured arms and old paths deleted, docs/plans/EXPERIMENT_DELETIONS_V2_2026-11-15.md (docs/TOMBSTONES.md).
    # MOJOLEARN_AFN_MAMBA3_BWD_ARENA: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_MAMBA3_BWD_CHUNK: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_SAMBA_ALL: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_GATHER_EPOCH_OFF.patch
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_GATHER_EPOCH_OFF"](), "removed 2026-11-15 (lane/expdel2-23-neural): AUTOGRAD_GATHER_EPOCH_OFF (old path) is gone; the default is kept; code at f96a014a39; restore: experiments/removed/MOJOLEARN_AUTOGRAD_GATHER_EPOCH_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_BYTE_LM_GENERIC_OPS.fb12-loser-arms.patch
    comptime assert not is_defined["MOJOLEARN_BYTE_LM_GENERIC_OPS"](), "removed 2026-11-17 (lane/fb12-loser-arms): BYTE_LM_GENERIC_OPS (decided B arm, generic OPS_EMBEDDING capture and no next-norm fusion; L40S ba08dac006 B 1.36x slower) is gone; the default path is the only one; code recoverable at 2203db7094; restore: experiments/removed/MOJOLEARN_BYTE_LM_GENERIC_OPS.fb12-loser-arms.patch"
    # MOJOLEARN_CNN_GENERIC_OPS: refusal removed 2026-11-15 (restored; owed B arm, see docs/TOMBSTONES.md Restored table)
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
    # MOJOLEARN_IDN_MAMBA12_FORWARD_COPIES_OLD: refusal removed 2026-11-15 (restored; owed B arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_IDN_MAMBA12_FORWARD_DOOR_OLD: refusal removed 2026-11-15 (restored; owed B arm, see docs/TOMBSTONES.md Restored table)
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
    # MOJOLEARN_MAMBA_CALL_PROGRAM: refusal removed 2026-11-15 (restored; owed B arm, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_NN50_OWNED_RUNS.patch
    comptime assert not is_defined["MOJOLEARN_NN50_OWNED_RUNS"](), "removed 2026-11-15 (lane/expdel2-23-neural): NN50_OWNED_RUNS (arm) is gone; the default path is the only one; code at 4f9d2d5288; restore: experiments/removed/MOJOLEARN_NN50_OWNED_RUNS.patch"
    # MOJOLEARN_TRANSFORMER_GENERIC_PREFILL: refusal removed 2026-11-15 (restored; owed B arm, see docs/TOMBSTONES.md Restored table)
    # Lane expdel2-26-neural (2026-11-15): unmeasured arms and superseded old paths deleted, docs/plans/EXPERIMENT_DELETIONS_V2_2026-11-15.md (docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_M1_SCAN.patch
    comptime assert not is_defined["MOJOLEARN_IDN_M1_SCAN"](), "removed 2026-11-15 (lane/expdel2-26-neural): IDN_M1_SCAN (unmeasured opt-in arms) is gone; the default is kept; code at 5fc464dce4; restore: experiments/removed/MOJOLEARN_IDN_M1_SCAN.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_M2_CB_LOWER_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_M2_CB_LOWER_OFF"](), "removed 2026-11-15 (lane/expdel2-26-neural): IDN_M2_CB_LOWER_OFF (unmeasured old path) is gone; the default is kept; code at 5fc464dce4; restore: experiments/removed/MOJOLEARN_IDN_M2_CB_LOWER_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_M3_ANGLE_BLOCK_128.patch
    comptime assert not is_defined["MOJOLEARN_IDN_M3_ANGLE_BLOCK_128"](), "removed 2026-11-15 (lane/expdel2-26-neural): IDN_M3_ANGLE_BLOCK_128 (unmeasured opt-in arm) is gone; the default is kept; code at 5fc464dce4; restore: experiments/removed/MOJOLEARN_IDN_M3_ANGLE_BLOCK_128.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_M3_ANGLE_BLOCK_32.patch
    comptime assert not is_defined["MOJOLEARN_IDN_M3_ANGLE_BLOCK_32"](), "removed 2026-11-15 (lane/expdel2-26-neural): IDN_M3_ANGLE_BLOCK_32 (unmeasured opt-in arm) is gone; the default is kept; code at 5fc464dce4; restore: experiments/removed/MOJOLEARN_IDN_M3_ANGLE_BLOCK_32.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_M3_ANGLE_PARALLEL.patch
    comptime assert not is_defined["MOJOLEARN_IDN_M3_ANGLE_PARALLEL"](), "removed 2026-11-15 (lane/expdel2-26-neural): IDN_M3_ANGLE_PARALLEL (unmeasured opt-in arm) is gone; the default is kept; code at 5fc464dce4; restore: experiments/removed/MOJOLEARN_IDN_M3_ANGLE_PARALLEL.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_M3_YINTRA_RESOURCE_TILE.patch
    comptime assert not is_defined["MOJOLEARN_IDN_M3_YINTRA_RESOURCE_TILE"](), "removed 2026-11-15 (lane/expdel2-26-neural): IDN_M3_YINTRA_RESOURCE_TILE (unmeasured opt-in arm) is gone; the default is kept; code at 5fc464dce4; restore: experiments/removed/MOJOLEARN_IDN_M3_YINTRA_RESOURCE_TILE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_FILL_BLOCKS.patch
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_FILL_BLOCKS"](), "removed 2026-11-15 (lane/expdel2-26-neural): IDN_NEURAL_FILL_BLOCKS (unmeasured opt-in knob) is gone; the default is kept; code at 5fc464dce4; restore: experiments/removed/MOJOLEARN_IDN_NEURAL_FILL_BLOCKS.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_FOLD_SHARED.patch
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_FOLD_SHARED"](), "removed 2026-11-15 (lane/expdel2-26-neural): IDN_NEURAL_FOLD_SHARED (unmeasured opt-in arm) is gone; the default is kept; code at 5fc464dce4; restore: experiments/removed/MOJOLEARN_IDN_NEURAL_FOLD_SHARED.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_GEMM_ARM.patch
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_GEMM_ARM"](), "removed 2026-11-15 (lane/expdel2-26-neural): IDN_NEURAL_GEMM_ARM (unmeasured opt-in knob) is gone; the default is kept; code at 5fc464dce4; restore: experiments/removed/MOJOLEARN_IDN_NEURAL_GEMM_ARM.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_GEMM_CONTROL.patch
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_GEMM_CONTROL"](), "removed 2026-11-15 (lane/expdel2-26-neural): IDN_NEURAL_GEMM_CONTROL (unmeasured opt-in arm) is gone; the default is kept; code at 5fc464dce4; restore: experiments/removed/MOJOLEARN_IDN_NEURAL_GEMM_CONTROL.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_PAIR_CONTROL.patch
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_PAIR_CONTROL"](), "removed 2026-11-15 (lane/expdel2-26-neural): IDN_NEURAL_PAIR_CONTROL (unmeasured opt-in arm) is gone; the default is kept; code at 5fc464dce4; restore: experiments/removed/MOJOLEARN_IDN_NEURAL_PAIR_CONTROL.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_RETAINED_FLOATS.patch
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_RETAINED_FLOATS"](), "removed 2026-11-15 (lane/expdel2-26-neural): IDN_NEURAL_RETAINED_FLOATS (unmeasured opt-in knob) is gone; the default is kept; code at 5fc464dce4; restore: experiments/removed/MOJOLEARN_IDN_NEURAL_RETAINED_FLOATS.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_STAGE_DEPTH.patch
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_STAGE_DEPTH"](), "removed 2026-11-15 (lane/expdel2-26-neural): IDN_NEURAL_STAGE_DEPTH (unmeasured opt-in knob) is gone; the default is kept; code at 5fc464dce4; restore: experiments/removed/MOJOLEARN_IDN_NEURAL_STAGE_DEPTH.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_STAGE_PAD.patch
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_STAGE_PAD"](), "removed 2026-11-15 (lane/expdel2-26-neural): IDN_NEURAL_STAGE_PAD (unmeasured opt-in knob) is gone; the default is kept; code at 5fc464dce4; restore: experiments/removed/MOJOLEARN_IDN_NEURAL_STAGE_PAD.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_STAGE_SWIZZLE.patch
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_STAGE_SWIZZLE"](), "removed 2026-11-15 (lane/expdel2-26-neural): IDN_NEURAL_STAGE_SWIZZLE (unmeasured opt-in arm) is gone; the default is kept; code at 5fc464dce4; restore: experiments/removed/MOJOLEARN_IDN_NEURAL_STAGE_SWIZZLE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_STREAM_GROUP.patch
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_STREAM_GROUP"](), "removed 2026-11-15 (lane/expdel2-26-neural): IDN_NEURAL_STREAM_GROUP (unmeasured opt-in knob) is gone; the default is kept; code at 5fc464dce4; restore: experiments/removed/MOJOLEARN_IDN_NEURAL_STREAM_GROUP.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_MAMBA3_BALANCED_INCREMENT_TILE.patch
    comptime assert not is_defined["MOJOLEARN_MAMBA3_BALANCED_INCREMENT_TILE"](), "removed 2026-11-15 (lane/expdel2-26-neural): MAMBA3_BALANCED_INCREMENT_TILE (unmeasured opt-in arm) is gone; the default is kept; code at 5fc464dce4; restore: experiments/removed/MOJOLEARN_MAMBA3_BALANCED_INCREMENT_TILE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_MAMBA3_LEGACY_ANGLE_INCREMENT.patch
    comptime assert not is_defined["MOJOLEARN_MAMBA3_LEGACY_ANGLE_INCREMENT"](), "removed 2026-11-15 (lane/expdel2-26-neural): MAMBA3_LEGACY_ANGLE_INCREMENT (unmeasured old path) is gone; the default is kept; code at 5fc464dce4; restore: experiments/removed/MOJOLEARN_MAMBA3_LEGACY_ANGLE_INCREMENT.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_MAMBA3_LEGACY_INCREMENT_TILE.patch
    comptime assert not is_defined["MOJOLEARN_MAMBA3_LEGACY_INCREMENT_TILE"](), "removed 2026-11-15 (lane/expdel2-26-neural): MAMBA3_LEGACY_INCREMENT_TILE (unmeasured old path) is gone; the default is kept; code at 5fc464dce4; restore: experiments/removed/MOJOLEARN_MAMBA3_LEGACY_INCREMENT_TILE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_MAMBA3_LEGACY_INCREMENT_V.patch
    comptime assert not is_defined["MOJOLEARN_MAMBA3_LEGACY_INCREMENT_V"](), "removed 2026-11-15 (lane/expdel2-26-neural): MAMBA3_LEGACY_INCREMENT_V (unmeasured old path) is gone; the default is kept; code at 5fc464dce4; restore: experiments/removed/MOJOLEARN_MAMBA3_LEGACY_INCREMENT_V.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_MAMBA3_LEGACY_QKS.patch
    comptime assert not is_defined["MOJOLEARN_MAMBA3_LEGACY_QKS"](), "removed 2026-11-15 (lane/expdel2-26-neural): MAMBA3_LEGACY_QKS (unmeasured old path) is gone; the default is kept; code at 5fc464dce4; restore: experiments/removed/MOJOLEARN_MAMBA3_LEGACY_QKS.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_MAMBA3_LEGACY_STATEPASS.patch
    comptime assert not is_defined["MOJOLEARN_MAMBA3_LEGACY_STATEPASS"](), "removed 2026-11-15 (lane/expdel2-26-neural): MAMBA3_LEGACY_STATEPASS (unmeasured old path) is gone; the default is kept; code at 5fc464dce4; restore: experiments/removed/MOJOLEARN_MAMBA3_LEGACY_STATEPASS.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_MAMBA3_LEGACY_YINTRA.patch
    comptime assert not is_defined["MOJOLEARN_MAMBA3_LEGACY_YINTRA"](), "removed 2026-11-15 (lane/expdel2-26-neural): MAMBA3_LEGACY_YINTRA (unmeasured old path) is gone; the default is kept; code at 5fc464dce4; restore: experiments/removed/MOJOLEARN_MAMBA3_LEGACY_YINTRA.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_MAMBA3_LEGACY_YSTATE.patch
    comptime assert not is_defined["MOJOLEARN_MAMBA3_LEGACY_YSTATE"](), "removed 2026-11-15 (lane/expdel2-26-neural): MAMBA3_LEGACY_YSTATE (unmeasured old path) is gone; the default is kept; code at 5fc464dce4; restore: experiments/removed/MOJOLEARN_MAMBA3_LEGACY_YSTATE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_MAMBA3_TILED_INCREMENT.patch
    comptime assert not is_defined["MOJOLEARN_MAMBA3_TILED_INCREMENT"](), "removed 2026-11-15 (lane/expdel2-26-neural): MAMBA3_TILED_INCREMENT (unmeasured opt-in arm) is gone; the default is kept; code at 5fc464dce4; restore: experiments/removed/MOJOLEARN_MAMBA3_TILED_INCREMENT.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_MAMBA3_TILED_YINTRA.patch
    comptime assert not is_defined["MOJOLEARN_MAMBA3_TILED_YINTRA"](), "removed 2026-11-15 (lane/expdel2-26-neural): MAMBA3_TILED_YINTRA (unmeasured opt-in arm) is gone; the default is kept; code at 5fc464dce4; restore: experiments/removed/MOJOLEARN_MAMBA3_TILED_YINTRA.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NN35_CONV_PAIR.patch
    comptime assert not is_defined["MOJOLEARN_NN35_CONV_PAIR"](), "removed 2026-11-15 (lane/expdel2-26-neural): NN35_CONV_PAIR (unmeasured opt-in arm) is gone; the default is kept; code at 5fc464dce4; restore: experiments/removed/MOJOLEARN_NN35_CONV_PAIR.patch"
    # Lane expdel2-25-neural (2026-11-15): unmeasured arms and superseded old paths deleted, docs/plans/EXPERIMENT_DELETIONS_V2_2026-11-15.md (docs/TOMBSTONES.md).
    # MOJOLEARN_AFN_GEMM2_ALL: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_GEMM2_BIGTILE: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_GEMM2_DBUF: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_GEMM2_DIRECT_B: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_GEMM2_SWIZZLE: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_GEMM2_SWZ_G: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_GEMM_ALL: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_GEMM_BF16_MMA: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_GEMM_CORES: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_GEMM_EPILOGUE: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_GEMM_INT8_MMA: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_GEMM_KB: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_GEMM_SIMDGROUP: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_GEMM_SPLITK: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_GEMM_TILESHAPE: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_EMBEDDING_OFF.patch
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_EMBEDDING_OFF"](), "removed 2026-11-15 (lane/expdel2-25-neural): AUTOGRAD_EMBEDDING_OFF (old path beside the default) is gone; the default is kept; code at 5ee3d0ca88; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_EMB_DEVICE_SCRATCH_OFF.patch
    comptime assert not is_defined["MOJOLEARN_EMB_DEVICE_SCRATCH_OFF"](), "removed 2026-11-15 (lane/expdel2-25-neural): EMB_DEVICE_SCRATCH_OFF (old path beside the default) is gone; the default is kept; code at 9b6b13052d; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_EMB_RESIDENT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_EMB_RESIDENT_OFF"](), "removed 2026-11-15 (lane/expdel2-25-neural): EMB_RESIDENT_OFF (old path beside the default) is gone; the default is kept; code at 63a7445f96; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_GEMM_FAST_RAGGED_K_OFF.patch
    comptime assert not is_defined["MOJOLEARN_GEMM_FAST_RAGGED_K_OFF"](), "removed 2026-11-15 (lane/expdel2-25-neural): GEMM_FAST_RAGGED_K_OFF (old path beside the default) is gone; the default is kept; code at 02da0a0217; see docs/TOMBSTONES.md"
    # MOJOLEARN_IDN_GEMM_INT_SLICES: refusal removed 2026-11-15 (restored; RUN #2 of docs/plans/DELETED_ARMS_WORTH_RUNNING_2026-11-15.md, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_CHAINS.patch
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_CHAINS"](), "removed 2026-11-15 (lane/expdel2-25-neural): IDN_NEURAL_CHAINS (unmeasured opt-in arm) is gone; the default is kept; code at a9109164e5; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_GEMM_EPILOGUE.patch
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_GEMM_EPILOGUE"](), "removed 2026-11-15 (lane/expdel2-25-neural): IDN_NEURAL_GEMM_EPILOGUE (unmeasured opt-in arm) is gone; the default is kept; code at d95e19da90; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE.patch
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE"](), "removed 2026-11-15 (lane/expdel2-25-neural): IDN_NEURAL_GEMM_SCHEDULE (unmeasured opt-in arm) is gone; the default is kept; code at 9813509a01; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE_ROLES.patch
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_GEMM_SCHEDULE_ROLES"](), "removed 2026-11-15 (lane/expdel2-25-neural): IDN_NEURAL_GEMM_SCHEDULE_ROLES (unmeasured opt-in arm) is gone; the default is kept; code at 7737754a09; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_NN07.patch
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_NN07"](), "removed 2026-11-15 (lane/expdel2-25-neural): IDN_NEURAL_NN07 (unmeasured opt-in arm) is gone; the default is kept; code at dd47bfb212; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_NN12.patch
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_NN12"](), "removed 2026-11-15 (lane/expdel2-25-neural): IDN_NEURAL_NN12 (unmeasured opt-in arm) is gone; the default is kept; code at a9cf951197; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_NEURAL_NN16.patch
    comptime assert not is_defined["MOJOLEARN_IDN_NEURAL_NN16"](), "removed 2026-11-15 (lane/expdel2-25-neural): IDN_NEURAL_NN16 (unmeasured opt-in arm) is gone; the default is kept; code at 1362c86435; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_INT_SLICES_FORCE_FLAT.patch
    comptime assert not is_defined["MOJOLEARN_INT_SLICES_FORCE_FLAT"](), "removed 2026-11-15 (lane/expdel2-25-neural): INT_SLICES_FORCE_FLAT (unmeasured opt-in arm) is gone; the default is kept; code at e62e7376e0; see docs/TOMBSTONES.md"
    # Restore: git apply experiments/removed/MOJOLEARN_NI07_GROUPED_PROJECTIONS.patch
    comptime assert not is_defined["MOJOLEARN_NI07_GROUPED_PROJECTIONS"](), "removed 2026-11-15 (lane/expdel2-25-neural): NI07_GROUPED_PROJECTIONS (unmeasured opt-in arm) is gone; the default is kept; code at e862759b5c; see docs/TOMBSTONES.md"
    # Lane expdel2-24-neural (2026-11-15): unmeasured arms and old paths deleted under "An unmeasured arm is deleted" (docs/plans/EXPERIMENT_DELETIONS_V2_2026-11-15.md, docs/TOMBSTONES.md).
    # MOJOLEARN_AFN_LMGRAD_ALL: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_LM_WGRAD_SPLIT: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN26_LM_WGRAD_SPLIT: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_LM_BWD_EPILOGUE: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN26_LM_BWD_EPILOGUE: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_LM_BWD_NORM1_RESID: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN26_LM_BWD_NORM1_RESID: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN26_LM_WGRAD_CAP8: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN26_LM_WGRAD_MIN128: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_CURVE_WORD_OFF.patch
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_CURVE_WORD_OFF"](), "removed 2026-11-15 (lane/expdel2-24-neural): AUTOGRAD_CURVE_WORD_OFF (old path, Neural (autograd model_fit curve word); no A/B recorded) is gone; the default is kept; code recoverable at fdb70ce42d; restore: experiments/removed/MOJOLEARN_AUTOGRAD_CURVE_WORD_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_FOLD_V2_OFF.patch
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_FOLD_V2_OFF"](), "removed 2026-11-15 (lane/expdel2-24-neural): AUTOGRAD_FOLD_V2_OFF (old path, Neural (autograd blocked fold geometry); no A/B recorded) is gone; the default is kept; code recoverable at 01ab3e0d37; restore: experiments/removed/MOJOLEARN_AUTOGRAD_FOLD_V2_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_GRAD_ALIAS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_GRAD_ALIAS_OFF"](), "removed 2026-11-15 (lane/expdel2-24-neural): AUTOGRAD_GRAD_ALIAS_OFF (old path, Neural (autograd gradient borrowing); no A/B recorded) is gone; the default is kept; code recoverable at 4b7b69e04f; restore: experiments/removed/MOJOLEARN_AUTOGRAD_GRAD_ALIAS_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_HAS_INFERENCE.patch
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_HAS_INFERENCE"](), "removed 2026-11-15 (lane/expdel2-24-neural): AUTOGRAD_HAS_INFERENCE (opt-in arm, Neural (autograd model_predict inference mode); no A/B recorded) is gone; the default path is the only one; code recoverable at fd8abcf837; restore: experiments/removed/MOJOLEARN_AUTOGRAD_HAS_INFERENCE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_LOSS_REDUCED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_LOSS_REDUCED_OFF"](), "removed 2026-11-15 (lane/expdel2-24-neural): AUTOGRAD_LOSS_REDUCED_OFF (old path, Neural (autograd elementwise losses); no A/B recorded) is gone; the default is kept; code recoverable at 56d165c105; restore: experiments/removed/MOJOLEARN_AUTOGRAD_LOSS_REDUCED_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_MAMBA_EXPORT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_MAMBA_EXPORT_OFF"](), "removed 2026-11-15 (lane/expdel2-24-neural): AUTOGRAD_MAMBA_EXPORT_OFF (old path, Neural (autograd Mamba stack export); no A/B recorded) is gone; the default is kept; code recoverable at e9b44b3e5b; restore: experiments/removed/MOJOLEARN_AUTOGRAD_MAMBA_EXPORT_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_MODEL_UNFUSED.patch
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_MODEL_UNFUSED"](), "removed 2026-11-15 (lane/expdel2-24-neural): AUTOGRAD_MODEL_UNFUSED (opt-in arm, Neural (autograd model forward); no A/B recorded) is gone; the default path is the only one; code recoverable at 5cc4c35385; restore: experiments/removed/MOJOLEARN_AUTOGRAD_MODEL_UNFUSED.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_OPT_CLIP_FUSE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_OPT_CLIP_FUSE_OFF"](), "removed 2026-11-15 (lane/expdel2-24-neural): AUTOGRAD_OPT_CLIP_FUSE_OFF (old path, Neural (autograd optimizer clip); no A/B recorded) is gone; the default is kept; code recoverable at e770fd25b9; restore: experiments/removed/MOJOLEARN_AUTOGRAD_OPT_CLIP_FUSE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_OPT_FUSE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_OPT_FUSE_OFF"](), "removed 2026-11-15 (lane/expdel2-24-neural): AUTOGRAD_OPT_FUSE_OFF (old path, Neural (autograd optimizer gradient finish); no A/B recorded) is gone; the default is kept; code recoverable at 8535432c53; restore: experiments/removed/MOJOLEARN_AUTOGRAD_OPT_FUSE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_PAIR_OFF.patch
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_PAIR_OFF"](), "removed 2026-11-15 (lane/expdel2-24-neural): AUTOGRAD_PAIR_OFF (old path, Neural (autograd linear backward launches); no A/B recorded) is gone; the default is kept; code recoverable at 24924c8fc2; restore: experiments/removed/MOJOLEARN_AUTOGRAD_PAIR_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_PASS2_FOLD_OFF.patch
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_PASS2_FOLD_OFF"](), "removed 2026-11-15 (lane/expdel2-24-neural): AUTOGRAD_PASS2_FOLD_OFF (old path, Neural (autograd bias gradient fold); no A/B recorded) is gone; the default is kept; code recoverable at 094332d07e; restore: experiments/removed/MOJOLEARN_AUTOGRAD_PASS2_FOLD_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_PASS3_DB_OFF.patch
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_PASS3_DB_OFF"](), "removed 2026-11-15 (lane/expdel2-24-neural): AUTOGRAD_PASS3_DB_OFF (old path, Neural (autograd bias gradient in dW launch); no A/B recorded) is gone; the default is kept; code recoverable at 0399c92e07; restore: experiments/removed/MOJOLEARN_AUTOGRAD_PASS3_DB_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_REFUSE_DEFER_OFF.patch
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_REFUSE_DEFER_OFF"](), "removed 2026-11-15 (lane/expdel2-24-neural): AUTOGRAD_REFUSE_DEFER_OFF (old path, Neural (autograd LM and decoder-layer refusal scans); no A/B recorded) is gone; the default is kept; code recoverable at 2266129859; restore: experiments/removed/MOJOLEARN_AUTOGRAD_REFUSE_DEFER_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_TRANSPOSE_VIEW_OFF.patch
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_TRANSPOSE_VIEW_OFF"](), "removed 2026-11-15 (lane/expdel2-24-neural): AUTOGRAD_TRANSPOSE_VIEW_OFF (old path, Neural (autograd transpose); no A/B recorded) is gone; the default is kept; code recoverable at 5b147f4046; restore: experiments/removed/MOJOLEARN_AUTOGRAD_TRANSPOSE_VIEW_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_BWD_NORM_LEGACY_SHAPE_RULE.patch
    comptime assert not is_defined["MOJOLEARN_BWD_NORM_LEGACY_SHAPE_RULE"](), "removed 2026-11-15 (lane/expdel2-24-neural): BWD_NORM_LEGACY_SHAPE_RULE (old path, Neural (RMSNorm backward fused-row dispatch); no A/B recorded) is gone; the default is kept; code recoverable at e7e660e1bc; restore: experiments/removed/MOJOLEARN_BWD_NORM_LEGACY_SHAPE_RULE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_AUC_BLOCK_SORT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_AUC_BLOCK_SORT_OFF"](), "removed 2026-11-15 (lane/expdel2-24-neural): IDN_AUC_BLOCK_SORT_OFF (old path, Neural (autograd roc_auc bitonic sort); no A/B recorded) is gone; the default is kept; code recoverable at fe0b691d9b; restore: experiments/removed/MOJOLEARN_IDN_AUC_BLOCK_SORT_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_BWD_ENTRY_ALIAS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_BWD_ENTRY_ALIAS_OFF"](), "removed 2026-11-15 (lane/expdel2-24-neural): IDN_BWD_ENTRY_ALIAS_OFF (old path, Neural (decoder-layer backward entry copy); no A/B recorded) is gone; the default is kept; code recoverable at 3d189418c5; restore: experiments/removed/MOJOLEARN_IDN_BWD_ENTRY_ALIAS_OFF.patch"
    # MOJOLEARN_IDN_MAMBA_BACKWARD_GENERIC: refusal removed 2026-11-15 (restored; owed B arm, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_IDN_MLP_GEMM_LEAF128_OFF: refusal removed 2026-11-15 (restored; owed B arm, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ROW_COOP_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ROW_COOP_OFF"](), "removed 2026-11-15 (lane/expdel2-24-neural): IDN_ROW_COOP_OFF (old path, Neural (autograd softmax-family rows); no A/B recorded) is gone; the default is kept; code recoverable at 5d81d356e0; restore: experiments/removed/MOJOLEARN_IDN_ROW_COOP_OFF.patch"
    # MOJOLEARN_IDN_SEQ_TILED_PREDICT: refusal removed 2026-11-15 (restored; RUN #4 of docs/plans/DELETED_ARMS_WORTH_RUNNING_2026-11-15.md, see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_SMALL_MLP_GENERIC_OPS: refusal removed 2026-11-15 (restored; owed B arm, see docs/TOMBSTONES.md Restored table)
    # ---- end lane expdel2-24-neural
    # Lane expdel2-30-neural (2026-11-15): unmeasured arms and superseded old paths deleted, docs/plans/EXPERIMENT_DELETIONS_V2_2026-11-15.md (docs/TOMBSTONES.md).
    # MOJOLEARN_AFN26_LOSS_BLOCK128: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN26_LOSS_BLOCK512: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN26_LOSS_FUSED: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN26_OPT_BLOCK128: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN26_OPT_BLOCK512: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN26_OPT_CLIP_FUSE: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN26_OPT_MULTITENSOR: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_LOSS_FUSED: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_OPT_CLIP_FUSE: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_OPT_MULTITENSOR: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_BPE_ALL: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_BPE_ENCODE_DEVICE: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_BPE_GROUP_FILTER: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_BPE_LIVEBUF: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_BPE_MERGE_BATCH: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_BPE_TRAIN_DEVICE: refusal removed 2026-11-15 (restored by lane/restore-apple-fast-classical: Apple fast paused, not abandoned; opt-in, default off; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_AF_RESIDENT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_AF_RESIDENT_OFF"](), "removed 2026-11-15 (lane/expdel2-30-neural): IDN_AF_RESIDENT_OFF (unmeasured old path) is gone; the default is kept; code at e39daf79ad; restore: experiments/removed/MOJOLEARN_IDN_AF_RESIDENT_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_OPT_PIPE_DOWN_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_OPT_PIPE_DOWN_OFF"](), "removed 2026-11-15 (lane/expdel2-30-neural): IDN_OPT_PIPE_DOWN_OFF (unmeasured old path) is gone; the default is kept; code at e39daf79ad; restore: experiments/removed/MOJOLEARN_IDN_OPT_PIPE_DOWN_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_OPT_ZERO_OPEN_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_OPT_ZERO_OPEN_OFF"](), "removed 2026-11-15 (lane/expdel2-30-neural): IDN_OPT_ZERO_OPEN_OFF (unmeasured old path) is gone; the default is kept; code at e39daf79ad; restore: experiments/removed/MOJOLEARN_IDN_OPT_ZERO_OPEN_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_PROPHET_COOP_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_PROPHET_COOP_OFF"](), "removed 2026-11-15 (lane/expdel2-30-neural): IDN_PROPHET_COOP_OFF (unmeasured old path) is gone; the default is kept; code at e39daf79ad; restore: experiments/removed/MOJOLEARN_IDN_PROPHET_COOP_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_OPT_FAST_STREAM_OFF.patch
    comptime assert not is_defined["MOJOLEARN_OPT_FAST_STREAM_OFF"](), "removed 2026-11-15 (lane/expdel2-30-neural): OPT_FAST_STREAM_OFF (unmeasured old path) is gone; the default is kept; code at e39daf79ad; restore: experiments/removed/MOJOLEARN_OPT_FAST_STREAM_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_OPT_PIPE_DOWN_OFF.patch
    comptime assert not is_defined["MOJOLEARN_OPT_PIPE_DOWN_OFF"](), "removed 2026-11-15 (lane/expdel2-30-neural): OPT_PIPE_DOWN_OFF (unmeasured old path) is gone; the default is kept; code at e39daf79ad; restore: experiments/removed/MOJOLEARN_OPT_PIPE_DOWN_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_OPT_RAW_UP_OFF.patch
    comptime assert not is_defined["MOJOLEARN_OPT_RAW_UP_OFF"](), "removed 2026-11-15 (lane/expdel2-30-neural): OPT_RAW_UP_OFF (unmeasured old path) is gone; the default is kept; code at e39daf79ad; restore: experiments/removed/MOJOLEARN_OPT_RAW_UP_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_OPT_ZERO_OPEN_OFF.patch
    comptime assert not is_defined["MOJOLEARN_OPT_ZERO_OPEN_OFF"](), "removed 2026-11-15 (lane/expdel2-30-neural): OPT_ZERO_OPEN_OFF (unmeasured old path) is gone; the default is kept; code at e39daf79ad; restore: experiments/removed/MOJOLEARN_OPT_ZERO_OPEN_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_PROPHET_COOP_OFF.patch
    comptime assert not is_defined["MOJOLEARN_PROPHET_COOP_OFF"](), "removed 2026-11-15 (lane/expdel2-30-neural): PROPHET_COOP_OFF (unmeasured old path) is gone; the default is kept; code at e39daf79ad; restore: experiments/removed/MOJOLEARN_PROPHET_COOP_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SEQ_RNN_SCAN.patch
    comptime assert not is_defined["MOJOLEARN_SEQ_RNN_SCAN"](), "removed 2026-11-15 (lane/expdel2-30-neural): SEQ_RNN_SCAN (unmeasured force arm) is gone; the default is kept; code at e39daf79ad; restore: experiments/removed/MOJOLEARN_SEQ_RNN_SCAN.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SEQ_RNN_SCAN_APPLE.patch
    comptime assert not is_defined["MOJOLEARN_SEQ_RNN_SCAN_APPLE"](), "removed 2026-11-15 (lane/expdel2-30-neural): SEQ_RNN_SCAN_APPLE (unmeasured opt-in arm) is gone; the default is kept; code at e39daf79ad; restore: experiments/removed/MOJOLEARN_SEQ_RNN_SCAN_APPLE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SEQ_RNN_SCAN_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SEQ_RNN_SCAN_OFF"](), "removed 2026-11-15 (lane/expdel2-30-neural): SEQ_RNN_SCAN_OFF (unmeasured force arm) is gone; the default is kept; code at e39daf79ad; restore: experiments/removed/MOJOLEARN_SEQ_RNN_SCAN_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SEQ_THETA_REG_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SEQ_THETA_REG_OFF"](), "removed 2026-11-15 (lane/expdel2-30-neural): SEQ_THETA_REG_OFF (unmeasured old path) is gone; the default is kept; code at e39daf79ad; restore: experiments/removed/MOJOLEARN_SEQ_THETA_REG_OFF.patch"
    # Lane expdel2-27-neural (2026-11-15): unmeasured arms and old paths deleted under "An unmeasured arm is deleted" (docs/plans/EXPERIMENT_DELETIONS_V2_2026-11-15.md, docs/TOMBSTONES.md).
    # MOJOLEARN_AFN_EPI_ALL: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_MAMBA1_CHUNKSCAN: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_MAMBA1_FUSE_IN: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_MAMBA2_SSD_MMA: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_MAMBA3_SISO_FUSED: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_MAMBA_ALL: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_MAMBA_ARENA: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_MAMBA_DEVICE_REFUSAL: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_MAMBA_PROJ_EPILOGUE: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_MAMBA_PROJ_SPLITK: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ACT_RETAIN.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ACT_RETAIN"](), "removed 2026-11-15 (lane/expdel2-27-neural): IDN_ACT_RETAIN (the =3 NI48 forward-tape arm; the default 2 stays (EXP:2365)) is gone; the default stays; code recoverable at 2c7917382c; restore: experiments/removed/MOJOLEARN_IDN_ACT_RETAIN.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ACT_RETAIN_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ACT_RETAIN_OFF"](), "removed 2026-11-15 (lane/expdel2-27-neural): IDN_ACT_RETAIN_OFF (old replay path beside the retained-activation default (EXP:2365)) is gone; the default stays; code recoverable at 85d71d91bc; restore: experiments/removed/MOJOLEARN_IDN_ACT_RETAIN_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_M1_PERSISTENT_SCAN_CH.patch
    comptime assert not is_defined["MOJOLEARN_IDN_M1_PERSISTENT_SCAN_CH"](), "removed 2026-11-15 (lane/expdel2-27-neural): IDN_M1_PERSISTENT_SCAN_CH (unmeasured channels-per-block sweep; 4 stays) is gone; the default stays; code recoverable at b7061d02fc; restore: experiments/removed/MOJOLEARN_IDN_M1_PERSISTENT_SCAN_CH.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_M1_PERSISTENT_SCAN_TOKENS.patch
    comptime assert not is_defined["MOJOLEARN_IDN_M1_PERSISTENT_SCAN_TOKENS"](), "removed 2026-11-15 (lane/expdel2-27-neural): IDN_M1_PERSISTENT_SCAN_TOKENS (unmeasured tokens-per-chunk sweep; 32 stays) is gone; the default stays; code recoverable at ba22935cb9; restore: experiments/removed/MOJOLEARN_IDN_M1_PERSISTENT_SCAN_TOKENS.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_M2_CS_PT.patch
    comptime assert not is_defined["MOJOLEARN_IDN_M2_CS_PT"](), "removed 2026-11-15 (lane/expdel2-27-neural): IDN_M2_CS_PT (unmeasured C_state tile arm =16; 8 stays) is gone; the default stays; code recoverable at d51cd251ff; restore: experiments/removed/MOJOLEARN_IDN_M2_CS_PT.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_M2_RETAIN_GL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_M2_RETAIN_GL_OFF"](), "removed 2026-11-15 (lane/expdel2-27-neural): IDN_M2_RETAIN_GL_OFF (old recompute path beside the promoted retained G*L default (EXP:2013)) is gone; the default stays; code recoverable at a93584a744; restore: experiments/removed/MOJOLEARN_IDN_M2_RETAIN_GL_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_M2_SSD_TILES_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_M2_SSD_TILES_OFF"](), "removed 2026-11-15 (lane/expdel2-27-neural): IDN_M2_SSD_TILES_OFF (unmeasured old path beside the SSD tile default) is gone; the default stays; code recoverable at 9bc4a91aba; restore: experiments/removed/MOJOLEARN_IDN_M2_SSD_TILES_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_M2_YD_ROWS.patch
    comptime assert not is_defined["MOJOLEARN_IDN_M2_YD_ROWS"](), "removed 2026-11-15 (lane/expdel2-27-neural): IDN_M2_YD_ROWS (the =4 old tile beside the promoted 8-row default (EXP:2012)) is gone; the default stays; code recoverable at 0de19ddf45; restore: experiments/removed/MOJOLEARN_IDN_M2_YD_ROWS.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_M2_YOFF_EXP.patch
    comptime assert not is_defined["MOJOLEARN_IDN_M2_YOFF_EXP"](), "removed 2026-11-15 (lane/expdel2-27-neural): IDN_M2_YOFF_EXP (unmeasured Y_off exp arms 1 (NN36) and 2 (NI39); arm 0 stays) is gone; the default stays; code recoverable at 4e23c6d887; restore: experiments/removed/MOJOLEARN_IDN_M2_YOFF_EXP.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_MAMBA_ARENA_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_MAMBA_ARENA_OFF"](), "removed 2026-11-15 (lane/expdel2-27-neural): IDN_MAMBA_ARENA_OFF (unmeasured old path beside the IDENTICAL arena default) is gone; the default stays; code recoverable at e437e599e6; restore: experiments/removed/MOJOLEARN_IDN_MAMBA_ARENA_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_MAMBA_GEMM_WS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_MAMBA_GEMM_WS_OFF"](), "removed 2026-11-15 (lane/expdel2-27-neural): IDN_MAMBA_GEMM_WS_OFF (unmeasured old path beside the retained GEMM workspace default) is gone; the default stays; code recoverable at 23341493ec; restore: experiments/removed/MOJOLEARN_IDN_MAMBA_GEMM_WS_OFF.patch"
    # MOJOLEARN_IDN_SSD_OCCUPANCY: refusal removed 2026-11-15 (restored; RUN #5 of docs/plans/DELETED_ARMS_WORTH_RUNNING_2026-11-15.md, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_MAMBA3_LEGACY_CALLER_TRANSFER.patch
    comptime assert not is_defined["MOJOLEARN_MAMBA3_LEGACY_CALLER_TRANSFER"](), "removed 2026-11-15 (lane/expdel2-27-neural): MAMBA3_LEGACY_CALLER_TRANSFER (unmeasured old path: host-buffer staging of Mamba-3 transfers) is gone; the default stays; code recoverable at d57cf5cdab; restore: experiments/removed/MOJOLEARN_MAMBA3_LEGACY_CALLER_TRANSFER.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_MAMBA3_LEGACY_DIRECT_TRANSFER.patch
    comptime assert not is_defined["MOJOLEARN_MAMBA3_LEGACY_DIRECT_TRANSFER"](), "removed 2026-11-15 (lane/expdel2-27-neural): MAMBA3_LEGACY_DIRECT_TRANSFER (unmeasured old path: Mamba-3 transfers through host lists) is gone; the default stays; code recoverable at 9c516e8231; restore: experiments/removed/MOJOLEARN_MAMBA3_LEGACY_DIRECT_TRANSFER.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NN37_TRIANGLE_TASKS.patch
    comptime assert not is_defined["MOJOLEARN_NN37_TRIANGLE_TASKS"](), "removed 2026-11-15 (lane/expdel2-27-neural): NN37_TRIANGLE_TASKS (unmeasured opt-in arm: packed lower-triangle S12 tasks) is gone; the default stays; code recoverable at 617acefe68; restore: experiments/removed/MOJOLEARN_NN37_TRIANGLE_TASKS.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NN40_OWNED_WEIGHTS.patch
    comptime assert not is_defined["MOJOLEARN_NN40_OWNED_WEIGHTS"](), "removed 2026-11-15 (lane/expdel2-27-neural): NN40_OWNED_WEIGHTS (unmeasured opt-in arm: owned Mamba-3 weight snapshots) is gone; the default stays; code recoverable at 8584e42dde; restore: experiments/removed/MOJOLEARN_NN40_OWNED_WEIGHTS.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NN40_WS_HEADROOM.patch
    comptime assert not is_defined["MOJOLEARN_NN40_WS_HEADROOM"](), "removed 2026-11-15 (lane/expdel2-27-neural): NN40_WS_HEADROOM (unmeasured opt-in arm: GEMM workspace grow-ahead headroom) is gone; the default stays; code recoverable at 340d379a3c; restore: experiments/removed/MOJOLEARN_NN40_WS_HEADROOM.patch"
    # Lane expdel2-28-neural (2026-11-15): unmeasured arms and superseded old paths deleted, docs/plans/EXPERIMENT_DELETIONS_V2_2026-11-15.md (docs/TOMBSTONES.md).
    # MOJOLEARN_CNN_PREDICT_PROGRAM_COPY: refusal removed 2026-11-15 (restored; owed B arm, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_AF_COL_CHUNK_FOLD.patch
    comptime assert not is_defined["MOJOLEARN_IDN_AF_COL_CHUNK_FOLD"](), "removed 2026-11-15 (lane/expdel2-28-neural): IDN_AF_COL_CHUNK_FOLD (unmeasured opt-in arm) is gone; the default is kept; code at ab421749bb; restore: experiments/removed/MOJOLEARN_IDN_AF_COL_CHUNK_FOLD.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_LN_ROW_WARP_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_LN_ROW_WARP_OFF"](), "removed 2026-11-15 (lane/expdel2-28-neural): IDN_LN_ROW_WARP_OFF (old path beside the measured default) is gone; the default is kept; code at 730ca5e639; restore: experiments/removed/MOJOLEARN_IDN_LN_ROW_WARP_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_M2_GRAD_FOLD.patch
    comptime assert not is_defined["MOJOLEARN_IDN_M2_GRAD_FOLD"](), "removed 2026-11-15 (lane/expdel2-28-neural): IDN_M2_GRAD_FOLD (unmeasured opt-in arms) is gone; the default is kept; code at 92dbaa2ecb; restore: experiments/removed/MOJOLEARN_IDN_M2_GRAD_FOLD.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SEQ_AF_COOP_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SEQ_AF_COOP_OFF"](), "removed 2026-11-15 (lane/expdel2-28-neural): IDN_SEQ_AF_COOP_OFF (unmeasured old path) is gone; the default is kept; code at 2794197ed7; restore: experiments/removed/MOJOLEARN_IDN_SEQ_AF_COOP_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SEQ_COOP_NVAMD_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SEQ_COOP_NVAMD_OFF"](), "removed 2026-11-15 (lane/expdel2-28-neural): IDN_SEQ_COOP_NVAMD_OFF (unmeasured old path) is gone; the default is kept; code at 14eb8b482b; restore: experiments/removed/MOJOLEARN_IDN_SEQ_COOP_NVAMD_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SEQ_LN_COOP_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SEQ_LN_COOP_OFF"](), "removed 2026-11-15 (lane/expdel2-28-neural): IDN_SEQ_LN_COOP_OFF (unmeasured old path) is gone; the default is kept; code at 2c0d457d23; restore: experiments/removed/MOJOLEARN_IDN_SEQ_LN_COOP_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SEQ_PIPE_DOWN_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SEQ_PIPE_DOWN_OFF"](), "removed 2026-11-15 (lane/expdel2-28-neural): IDN_SEQ_PIPE_DOWN_OFF (unmeasured old path) is gone; the default is kept; code at c0a04b3bfd; restore: experiments/removed/MOJOLEARN_IDN_SEQ_PIPE_DOWN_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SEQ_PIPE_UP_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SEQ_PIPE_UP_OFF"](), "removed 2026-11-15 (lane/expdel2-28-neural): IDN_SEQ_PIPE_UP_OFF (unmeasured old path) is gone; the default is kept; code at b25786e9b3; restore: experiments/removed/MOJOLEARN_IDN_SEQ_PIPE_UP_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SEQ_WGRAD.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SEQ_WGRAD"](), "removed 2026-11-15 (lane/expdel2-28-neural): IDN_SEQ_WGRAD (unmeasured opt-in arms) is gone; the default is kept; code at d09d3a3b71; restore: experiments/removed/MOJOLEARN_IDN_SEQ_WGRAD.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SEQ_WGRAD_BLOCKED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SEQ_WGRAD_BLOCKED_OFF"](), "removed 2026-11-15 (lane/expdel2-28-neural): IDN_SEQ_WGRAD_BLOCKED_OFF (unmeasured old path) is gone; the default is kept; code at 105cf60fa6; restore: experiments/removed/MOJOLEARN_IDN_SEQ_WGRAD_BLOCKED_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SEQ_FAST_PIPE_CH.patch
    comptime assert not is_defined["MOJOLEARN_SEQ_FAST_PIPE_CH"](), "removed 2026-11-15 (lane/expdel2-28-neural): SEQ_FAST_PIPE_CH (unmeasured opt-in knob) is gone; the default is kept; code at c5730b729c; restore: experiments/removed/MOJOLEARN_SEQ_FAST_PIPE_CH.patch"
    # Lane expdel2-32-neural (2026-11-15): unmeasured arms and old paths deleted under "An unmeasured arm is deleted" (docs/plans/EXPERIMENT_DELETIONS_V2_2026-11-15.md, docs/TOMBSTONES.md).
    # MOJOLEARN_AFN_SAMBA_ARENA: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_SAMBA_DEVICE_ADMIT: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_SAMBA_EMB_ATOMIC: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_SAMBA_FUSE: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_SAMBA_HEAD_GEMM: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_ATTN_ES_TQ16.patch
    comptime assert not is_defined["MOJOLEARN_ATTN_ES_TQ16"](), "removed 2026-11-15 (lane/expdel2-32-neural): ATTN_ES_TQ16 (opt-in arm, no record) is gone; the default path is the only one; code recoverable at 284d3f2314; restore: experiments/removed/MOJOLEARN_ATTN_ES_TQ16.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ATTN_ES_TQ8.patch
    comptime assert not is_defined["MOJOLEARN_ATTN_ES_TQ8"](), "removed 2026-11-15 (lane/expdel2-32-neural): ATTN_ES_TQ8 (opt-in arm, no record) is gone; the default path is the only one; code recoverable at 284d3f2314; restore: experiments/removed/MOJOLEARN_ATTN_ES_TQ8.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_HOST_RMS_FORCE_SERIAL.patch
    comptime assert not is_defined["MOJOLEARN_HOST_RMS_FORCE_SERIAL"](), "removed 2026-11-15 (lane/expdel2-32-neural): HOST_RMS_FORCE_SERIAL (opt-in arm, no record) is gone; the default path is the only one; code recoverable at a7c1f95c40; restore: experiments/removed/MOJOLEARN_HOST_RMS_FORCE_SERIAL.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_CE_DENOM_ROWFOLD.patch
    comptime assert not is_defined["MOJOLEARN_IDN_CE_DENOM_ROWFOLD"](), "removed 2026-11-15 (lane/expdel2-32-neural): IDN_CE_DENOM_ROWFOLD (opt-in arm, no record) is gone; the default path is the only one; code recoverable at 311b50e05c; restore: experiments/removed/MOJOLEARN_IDN_CE_DENOM_ROWFOLD.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_CE_GRAD_FUSED.patch
    comptime assert not is_defined["MOJOLEARN_IDN_CE_GRAD_FUSED"](), "removed 2026-11-15 (lane/expdel2-32-neural): IDN_CE_GRAD_FUSED (opt-in arm, no record) is gone; the default path is the only one; code recoverable at 311b50e05c; restore: experiments/removed/MOJOLEARN_IDN_CE_GRAD_FUSED.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_CE_TOKEN_FOLD.patch
    comptime assert not is_defined["MOJOLEARN_IDN_CE_TOKEN_FOLD"](), "removed 2026-11-15 (lane/expdel2-32-neural): IDN_CE_TOKEN_FOLD (opt-in arm, EXP:2009 DROPPED (noise)) is gone; the default path is the only one; code recoverable at 19d2549589; restore: experiments/removed/MOJOLEARN_IDN_CE_TOKEN_FOLD.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_CHUNKED_LM_HEAD_V2.patch
    comptime assert not is_defined["MOJOLEARN_IDN_CHUNKED_LM_HEAD_V2"](), "removed 2026-11-15 (lane/expdel2-32-neural): IDN_CHUNKED_LM_HEAD_V2 (opt-in arm, record read: never measured) is gone; the default path is the only one; code recoverable at 311b50e05c; restore: experiments/removed/MOJOLEARN_IDN_CHUNKED_LM_HEAD_V2.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_LM_VIEWS.patch
    comptime assert not is_defined["MOJOLEARN_IDN_LM_VIEWS"](), "removed 2026-11-15 (lane/expdel2-32-neural): IDN_LM_VIEWS (opt-in arm, no record) is gone; the default path is the only one; code recoverable at 311b50e05c; restore: experiments/removed/MOJOLEARN_IDN_LM_VIEWS.patch"
    # MOJOLEARN_IDN_NORM_FIXED_LANES: refusal removed 2026-11-15 (restored; RUN #6 of docs/plans/DELETED_ARMS_WORTH_RUNNING_2026-11-15.md, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_OPT_CLIP_BATCHED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_OPT_CLIP_BATCHED_OFF"](), "removed 2026-11-15 (lane/expdel2-32-neural): IDN_OPT_CLIP_BATCHED_OFF (old path, no record) is gone; the default is kept; code recoverable at 25599a616f; restore: experiments/removed/MOJOLEARN_IDN_OPT_CLIP_BATCHED_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_OPT_GATE_SCAN_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_OPT_GATE_SCAN_OFF"](), "removed 2026-11-15 (lane/expdel2-32-neural): IDN_OPT_GATE_SCAN_OFF (old path, no record) is gone; the default is kept; code recoverable at 25599a616f; restore: experiments/removed/MOJOLEARN_IDN_OPT_GATE_SCAN_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_OPT_SGD_ONE_LAUNCH_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_OPT_SGD_ONE_LAUNCH_OFF"](), "removed 2026-11-15 (lane/expdel2-32-neural): IDN_OPT_SGD_ONE_LAUNCH_OFF (old path, no record) is gone; the default is kept; code recoverable at 25599a616f; restore: experiments/removed/MOJOLEARN_IDN_OPT_SGD_ONE_LAUNCH_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ROPE.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ROPE"](), "removed 2026-11-15 (lane/expdel2-32-neural): IDN_ROPE (opt-in arm, no record) is gone; the default path is the only one; code recoverable at 311b50e05c; restore: experiments/removed/MOJOLEARN_IDN_ROPE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SAMBA_ATTN_RECOMPUTE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SAMBA_ATTN_RECOMPUTE_OFF"](), "removed 2026-11-15 (lane/expdel2-32-neural): IDN_SAMBA_ATTN_RECOMPUTE_OFF (old path, EXP:1757 PROMOTED (recompute is the Samba default)) is gone; the default is kept; code recoverable at 311b50e05c; restore: experiments/removed/MOJOLEARN_IDN_SAMBA_ATTN_RECOMPUTE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SAMBA_DEV_REFUSE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SAMBA_DEV_REFUSE_OFF"](), "removed 2026-11-15 (lane/expdel2-32-neural): IDN_SAMBA_DEV_REFUSE_OFF (old path, no record) is gone; the default is kept; code recoverable at 0840b5da59; restore: experiments/removed/MOJOLEARN_IDN_SAMBA_DEV_REFUSE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_TRAIN_NO_DECODE_CACHE.patch
    comptime assert not is_defined["MOJOLEARN_IDN_TRAIN_NO_DECODE_CACHE"](), "removed 2026-11-15 (lane/expdel2-32-neural): IDN_TRAIN_NO_DECODE_CACHE (opt-in arm, no record) is gone; the default path is the only one; code recoverable at 311b50e05c; restore: experiments/removed/MOJOLEARN_IDN_TRAIN_NO_DECODE_CACHE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_TRAIN_SCRATCH.patch
    comptime assert not is_defined["MOJOLEARN_IDN_TRAIN_SCRATCH"](), "removed 2026-11-15 (lane/expdel2-32-neural): IDN_TRAIN_SCRATCH (opt-in arm, no record) is gone; the default path is the only one; code recoverable at 311b50e05c; restore: experiments/removed/MOJOLEARN_IDN_TRAIN_SCRATCH.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_TRAIN_SWIGLU.patch
    comptime assert not is_defined["MOJOLEARN_IDN_TRAIN_SWIGLU"](), "removed 2026-11-15 (lane/expdel2-32-neural): IDN_TRAIN_SWIGLU (opt-in arm, no record) is gone; the default path is the only one; code recoverable at 311b50e05c; restore: experiments/removed/MOJOLEARN_IDN_TRAIN_SWIGLU.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NI01_TRAINING_WORKSPACE.patch
    comptime assert not is_defined["MOJOLEARN_NI01_TRAINING_WORKSPACE"](), "removed 2026-11-15 (lane/expdel2-32-neural): NI01_TRAINING_WORKSPACE (opt-in arm, no record) is gone; the default path is the only one; code recoverable at 19d2549589; restore: experiments/removed/MOJOLEARN_NI01_TRAINING_WORKSPACE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NN21_SCALE_MASK.patch
    comptime assert not is_defined["MOJOLEARN_NN21_SCALE_MASK"](), "removed 2026-11-15 (lane/expdel2-32-neural): NN21_SCALE_MASK (opt-in arm, no record) is gone; the default path is the only one; code recoverable at 326b0d6e3d; restore: experiments/removed/MOJOLEARN_NN21_SCALE_MASK.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NN29_RING_PAIR.patch
    comptime assert not is_defined["MOJOLEARN_NN29_RING_PAIR"](), "removed 2026-11-15 (lane/expdel2-32-neural): NN29_RING_PAIR (opt-in arm, no record) is gone; the default path is the only one; code recoverable at 326b0d6e3d; restore: experiments/removed/MOJOLEARN_NN29_RING_PAIR.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NN58_ACCUMULATE_STATUS.patch
    comptime assert not is_defined["MOJOLEARN_NN58_ACCUMULATE_STATUS"](), "removed 2026-11-15 (lane/expdel2-32-neural): NN58_ACCUMULATE_STATUS (opt-in arm, no record) is gone; the default path is the only one; code recoverable at 5b6269eefe; restore: experiments/removed/MOJOLEARN_NN58_ACCUMULATE_STATUS.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NN59_CONTROL.patch
    comptime assert not is_defined["MOJOLEARN_NN59_CONTROL"](), "removed 2026-11-15 (lane/expdel2-32-neural): NN59_CONTROL (opt-in arm, no record) is gone; the default path is the only one; code recoverable at f75eb67b37; restore: experiments/removed/MOJOLEARN_NN59_CONTROL.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NN59_DROPOUT_RESIDUAL.patch
    comptime assert not is_defined["MOJOLEARN_NN59_DROPOUT_RESIDUAL"](), "removed 2026-11-15 (lane/expdel2-32-neural): NN59_DROPOUT_RESIDUAL (opt-in arm, no record) is gone; the default path is the only one; code recoverable at f75eb67b37; restore: experiments/removed/MOJOLEARN_NN59_DROPOUT_RESIDUAL.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NN63_CANONICAL_SHARD_MERGE.patch
    comptime assert not is_defined["MOJOLEARN_NN63_CANONICAL_SHARD_MERGE"](), "removed 2026-11-15 (lane/expdel2-32-neural): NN63_CANONICAL_SHARD_MERGE (opt-in arm, no record) is gone; the default path is the only one; code recoverable at 5f27699bee; restore: experiments/removed/MOJOLEARN_NN63_CANONICAL_SHARD_MERGE.patch"
    # MOJOLEARN_SAMBA_PROGRAM_FORWARD: refusal removed 2026-11-15 (restored; owed B arm, see docs/TOMBSTONES.md Restored table)
    # ---- end lane expdel2-32-neural
    # Lane expdel2-31-neural (2026-11-15): unmeasured arms and old paths deleted under "An unmeasured arm is deleted" (docs/plans/EXPERIMENT_DELETIONS_V2_2026-11-15.md, docs/TOMBSTONES.md).
    # MOJOLEARN_AFN_LM_ALL: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_LM_NOSYNC: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN26_LM_NOSYNC: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_LM_BWD_FUSE: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN26_LM_BWD_FUSE: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_LM_PARAM_VIEWS: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN26_LM_PARAM_VIEWS: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_LM_HEAD_FUSE: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN26_LM_HEAD_FUSE: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN26_LM_CE_BLOCK128: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN26_LM_CE_BLOCK512: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_LM_BWD_NOSYNC_OFF: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_BYTE_LM_CE_UNALIASED.patch
    comptime assert not is_defined["MOJOLEARN_BYTE_LM_CE_UNALIASED"](), "removed 2026-11-15 (lane/expdel2-31-neural): BYTE_LM_CE_UNALIASED (opt-in arm, Neural (byte LM CE buffers); no A/B recorded) is gone; the default path is the only one; code recoverable at 578fb1af7e; restore: experiments/removed/MOJOLEARN_BYTE_LM_CE_UNALIASED.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_BYTE_LM_STICKY_EAGER.patch
    comptime assert not is_defined["MOJOLEARN_BYTE_LM_STICKY_EAGER"](), "removed 2026-11-15 (lane/expdel2-31-neural): BYTE_LM_STICKY_EAGER (opt-in arm, Neural (byte LM attention eager fallback); no A/B recorded) is gone; the default path is the only one; code recoverable at d879623fb2; restore: experiments/removed/MOJOLEARN_BYTE_LM_STICKY_EAGER.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_BYTE_LM_RELEASE_EAGER.patch
    comptime assert not is_defined["MOJOLEARN_BYTE_LM_RELEASE_EAGER"](), "removed 2026-11-15 (lane/expdel2-31-neural): BYTE_LM_RELEASE_EAGER (opt-in arm, Neural (byte LM eager scratch release); no A/B recorded) is gone; the default path is the only one; code recoverable at d879623fb2; restore: experiments/removed/MOJOLEARN_BYTE_LM_RELEASE_EAGER.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_BYTE_LM_RETAIN_EAGER.patch
    comptime assert not is_defined["MOJOLEARN_BYTE_LM_RETAIN_EAGER"](), "removed 2026-11-15 (lane/expdel2-31-neural): BYTE_LM_RETAIN_EAGER (opt-in arm, Neural (byte LM eager scratch release); no A/B recorded) is gone; the default path is the only one; code recoverable at d879623fb2; restore: experiments/removed/MOJOLEARN_BYTE_LM_RETAIN_EAGER.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_BYTE_LM_COPY_EMB_HEAD.patch
    comptime assert not is_defined["MOJOLEARN_BYTE_LM_COPY_EMB_HEAD"](), "removed 2026-11-15 (lane/expdel2-31-neural): BYTE_LM_COPY_EMB_HEAD (opt-in arm, Neural (byte LM embedding / head weights); no A/B recorded) is gone; the default path is the only one; code recoverable at 9cc10dddc6; restore: experiments/removed/MOJOLEARN_BYTE_LM_COPY_EMB_HEAD.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_BYTE_LM_PROGRAM_STEP.fb12-loser-arms.patch
    comptime assert not is_defined["MOJOLEARN_BYTE_LM_PROGRAM_STEP"](), "removed 2026-11-17 (lane/fb12-loser-arms): BYTE_LM_PROGRAM_STEP (decided B arm, lane x7 program step on every column; L40S ba08dac006 lm-train-step bytes B 1.36x slower) is gone; the default path is the only one; code recoverable at 2203db7094; restore: experiments/removed/MOJOLEARN_BYTE_LM_PROGRAM_STEP.fb12-loser-arms.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_NORM.patch
    comptime assert not is_defined["MOJOLEARN_IDN_NORM"](), "removed 2026-11-15 (lane/expdel2-31-neural): IDN_NORM (opt-in arm, Neural (host byte LM legacy norm block); no A/B recorded) is gone; the default path is the only one; code recoverable at dc698300f4; restore: experiments/removed/MOJOLEARN_IDN_NORM.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NN53_HEAD_CHUNK512.patch
    comptime assert not is_defined["MOJOLEARN_NN53_HEAD_CHUNK512"](), "removed 2026-11-15 (lane/expdel2-31-neural): NN53_HEAD_CHUNK512 (opt-in arm, Neural (chunked LM head panel width); no A/B recorded) is gone; the default path is the only one; code recoverable at cba6209bbd; restore: experiments/removed/MOJOLEARN_NN53_HEAD_CHUNK512.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NN53_HEAD_CHUNK2048.patch
    comptime assert not is_defined["MOJOLEARN_NN53_HEAD_CHUNK2048"](), "removed 2026-11-15 (lane/expdel2-31-neural): NN53_HEAD_CHUNK2048 (opt-in arm, Neural (chunked LM head panel width); no A/B recorded) is gone; the default path is the only one; code recoverable at cba6209bbd; restore: experiments/removed/MOJOLEARN_NN53_HEAD_CHUNK2048.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_TRAIN_DEV_TENSORS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_TRAIN_DEV_TENSORS_OFF"](), "removed 2026-11-15 (lane/expdel2-31-neural): IDN_TRAIN_DEV_TENSORS_OFF (old path, Neural (training device tensors); no A/B recorded) is gone; the default is kept; code recoverable at 89a7547f4e; restore: experiments/removed/MOJOLEARN_IDN_TRAIN_DEV_TENSORS_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_OPT_SCRATCH_POOL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_OPT_SCRATCH_POOL_OFF"](), "removed 2026-11-15 (lane/expdel2-31-neural): IDN_OPT_SCRATCH_POOL_OFF (old path, Neural (optimizer scratch pool); no A/B recorded) is gone; the default is kept; code recoverable at 51b81017d7; restore: experiments/removed/MOJOLEARN_IDN_OPT_SCRATCH_POOL_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_MAXIMIZE_DEV_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_MAXIMIZE_DEV_OFF"](), "removed 2026-11-15 (lane/expdel2-31-neural): IDN_MAXIMIZE_DEV_OFF (old path, Neural (optimizer maximize on device); no A/B recorded) is gone; the default is kept; code recoverable at 51b81017d7; restore: experiments/removed/MOJOLEARN_IDN_MAXIMIZE_DEV_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_OPT_PARAMS_RESIDENT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_OPT_PARAMS_RESIDENT_OFF"](), "removed 2026-11-15 (lane/expdel2-31-neural): IDN_OPT_PARAMS_RESIDENT_OFF (old path, Neural (optimizer resident params entries); no A/B recorded) is gone; the default is kept; code recoverable at 51b81017d7; restore: experiments/removed/MOJOLEARN_IDN_OPT_PARAMS_RESIDENT_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_CE_PIPE_UP_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_CE_PIPE_UP_OFF"](), "removed 2026-11-15 (lane/expdel2-31-neural): IDN_CE_PIPE_UP_OFF (old path, Neural (CE logits upload pipeline); no A/B recorded) is gone; the default is kept; code recoverable at 51b81017d7; restore: experiments/removed/MOJOLEARN_IDN_CE_PIPE_UP_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TRAIN_OPT_FAST_PIPE_DOWN_OFF.patch
    comptime assert not is_defined["MOJOLEARN_TRAIN_OPT_FAST_PIPE_DOWN_OFF"](), "removed 2026-11-15 (lane/expdel2-31-neural): TRAIN_OPT_FAST_PIPE_DOWN_OFF (old path, Neural (Apple FAST optimizer download pipeline); no A/B recorded) is gone; the default is kept; code recoverable at 51b81017d7; restore: experiments/removed/MOJOLEARN_TRAIN_OPT_FAST_PIPE_DOWN_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_BYTE_LM_VSUM_HD_GENERIC_OFF.patch
    comptime assert not is_defined["MOJOLEARN_BYTE_LM_VSUM_HD_GENERIC_OFF"](), "removed 2026-11-15 (lane/expdel2-31-neural): BYTE_LM_VSUM_HD_GENERIC_OFF (old path, Neural (host byte LM attention value sum); no A/B recorded) is gone; the default is kept; code recoverable at 68a9b6cf9e; restore: experiments/removed/MOJOLEARN_BYTE_LM_VSUM_HD_GENERIC_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_BYTE_LM_HOST_LEGACY_GEMM.patch
    comptime assert not is_defined["MOJOLEARN_BYTE_LM_HOST_LEGACY_GEMM"](), "removed 2026-11-15 (lane/expdel2-31-neural): BYTE_LM_HOST_LEGACY_GEMM (old path, Neural (host byte LM weight GEMMs); no A/B recorded) is gone; the default is kept; code recoverable at fb518878c4; restore: experiments/removed/MOJOLEARN_BYTE_LM_HOST_LEGACY_GEMM.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_OPT_TRUST_INPUTS.patch
    comptime assert not is_defined["MOJOLEARN_OPT_TRUST_INPUTS"](), "removed 2026-11-15 (lane/expdel2-31-neural): OPT_TRUST_INPUTS (opt-in arm, Neural (optimizer input refusal scans); no A/B recorded) is gone; the default path is the only one; code recoverable at 943c1ff5fe; restore: experiments/removed/MOJOLEARN_OPT_TRUST_INPUTS.patch"
    # ---- end lane expdel2-31-neural
    # Lane expdel2-29-neural (2026-11-15): unmeasured arms and old paths deleted, docs/plans/EXPERIMENT_DELETIONS_V2_2026-11-15.md (docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_GEMM_DISPATCH_V1.patch
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_GEMM_DISPATCH_V1"](), "removed 2026-11-15 (lane/expdel2-29-neural): AUTOGRAD_GEMM_DISPATCH_V1 (old path) is gone; the default is kept; code at 8747cf5fa2; restore: experiments/removed/MOJOLEARN_AUTOGRAD_GEMM_DISPATCH_V1.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_MLP_UNFUSED.patch
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_MLP_UNFUSED"](), "removed 2026-11-15 (lane/expdel2-29-neural): AUTOGRAD_MLP_UNFUSED (arm) is gone; the default path is the only one; code at a03ced9b31; restore: experiments/removed/MOJOLEARN_AUTOGRAD_MLP_UNFUSED.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_AUTOGRAD_OPTIM.fb12-loser-arms.patch
    comptime assert not is_defined["MOJOLEARN_AUTOGRAD_OPTIM"](), "removed 2026-11-17 (lane/fb12-loser-arms): AUTOGRAD_OPTIM (decided B arm, executor optimizer step_lr in mlp_fit / rnn_fit; L40S ba08dac006 B 0.99-1.01x, same speed) is gone; the default path is the only one; code recoverable at 2203db7094; restore: experiments/removed/MOJOLEARN_AUTOGRAD_OPTIM.fb12-loser-arms.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_GARCH_COOP_OFF.patch
    comptime assert not is_defined["MOJOLEARN_GARCH_COOP_OFF"](), "removed 2026-11-15 (lane/expdel2-29-neural): GARCH_COOP_OFF (old path) is gone; the default is kept; code at 92fec7163d; restore: experiments/removed/MOJOLEARN_GARCH_COOP_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GARCH_COOP_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GARCH_COOP_OFF"](), "removed 2026-11-15 (lane/expdel2-29-neural): IDN_GARCH_COOP_OFF (old path) is gone; the default is kept; code at 02b0c1f6db; restore: experiments/removed/MOJOLEARN_IDN_GARCH_COOP_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GARCH_FOLD32_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GARCH_FOLD32_OFF"](), "removed 2026-11-15 (lane/expdel2-29-neural): IDN_GARCH_FOLD32_OFF (old path) is gone; the default is kept; code at 869fa6e1fb; restore: experiments/removed/MOJOLEARN_IDN_GARCH_FOLD32_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GARCH_LEAD_REG_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GARCH_LEAD_REG_OFF"](), "removed 2026-11-15 (lane/expdel2-29-neural): IDN_GARCH_LEAD_REG_OFF (old path) is gone; the default is kept; code at 476f794798; restore: experiments/removed/MOJOLEARN_IDN_GARCH_LEAD_REG_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GEMM_ROUTE_P6_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GEMM_ROUTE_P6_OFF"](), "removed 2026-11-15 (lane/expdel2-29-neural): IDN_GEMM_ROUTE_P6_OFF (old path) is gone; the default is kept; code at a9e416c657; restore: experiments/removed/MOJOLEARN_IDN_GEMM_ROUTE_P6_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GEMM_SPLITK_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GEMM_SPLITK_OFF"](), "removed 2026-11-15 (lane/expdel2-29-neural): IDN_GEMM_SPLITK_OFF (old path) is gone; the default is kept; code at 6817978647; restore: experiments/removed/MOJOLEARN_IDN_GEMM_SPLITK_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GEMM_STAGE64.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GEMM_STAGE64"](), "removed 2026-11-15 (lane/expdel2-29-neural): IDN_GEMM_STAGE64 (arm) is gone; the default path is the only one; code at 6a66db5157; restore: experiments/removed/MOJOLEARN_IDN_GEMM_STAGE64.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GEMM_STAGE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GEMM_STAGE_OFF"](), "removed 2026-11-15 (lane/expdel2-29-neural): IDN_GEMM_STAGE_OFF (old path) is gone; the default is kept; code at 6a66db5157; restore: experiments/removed/MOJOLEARN_IDN_GEMM_STAGE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_LN_ONE_PASS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_LN_ONE_PASS_OFF"](), "removed 2026-11-15 (lane/expdel2-29-neural): IDN_LN_ONE_PASS_OFF (old path) is gone; the default is kept; code at f12b3def4e; restore: experiments/removed/MOJOLEARN_IDN_LN_ONE_PASS_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_MLP_BLOCKED_FOLDS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_MLP_BLOCKED_FOLDS_OFF"](), "removed 2026-11-15 (lane/expdel2-29-neural): IDN_MLP_BLOCKED_FOLDS_OFF (old path) is gone; the default is kept; code at 92d4b79ec8; restore: experiments/removed/MOJOLEARN_IDN_MLP_BLOCKED_FOLDS_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_MLP_EPOCH_DEV_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_MLP_EPOCH_DEV_OFF"](), "removed 2026-11-15 (lane/expdel2-29-neural): IDN_MLP_EPOCH_DEV_OFF (old path) is gone; the default is kept; code at 0064646ba8; restore: experiments/removed/MOJOLEARN_IDN_MLP_EPOCH_DEV_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_MOE_GROUPED_GEMM_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_MOE_GROUPED_GEMM_OFF"](), "removed 2026-11-15 (lane/expdel2-29-neural): IDN_MOE_GROUPED_GEMM_OFF (old path) is gone; the default is kept; code at 072f2eee14; restore: experiments/removed/MOJOLEARN_IDN_MOE_GROUPED_GEMM_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_MOE_STABLE_PACK.patch
    comptime assert not is_defined["MOJOLEARN_IDN_MOE_STABLE_PACK"](), "removed 2026-11-15 (lane/expdel2-29-neural): IDN_MOE_STABLE_PACK (arm) is gone; the default path is the only one; code at d1f44888a3; restore: experiments/removed/MOJOLEARN_IDN_MOE_STABLE_PACK.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SEQ_GEMM_MICRO.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SEQ_GEMM_MICRO"](), "removed 2026-11-15 (lane/expdel2-29-neural): IDN_SEQ_GEMM_MICRO (arm) is gone; the default path is the only one; code at 84eb54a8c8; restore: experiments/removed/MOJOLEARN_IDN_SEQ_GEMM_MICRO.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SEQ_GEMM_MICRO_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SEQ_GEMM_MICRO_OFF"](), "removed 2026-11-15 (lane/expdel2-29-neural): IDN_SEQ_GEMM_MICRO_OFF (old path) is gone; the default is kept; code at 84eb54a8c8; restore: experiments/removed/MOJOLEARN_IDN_SEQ_GEMM_MICRO_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SEQ_GEMM_TILED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SEQ_GEMM_TILED_OFF"](), "removed 2026-11-15 (lane/expdel2-29-neural): IDN_SEQ_GEMM_TILED_OFF (old path) is gone; the default is kept; code at 4f0461d50d; restore: experiments/removed/MOJOLEARN_IDN_SEQ_GEMM_TILED_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SEQ_GEMM_TILED_V2_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SEQ_GEMM_TILED_V2_OFF"](), "removed 2026-11-15 (lane/expdel2-29-neural): IDN_SEQ_GEMM_TILED_V2_OFF (old path) is gone; the default is kept; code at 27a4108c9f; restore: experiments/removed/MOJOLEARN_IDN_SEQ_GEMM_TILED_V2_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SEQ_GEMM_TILED_V3_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SEQ_GEMM_TILED_V3_OFF"](), "removed 2026-11-15 (lane/expdel2-29-neural): IDN_SEQ_GEMM_TILED_V3_OFF (old path) is gone; the default is kept; code at fcf169192e; restore: experiments/removed/MOJOLEARN_IDN_SEQ_GEMM_TILED_V3_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SEQ_LN_LEAF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SEQ_LN_LEAF"](), "removed 2026-11-15 (lane/expdel2-29-neural): IDN_SEQ_LN_LEAF (arm) is gone; the default path is the only one; code at 07dabb1b82; restore: experiments/removed/MOJOLEARN_IDN_SEQ_LN_LEAF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_SEQ_SOFTMAX_ONE_EXP_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_SEQ_SOFTMAX_ONE_EXP_OFF"](), "removed 2026-11-15 (lane/expdel2-29-neural): IDN_SEQ_SOFTMAX_ONE_EXP_OFF (old path) is gone; the default is kept; code at 38b93382d0; restore: experiments/removed/MOJOLEARN_IDN_SEQ_SOFTMAX_ONE_EXP_OFF.patch"
    # MOJOLEARN_LN_FAST_BLOCK_FOLD: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_LN_FOLD_BLOCK_OFF.patch
    comptime assert not is_defined["MOJOLEARN_LN_FOLD_BLOCK_OFF"](), "removed 2026-11-15 (lane/expdel2-29-neural): LN_FOLD_BLOCK_OFF (old path) is gone; the default is kept; code at 8e9b1a9eb5; restore: experiments/removed/MOJOLEARN_LN_FOLD_BLOCK_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_MLP_LOSSBLOCK_OFF.patch
    comptime assert not is_defined["MOJOLEARN_MLP_LOSSBLOCK_OFF"](), "removed 2026-11-15 (lane/expdel2-29-neural): MLP_LOSSBLOCK_OFF (old path) is gone; the default is kept; code at e06ff6f0f3; restore: experiments/removed/MOJOLEARN_MLP_LOSSBLOCK_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_MLP_LOSSBLOCK_ROWS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_MLP_LOSSBLOCK_ROWS_OFF"](), "removed 2026-11-15 (lane/expdel2-29-neural): MLP_LOSSBLOCK_ROWS_OFF (old path) is gone; the default is kept; code at 6ed945170f; restore: experiments/removed/MOJOLEARN_MLP_LOSSBLOCK_ROWS_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_MOE_DEVGROUP_OFF.patch
    comptime assert not is_defined["MOJOLEARN_MOE_DEVGROUP_OFF"](), "removed 2026-11-15 (lane/expdel2-29-neural): MOE_DEVGROUP_OFF (old path) is gone; the default is kept; code at 2ca18f9b39; restore: experiments/removed/MOJOLEARN_MOE_DEVGROUP_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_MOE_FAST_MMA_OFF.patch
    comptime assert not is_defined["MOJOLEARN_MOE_FAST_MMA_OFF"](), "removed 2026-11-15 (lane/expdel2-29-neural): MOE_FAST_MMA_OFF (old path) is gone; the default is kept; code at 1e5b507b04; restore: experiments/removed/MOJOLEARN_MOE_FAST_MMA_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_MOE_REGTILE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_MOE_REGTILE_OFF"](), "removed 2026-11-15 (lane/expdel2-29-neural): MOE_REGTILE_OFF (old path) is gone; the default is kept; code at 253d6543b0; restore: experiments/removed/MOJOLEARN_MOE_REGTILE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NN42_INPUT_BIAS_FUSED.patch
    comptime assert not is_defined["MOJOLEARN_NN42_INPUT_BIAS_FUSED"](), "removed 2026-11-15 (lane/expdel2-29-neural): NN42_INPUT_BIAS_FUSED (arm) is gone; the default path is the only one; code at c906dcaf57; restore: experiments/removed/MOJOLEARN_NN42_INPUT_BIAS_FUSED.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NN44_EXPERT_BISECT.patch
    comptime assert not is_defined["MOJOLEARN_NN44_EXPERT_BISECT"](), "removed 2026-11-15 (lane/expdel2-29-neural): NN44_EXPERT_BISECT (arm) is gone; the default path is the only one; code at 5a1c6385fe; restore: experiments/removed/MOJOLEARN_NN44_EXPERT_BISECT.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SEQ_FAST_FMA_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SEQ_FAST_FMA_OFF"](), "removed 2026-11-15 (lane/expdel2-29-neural): SEQ_FAST_FMA_OFF (old path) is gone; the default is kept; code at 2f67dc66d0; restore: experiments/removed/MOJOLEARN_SEQ_FAST_FMA_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SEQ_FAST_VAR_ONECOPY_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SEQ_FAST_VAR_ONECOPY_OFF"](), "removed 2026-11-15 (lane/expdel2-29-neural): SEQ_FAST_VAR_ONECOPY_OFF (old path) is gone; the default is kept; code at c860d9566d; restore: experiments/removed/MOJOLEARN_SEQ_FAST_VAR_ONECOPY_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SEQ_GARCH_GRID_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SEQ_GARCH_GRID_OFF"](), "removed 2026-11-15 (lane/expdel2-29-neural): SEQ_GARCH_GRID_OFF (old path) is gone; the default is kept; code at bc254a427e; restore: experiments/removed/MOJOLEARN_SEQ_GARCH_GRID_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SEQ_GARCH_REG_OFF.patch
    comptime assert not is_defined["MOJOLEARN_SEQ_GARCH_REG_OFF"](), "removed 2026-11-15 (lane/expdel2-29-neural): SEQ_GARCH_REG_OFF (old path) is gone; the default is kept; code at 7a79b3b608; restore: experiments/removed/MOJOLEARN_SEQ_GARCH_REG_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_SEQ_REC_STAGE64.patch
    comptime assert not is_defined["MOJOLEARN_SEQ_REC_STAGE64"](), "removed 2026-11-15 (lane/expdel2-29-neural): SEQ_REC_STAGE64 (arm) is gone; the default path is the only one; code at d5aeff2e08; restore: experiments/removed/MOJOLEARN_SEQ_REC_STAGE64.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TSA2_STL_OFF.patch
    comptime assert not is_defined["MOJOLEARN_TSA2_STL_OFF"](), "removed 2026-11-15 (lane/expdel2-29-neural): TSA2_STL_OFF (old path) is gone; the default is kept; code at eb3180868d; restore: experiments/removed/MOJOLEARN_TSA2_STL_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TSA2_VAR_OFF.patch
    comptime assert not is_defined["MOJOLEARN_TSA2_VAR_OFF"](), "removed 2026-11-15 (lane/expdel2-29-neural): TSA2_VAR_OFF (old path) is gone; the default is kept; code at b23d7039b8; restore: experiments/removed/MOJOLEARN_TSA2_VAR_OFF.patch"
    # Lane expdel2-33-neural (2026-11-15): unmeasured arms and superseded old paths deleted, docs/plans/EXPERIMENT_DELETIONS_V2_2026-11-15.md (docs/TOMBSTONES.md).
    # MOJOLEARN_AFN_ATTN_ALL: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_ATTN_NORM_SG: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_ATTN_ROPE_CACHE: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_ATTN_GQA_TILE: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_ATTN_FLASH: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_ATTN_FUSE_PRE: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_ATTN_FUSE_MLP: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_ATTN_AUDIT: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_AFN_FLASH_HD_GENERIC_OFF.patch
    comptime assert not is_defined["MOJOLEARN_AFN_FLASH_HD_GENERIC_OFF"](), "removed 2026-11-15 (lane/expdel2-33-neural): AFN_FLASH_HD_GENERIC_OFF (unmeasured old path) is gone; the default is kept; code at c3d1c7865f; restore: experiments/removed/MOJOLEARN_AFN_FLASH_HD_GENERIC_OFF.patch"
    # MOJOLEARN_AFN26_ATTN_NORM_TPB128: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN26_ATTN_NORM_TPB512: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN26_ATTN_FLASH_BK16: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN26_ATTN_FLASH_TQ16: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN26_ATTN_PROJ_KB16: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN26_ATTN_PROJ_BM32: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN_CNN_DIRECT: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN26_CNN_ROWS32: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN26_CNN_CHANNELS16: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN26_CNN_K16: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_AFN26_CNN_COLS_ONCE: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_BN_FOLD_BLOCK_OFF.patch
    comptime assert not is_defined["MOJOLEARN_BN_FOLD_BLOCK_OFF"](), "removed 2026-11-15 (lane/expdel2-33-neural): BN_FOLD_BLOCK_OFF (unmeasured old path) is gone; the default is kept; code at c3d1c7865f; restore: experiments/removed/MOJOLEARN_BN_FOLD_BLOCK_OFF.patch"
    # MOJOLEARN_IDN_ATTN_FOLD_V2: refusal removed 2026-11-15 (restored; RUN #3 of docs/plans/DELETED_ARMS_WORTH_RUNNING_2026-11-15.md, see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_DISABLE_RESIDUAL2_NEXT_NORM.patch
    comptime assert not is_defined["MOJOLEARN_DISABLE_RESIDUAL2_NEXT_NORM"](), "removed 2026-11-15 (lane/expdel2-33-neural): DISABLE_RESIDUAL2_NEXT_NORM (unmeasured opt-in arm) is gone; the default is kept; code at c3d1c7865f; restore: experiments/removed/MOJOLEARN_DISABLE_RESIDUAL2_NEXT_NORM.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_FUSE_RESIDUAL1_NORM2.patch
    comptime assert not is_defined["MOJOLEARN_FUSE_RESIDUAL1_NORM2"](), "removed 2026-11-15 (lane/expdel2-33-neural): FUSE_RESIDUAL1_NORM2 (unmeasured opt-in arm) is gone; the default is kept; code at c3d1c7865f; restore: experiments/removed/MOJOLEARN_FUSE_RESIDUAL1_NORM2.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ATTN_CACHE_WAIT.patch
    comptime assert not is_defined["MOJOLEARN_ATTN_CACHE_WAIT"](), "removed 2026-11-15 (lane/expdel2-33-neural): ATTN_CACHE_WAIT (unmeasured opt-in arm) is gone; the default is kept; code at c3d1c7865f; restore: experiments/removed/MOJOLEARN_ATTN_CACHE_WAIT.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ATTN_CACHE_NOWAIT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ATTN_CACHE_NOWAIT_OFF"](), "removed 2026-11-15 (lane/expdel2-33-neural): IDN_ATTN_CACHE_NOWAIT_OFF (unmeasured old path) is gone; the default is kept; code at c3d1c7865f; restore: experiments/removed/MOJOLEARN_IDN_ATTN_CACHE_NOWAIT_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_LLAMA_REFUSE_BATCH_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_LLAMA_REFUSE_BATCH_OFF"](), "removed 2026-11-15 (lane/expdel2-33-neural): IDN_LLAMA_REFUSE_BATCH_OFF (unmeasured old path) is gone; the default is kept; code at c3d1c7865f; restore: experiments/removed/MOJOLEARN_IDN_LLAMA_REFUSE_BATCH_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TRANSFORMER_LEGACY_CALLER_TRANSFER.patch
    comptime assert not is_defined["MOJOLEARN_TRANSFORMER_LEGACY_CALLER_TRANSFER"](), "removed 2026-11-15 (lane/expdel2-33-neural): TRANSFORMER_LEGACY_CALLER_TRANSFER (unmeasured old path) is gone; the default is kept; code at c3d1c7865f; restore: experiments/removed/MOJOLEARN_TRANSFORMER_LEGACY_CALLER_TRANSFER.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_TRANSFORMER_LEGACY_WEIGHT_COPY.patch
    comptime assert not is_defined["MOJOLEARN_TRANSFORMER_LEGACY_WEIGHT_COPY"](), "removed 2026-11-15 (lane/expdel2-33-neural): TRANSFORMER_LEGACY_WEIGHT_COPY (unmeasured old path) is gone; the default is kept; code at c3d1c7865f; restore: experiments/removed/MOJOLEARN_TRANSFORMER_LEGACY_WEIGHT_COPY.patch"
    # Lane expdel2-36-neural-series-linear (2026-11-15): unmeasured arms and old paths deleted under "An unmeasured arm is deleted" (docs/plans/EXPERIMENT_DELETIONS_V2_2026-11-15.md, docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_CNN_CONV_RELU.patch
    comptime assert not is_defined["MOJOLEARN_IDN_CNN_CONV_RELU"](), "removed 2026-11-15 (lane/expdel2-36-neural-series-linear): IDN_CNN_CONV_RELU (opt-in arm, no A/B recorded, Neural (CNN conv + bias + ReLU fusion arms 1/2)) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_IDN_CNN_CONV_RELU.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NN46_GATHER_BOUNDS.patch
    comptime assert not is_defined["MOJOLEARN_NN46_GATHER_BOUNDS"](), "removed 2026-11-15 (lane/expdel2-36-neural-series-linear): NN46_GATHER_BOUNDS (opt-in arm, no A/B recorded, Neural (CNN backward gather tap bounds)) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_NN46_GATHER_BOUNDS.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NN47_APPLY_RUNNING.patch
    comptime assert not is_defined["MOJOLEARN_NN47_APPLY_RUNNING"](), "removed 2026-11-15 (lane/expdel2-36-neural-series-linear): NN47_APPLY_RUNNING (opt-in arm, no A/B recorded, Neural (BatchNorm apply + running-state update in one launch)) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_NN47_APPLY_RUNNING.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NI10_BUDGETED_CNN_TAPE.patch
    comptime assert not is_defined["MOJOLEARN_NI10_BUDGETED_CNN_TAPE"](), "removed 2026-11-15 (lane/expdel2-36-neural-series-linear): NI10_BUDGETED_CNN_TAPE (opt-in arm, no A/B recorded, Neural (CNN tape lifetime policy, 64 MiB budget)) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_NI10_BUDGETED_CNN_TAPE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NI10_RECOMPUTE_CNN_TAPE.patch
    comptime assert not is_defined["MOJOLEARN_NI10_RECOMPUTE_CNN_TAPE"](), "removed 2026-11-15 (lane/expdel2-36-neural-series-linear): NI10_RECOMPUTE_CNN_TAPE (opt-in arm, no A/B recorded, Neural (CNN tape lifetime policy, recompute)) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_NI10_RECOMPUTE_CNN_TAPE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NI17_BALANCED_NORM.patch
    comptime assert not is_defined["MOJOLEARN_NI17_BALANCED_NORM"](), "removed 2026-11-15 (lane/expdel2-36-neural-series-linear): NI17_BALANCED_NORM (opt-in arm, no A/B recorded, Neural (BatchNorm / pooling fold shape)) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_NI17_BALANCED_NORM.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_XENT_FOLD_BLOCK_256.patch
    comptime assert not is_defined["MOJOLEARN_IDN_XENT_FOLD_BLOCK_256"](), "removed 2026-11-15 (lane/expdel2-36-neural-series-linear): IDN_XENT_FOLD_BLOCK_256 (opt-in arm, no A/B recorded, Neural (CNN loss blocked fold width)) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_IDN_XENT_FOLD_BLOCK_256.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NN14_BOUNDED_IM2COL.patch
    comptime assert not is_defined["MOJOLEARN_NN14_BOUNDED_IM2COL"](), "removed 2026-11-15 (lane/expdel2-36-neural-series-linear): NN14_BOUNDED_IM2COL (opt-in arm, no A/B recorded, Neural (CNN bounded im2col forward / virtual im2col backward)) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_NN14_BOUNDED_IM2COL.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_XCNN_NO_POOL_BWD_TILE.patch
    comptime assert not is_defined["MOJOLEARN_XCNN_NO_POOL_BWD_TILE"](), "removed 2026-11-15 (lane/expdel2-36-neural-series-linear): XCNN_NO_POOL_BWD_TILE (opt-in arm, no A/B recorded, Neural (MaxPool backward tiling shortcut)) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_XCNN_NO_POOL_BWD_TILE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_AVGPOOL_BWD_TILE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_AVGPOOL_BWD_TILE_OFF"](), "removed 2026-11-15 (lane/expdel2-36-neural-series-linear): IDN_AVGPOOL_BWD_TILE_OFF (old path, no A/B recorded, Neural (AvgPool2d backward tile)) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_IDN_AVGPOOL_BWD_TILE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_PAD_BWD_BOUNDED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_PAD_BWD_BOUNDED_OFF"](), "removed 2026-11-15 (lane/expdel2-36-neural-series-linear): IDN_PAD_BWD_BOUNDED_OFF (old path, no A/B recorded, Neural (pad backward bounds)) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_IDN_PAD_BWD_BOUNDED_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ADAPT_BWD_BOUNDED_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ADAPT_BWD_BOUNDED_OFF"](), "removed 2026-11-15 (lane/expdel2-36-neural-series-linear): IDN_ADAPT_BWD_BOUNDED_OFF (old path, no A/B recorded, Neural (adaptive pool backward bounds)) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_IDN_ADAPT_BWD_BOUNDED_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_XCNN_SOFTMAX_ONE_EXP_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_XCNN_SOFTMAX_ONE_EXP_OFF"](), "removed 2026-11-15 (lane/expdel2-36-neural-series-linear): IDN_XCNN_SOFTMAX_ONE_EXP_OFF (old path, no A/B recorded, Neural (CNN softmax cross entropy)) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_IDN_XCNN_SOFTMAX_ONE_EXP_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_DROPOUT2D_CH_MASK_OFF.patch
    comptime assert not is_defined["MOJOLEARN_DROPOUT2D_CH_MASK_OFF"](), "removed 2026-11-15 (lane/expdel2-36-neural-series-linear): DROPOUT2D_CH_MASK_OFF (old path, no A/B recorded, Neural (Dropout2d channel mask)) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_DROPOUT2D_CH_MASK_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GAP_BLOCK_FOLD_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GAP_BLOCK_FOLD_OFF"](), "removed 2026-11-15 (lane/expdel2-36-neural-series-linear): IDN_GAP_BLOCK_FOLD_OFF (old path, no A/B recorded, Neural (global average pool fold)) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_IDN_GAP_BLOCK_FOLD_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_XENT_DEV_FOLD_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_XENT_DEV_FOLD_OFF"](), "removed 2026-11-15 (lane/expdel2-36-neural-series-linear): IDN_XENT_DEV_FOLD_OFF (old path, no A/B recorded, Neural (CNN loss mean fold on the device)) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_IDN_XENT_DEV_FOLD_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GCN_LOOPS_DEV_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GCN_LOOPS_DEV_OFF"](), "removed 2026-11-15 (lane/expdel2-36-neural-series-linear): IDN_GCN_LOOPS_DEV_OFF (old path, no A/B recorded, Neural (GCN self loops on the device)) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_IDN_GCN_LOOPS_DEV_OFF.patch"
    # ---- end lane expdel2-36-neural-series-linear
    # Lane expdel2-35-neural (2026-11-15): unmeasured arms and old paths deleted under "An unmeasured arm is deleted" (docs/plans/EXPERIMENT_DELETIONS_V2_2026-11-15.md, docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ADAPT_M_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ADAPT_M_OFF"](), "removed 2026-11-15 (lane/expdel2-35-neural): IDN_ADAPT_M_OFF (old path, Neural (x_cnn adaptive pooling device entries); no A/B recorded) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_IDN_ADAPT_M_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GRAPH_M_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GRAPH_M_OFF"](), "removed 2026-11-15 (lane/expdel2-35-neural): IDN_GRAPH_M_OFF (old path, Neural (x_cnn SAGE / L2 / gcn_norm device entries); no A/B recorded) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_IDN_GRAPH_M_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_PAD_M_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_PAD_M_OFF"](), "removed 2026-11-15 (lane/expdel2-35-neural): IDN_PAD_M_OFF (old path, Neural (x_cnn padded convolution residency); no A/B recorded) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_IDN_PAD_M_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GROUP_M_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GROUP_M_OFF"](), "removed 2026-11-15 (lane/expdel2-35-neural): IDN_GROUP_M_OFF (old path, Neural (x_cnn grouped convolution residency); no A/B recorded) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_IDN_GROUP_M_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_LAYER_DEV_IO_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_LAYER_DEV_IO_OFF"](), "removed 2026-11-15 (lane/expdel2-35-neural): IDN_LAYER_DEV_IO_OFF (old path, Neural (x_cnn BatchNorm/Dropout2d/ReLU resident IO); no A/B recorded) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_IDN_LAYER_DEV_IO_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_PIN_WEIGHTS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_PIN_WEIGHTS_OFF"](), "removed 2026-11-15 (lane/expdel2-35-neural): IDN_PIN_WEIGHTS_OFF (old path, Neural (x_cnn pinned layer weights); no A/B recorded) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_IDN_PIN_WEIGHTS_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_CNN_RESIDENT_GRADS_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_CNN_RESIDENT_GRADS_OFF"](), "removed 2026-11-15 (lane/expdel2-35-neural): IDN_CNN_RESIDENT_GRADS_OFF (old path, Neural (x_cnn resident CNN grads glue); no A/B recorded) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_IDN_CNN_RESIDENT_GRADS_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_DIRECT_CONV_NVAMD_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_DIRECT_CONV_NVAMD_OFF"](), "removed 2026-11-15 (lane/expdel2-35-neural): IDN_DIRECT_CONV_NVAMD_OFF (old path, Neural (x_cnn direct first-layer conv on NVIDIA/AMD); no A/B recorded) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_IDN_DIRECT_CONV_NVAMD_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_XCNN_BIAS_FOLD_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_XCNN_BIAS_FOLD_OFF"](), "removed 2026-11-15 (lane/expdel2-35-neural): IDN_XCNN_BIAS_FOLD_OFF (old path, Neural (x_cnn bias gradient column fold); no A/B recorded) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_IDN_XCNN_BIAS_FOLD_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NI11_DIRECT_CONV_TAPS64.patch
    comptime assert not is_defined["MOJOLEARN_NI11_DIRECT_CONV_TAPS64"](), "removed 2026-11-15 (lane/expdel2-35-neural): NI11_DIRECT_CONV_TAPS64 (opt-in arm, Neural (x_cnn direct conv register taps); no A/B recorded) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_NI11_DIRECT_CONV_TAPS64.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NI14_COL2IM.patch
    comptime assert not is_defined["MOJOLEARN_NI14_COL2IM"](), "removed 2026-11-15 (lane/expdel2-35-neural): NI14_COL2IM (opt-in arm, Neural (x_cnn col2im seam arms); no A/B recorded) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_NI14_COL2IM.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NI15_BIAS_NO_ONES.patch
    comptime assert not is_defined["MOJOLEARN_NI15_BIAS_NO_ONES"](), "removed 2026-11-15 (lane/expdel2-35-neural): NI15_BIAS_NO_ONES (opt-in arm, Neural (x_cnn bias gradient without ones); no A/B recorded) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_NI15_BIAS_NO_ONES.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NI55_GRAPH_FEATURE4.patch
    comptime assert not is_defined["MOJOLEARN_NI55_GRAPH_FEATURE4"](), "removed 2026-11-15 (lane/expdel2-35-neural): NI55_GRAPH_FEATURE4 (opt-in arm, Neural (GCN spmm four-feature tile); no A/B recorded) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_NI55_GRAPH_FEATURE4.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NI57_GRAPH_TREE64.patch
    comptime assert not is_defined["MOJOLEARN_NI57_GRAPH_TREE64"](), "removed 2026-11-15 (lane/expdel2-35-neural): NI57_GRAPH_TREE64 (opt-in arm, Neural (graph edge 64-leaf tree fold); no A/B recorded) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_NI57_GRAPH_TREE64.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NI58_SAGE_FEATURE4.patch
    comptime assert not is_defined["MOJOLEARN_NI58_SAGE_FEATURE4"](), "removed 2026-11-15 (lane/expdel2-35-neural): NI58_SAGE_FEATURE4 (opt-in arm, Neural (SAGE max four-feature tile); no A/B recorded) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_NI58_SAGE_FEATURE4.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NI59_DROPOUT_CHANNEL.patch
    comptime assert not is_defined["MOJOLEARN_NI59_DROPOUT_CHANNEL"](), "removed 2026-11-15 (lane/expdel2-35-neural): NI59_DROPOUT_CHANNEL (opt-in arm, Neural (Dropout2d one block per channel); no A/B recorded) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_NI59_DROPOUT_CHANNEL.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NI60_DROPOUT_APPLY4.patch
    comptime assert not is_defined["MOJOLEARN_NI60_DROPOUT_APPLY4"](), "removed 2026-11-15 (lane/expdel2-35-neural): NI60_DROPOUT_APPLY4 (opt-in arm, Neural (Dropout2d four-wide apply); no A/B recorded) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_NI60_DROPOUT_APPLY4.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NN48_CSR_TILES.patch
    comptime assert not is_defined["MOJOLEARN_NN48_CSR_TILES"](), "removed 2026-11-15 (lane/expdel2-35-neural): NN48_CSR_TILES (opt-in arm, Neural (GCN spmm CSR tiles); no A/B recorded) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_NN48_CSR_TILES.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_XCNN_NO_BN_BLOCK.patch
    comptime assert not is_defined["MOJOLEARN_XCNN_NO_BN_BLOCK"](), "removed 2026-11-15 (lane/expdel2-35-neural): XCNN_NO_BN_BLOCK (opt-in arm, Neural (x_cnn BatchNorm block statistics, before arm); no A/B recorded) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_XCNN_NO_BN_BLOCK.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_XCNN_NO_DIRECT_CONV.patch
    comptime assert not is_defined["MOJOLEARN_XCNN_NO_DIRECT_CONV"](), "removed 2026-11-15 (lane/expdel2-35-neural): XCNN_NO_DIRECT_CONV (opt-in arm, Neural (x_cnn direct first-layer conv, before arm); no A/B recorded) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_XCNN_NO_DIRECT_CONV.patch"
    # MOJOLEARN_XCNN_NO_FAST_TUNE: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_XCNN_NO_IM2COL_TAPS.patch
    comptime assert not is_defined["MOJOLEARN_XCNN_NO_IM2COL_TAPS"](), "removed 2026-11-15 (lane/expdel2-35-neural): XCNN_NO_IM2COL_TAPS (opt-in arm, Neural (x_cnn im2col per (row, channel), before arm); no A/B recorded) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_XCNN_NO_IM2COL_TAPS.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_XCNN_NO_MMA_SPLIT.patch
    comptime assert not is_defined["MOJOLEARN_XCNN_NO_MMA_SPLIT"](), "removed 2026-11-15 (lane/expdel2-35-neural): XCNN_NO_MMA_SPLIT (opt-in arm, Neural (x_cnn Apple MMA split plans, before arm); no A/B recorded) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_XCNN_NO_MMA_SPLIT.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_XCNN_NO_NT_TUNE.patch
    comptime assert not is_defined["MOJOLEARN_XCNN_NO_NT_TUNE"](), "removed 2026-11-15 (lane/expdel2-35-neural): XCNN_NO_NT_TUNE (opt-in arm, Neural (x_cnn Apple IDENTICAL NT plan timing, before arm); no A/B recorded) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_XCNN_NO_NT_TUNE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_XCNN_NO_RES_POOL.patch
    comptime assert not is_defined["MOJOLEARN_XCNN_NO_RES_POOL"](), "removed 2026-11-15 (lane/expdel2-35-neural): XCNN_NO_RES_POOL (opt-in arm, Neural (x_cnn resident reuse pool, before arm); no A/B recorded) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_XCNN_NO_RES_POOL.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_XCNN_NO_TILED_LAYOUT.patch
    comptime assert not is_defined["MOJOLEARN_XCNN_NO_TILED_LAYOUT"](), "removed 2026-11-15 (lane/expdel2-35-neural): XCNN_NO_TILED_LAYOUT (opt-in arm, Neural (x_cnn tiled NCHW layout, before arm); no A/B recorded) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_XCNN_NO_TILED_LAYOUT.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_XCNN_ONES_CACHE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_XCNN_ONES_CACHE_OFF"](), "removed 2026-11-15 (lane/expdel2-35-neural): XCNN_ONES_CACHE_OFF (old path, Neural (x_cnn resident ones vector); no A/B recorded) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_XCNN_ONES_CACHE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_XCNN_RES_FREE_ASYNC_OFF.patch
    comptime assert not is_defined["MOJOLEARN_XCNN_RES_FREE_ASYNC_OFF"](), "removed 2026-11-15 (lane/expdel2-35-neural): XCNN_RES_FREE_ASYNC_OFF (old path, Neural (x_cnn res_free queue wait); no A/B recorded) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_XCNN_RES_FREE_ASYNC_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_XCNN_RES_NEW_NOZERO_OFF.patch
    comptime assert not is_defined["MOJOLEARN_XCNN_RES_NEW_NOZERO_OFF"](), "removed 2026-11-15 (lane/expdel2-35-neural): XCNN_RES_NEW_NOZERO_OFF (old path, Neural (x_cnn unfilled resident outputs); no A/B recorded) is gone; the default path is the only one; code recoverable at 145be10898; restore: experiments/removed/MOJOLEARN_XCNN_RES_NEW_NOZERO_OFF.patch"
    # MOJOLEARN_XCNN_TILED_ROWS: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # ---- end lane expdel2-35-neural
    # Lane expdel2-34-neural (2026-11-15): unmeasured arms and superseded old paths deleted, docs/plans/EXPERIMENT_DELETIONS_V2_2026-11-15.md (docs/TOMBSTONES.md).
    # MOJOLEARN_ATTN_AMMA_ADMIT: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_ATTN_APPLE_ERECOMP: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_ATTN_APPLE_ESTASH_FORCE_DENY.patch
    comptime assert not is_defined["MOJOLEARN_ATTN_APPLE_ESTASH_FORCE_DENY"](), "removed 2026-11-15 (lane/expdel2-34-neural): ATTN_APPLE_ESTASH_FORCE_DENY (unmeasured Apple opt-in arm) is gone; the default is kept; code at 9125339d58; restore: experiments/removed/MOJOLEARN_ATTN_APPLE_ESTASH_FORCE_DENY.patch"
    # MOJOLEARN_ATTN_DKDV_AMMA: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_ATTN_DKDV_BJ16.patch
    comptime assert not is_defined["MOJOLEARN_ATTN_DKDV_BJ16"](), "removed 2026-11-15 (lane/expdel2-34-neural): ATTN_DKDV_BJ16 (unmeasured opt-in arm) is gone; the default is kept; code at 6955f6e721; restore: experiments/removed/MOJOLEARN_ATTN_DKDV_BJ16.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ATTN_DKDV_BJ32.patch
    comptime assert not is_defined["MOJOLEARN_ATTN_DKDV_BJ32"](), "removed 2026-11-15 (lane/expdel2-34-neural): ATTN_DKDV_BJ32 (unmeasured opt-in arm) is gone; the default is kept; code at a5bc5b1913; restore: experiments/removed/MOJOLEARN_ATTN_DKDV_BJ32.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ATTN_DKDV_MFMA.patch
    comptime assert not is_defined["MOJOLEARN_ATTN_DKDV_MFMA"](), "removed 2026-11-15 (lane/expdel2-34-neural): ATTN_DKDV_MFMA (unmeasured opt-in arm) is gone; the default is kept; code at 0447fb7252; restore: experiments/removed/MOJOLEARN_ATTN_DKDV_MFMA.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ATTN_DKDV_NO_T4.patch
    comptime assert not is_defined["MOJOLEARN_ATTN_DKDV_NO_T4"](), "removed 2026-11-15 (lane/expdel2-34-neural): ATTN_DKDV_NO_T4 (unmeasured opt-in arm) is gone; the default is kept; code at 6ea086845c; restore: experiments/removed/MOJOLEARN_ATTN_DKDV_NO_T4.patch"
    # MOJOLEARN_ATTN_DKDV_T4: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_ATTN_DQ_MFMA.patch
    comptime assert not is_defined["MOJOLEARN_ATTN_DQ_MFMA"](), "removed 2026-11-15 (lane/expdel2-34-neural): ATTN_DQ_MFMA (unmeasured opt-in arm) is gone; the default is kept; code at 9ed7579367; restore: experiments/removed/MOJOLEARN_ATTN_DQ_MFMA.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ATTN_DQ_TQ16.patch
    comptime assert not is_defined["MOJOLEARN_ATTN_DQ_TQ16"](), "removed 2026-11-15 (lane/expdel2-34-neural): ATTN_DQ_TQ16 (unmeasured opt-in arm) is gone; the default is kept; code at 347eba6c25; restore: experiments/removed/MOJOLEARN_ATTN_DQ_TQ16.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ATTN_DQ_TQ32.patch
    comptime assert not is_defined["MOJOLEARN_ATTN_DQ_TQ32"](), "removed 2026-11-15 (lane/expdel2-34-neural): ATTN_DQ_TQ32 (unmeasured opt-in arm) is gone; the default is kept; code at 8b3e85074f; restore: experiments/removed/MOJOLEARN_ATTN_DQ_TQ32.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ATTN_DQ_TQ64.patch
    comptime assert not is_defined["MOJOLEARN_ATTN_DQ_TQ64"](), "removed 2026-11-15 (lane/expdel2-34-neural): ATTN_DQ_TQ64 (unmeasured opt-in arm) is gone; the default is kept; code at 1f621f3a80; restore: experiments/removed/MOJOLEARN_ATTN_DQ_TQ64.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ATTN_EXACT_TAIL_GUARD.patch
    comptime assert not is_defined["MOJOLEARN_ATTN_EXACT_TAIL_GUARD"](), "removed 2026-11-15 (lane/expdel2-34-neural): ATTN_EXACT_TAIL_GUARD (unmeasured opt-in arm) is gone; the default is kept; code at c7f24015f7; restore: experiments/removed/MOJOLEARN_ATTN_EXACT_TAIL_GUARD.patch"
    # MOJOLEARN_ATTN_FWD_AMMA_P1_BARRIER: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_ATTN_FWD_AMMA_P2_BARRIER: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_ATTN_FWD_TQ16.patch
    comptime assert not is_defined["MOJOLEARN_ATTN_FWD_TQ16"](), "removed 2026-11-15 (lane/expdel2-34-neural): ATTN_FWD_TQ16 (unmeasured opt-in arm) is gone; the default is kept; code at 9a253b44b2; restore: experiments/removed/MOJOLEARN_ATTN_FWD_TQ16.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ATTN_FWD_TQ_RULE.patch
    comptime assert not is_defined["MOJOLEARN_ATTN_FWD_TQ_RULE"](), "removed 2026-11-15 (lane/expdel2-34-neural): ATTN_FWD_TQ_RULE (unmeasured opt-in arm) is gone; the default is kept; code at dd89e16295; restore: experiments/removed/MOJOLEARN_ATTN_FWD_TQ_RULE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ATTN_KV_CORNER_GUARD.patch
    comptime assert not is_defined["MOJOLEARN_ATTN_KV_CORNER_GUARD"](), "removed 2026-11-15 (lane/expdel2-34-neural): ATTN_KV_CORNER_GUARD (unmeasured opt-in arm) is gone; the default is kept; code at 7a7282ec56; restore: experiments/removed/MOJOLEARN_ATTN_KV_CORNER_GUARD.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ATTN_LEGACY_CORNER.patch
    comptime assert not is_defined["MOJOLEARN_ATTN_LEGACY_CORNER"](), "removed 2026-11-15 (lane/expdel2-34-neural): ATTN_LEGACY_CORNER (unmeasured old path) is gone; the default is kept; code at 8eda4c8a03; restore: experiments/removed/MOJOLEARN_ATTN_LEGACY_CORNER.patch"
    # MOJOLEARN_ATTN_NO_APPLE_MMA: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_ATTN_NO_BWD_CORNER.patch
    comptime assert not is_defined["MOJOLEARN_ATTN_NO_BWD_CORNER"](), "removed 2026-11-15 (lane/expdel2-34-neural): ATTN_NO_BWD_CORNER (unmeasured opt-in arm) is gone; the default is kept; code at a7bd96b41b; restore: experiments/removed/MOJOLEARN_ATTN_NO_BWD_CORNER.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ATTN_NO_MODE_FLUSH.patch
    comptime assert not is_defined["MOJOLEARN_ATTN_NO_MODE_FLUSH"](), "removed 2026-11-15 (lane/expdel2-34-neural): ATTN_NO_MODE_FLUSH (unmeasured opt-in arm) is gone; the default is kept; code at 4eb171e657; restore: experiments/removed/MOJOLEARN_ATTN_NO_MODE_FLUSH.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ATTN_NO_SCRATCH_CACHE.patch
    comptime assert not is_defined["MOJOLEARN_ATTN_NO_SCRATCH_CACHE"](), "removed 2026-11-15 (lane/expdel2-34-neural): ATTN_NO_SCRATCH_CACHE (unmeasured opt-in arm) is gone; the default is kept; code at 5c0ca62e33; restore: experiments/removed/MOJOLEARN_ATTN_NO_SCRATCH_CACHE.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ATTN_REPAIR_MASKED_TAIL.patch
    comptime assert not is_defined["MOJOLEARN_ATTN_REPAIR_MASKED_TAIL"](), "removed 2026-11-15 (lane/expdel2-34-neural): ATTN_REPAIR_MASKED_TAIL (unmeasured opt-in arm) is gone; the default is kept; code at c3d1c7865f; restore: experiments/removed/MOJOLEARN_ATTN_REPAIR_MASKED_TAIL.patch"
    # MOJOLEARN_ATTN_STASH_ZDOT_AMMA: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_ATTN_STICKY.patch
    comptime assert not is_defined["MOJOLEARN_ATTN_STICKY"](), "removed 2026-11-15 (lane/expdel2-34-neural): ATTN_STICKY (unmeasured opt-in arm) is gone; the default is kept; code at 99987befbf; restore: experiments/removed/MOJOLEARN_ATTN_STICKY.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_ATTN_TILE_RULE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_ATTN_TILE_RULE_OFF"](), "removed 2026-11-15 (lane/expdel2-34-neural): ATTN_TILE_RULE_OFF (unmeasured old path) is gone; the default is kept; code at 3a02a72be4; restore: experiments/removed/MOJOLEARN_ATTN_TILE_RULE_OFF.patch"
    # MOJOLEARN_ATTN_ZDOT_AMMA_FOLD_BARRIER: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # MOJOLEARN_ATTN_ZDOT_STASH_FOLD_BARRIER: refusal removed 2026-11-15 (restored; Apple fast opt-in arm, default off, lane/restore-apple-fast-neural; see docs/TOMBSTONES.md Restored table)
    # Restore: git apply experiments/removed/MOJOLEARN_GAPLM_ZDOT_KREG.patch
    comptime assert not is_defined["MOJOLEARN_GAPLM_ZDOT_KREG"](), "removed 2026-11-15 (lane/expdel2-34-neural): GAPLM_ZDOT_KREG (unmeasured opt-in arm) is gone; the default is kept; code at 62df7ffb39; restore: experiments/removed/MOJOLEARN_GAPLM_ZDOT_KREG.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ATTN_BWD_SCAN_REUSE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ATTN_BWD_SCAN_REUSE_OFF"](), "removed 2026-11-15 (lane/expdel2-34-neural): IDN_ATTN_BWD_SCAN_REUSE_OFF (unmeasured old path) is gone; the default is kept; code at f0d7af0f4d; restore: experiments/removed/MOJOLEARN_IDN_ATTN_BWD_SCAN_REUSE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ATTN_ONE_FLAG_WAIT_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ATTN_ONE_FLAG_WAIT_OFF"](), "removed 2026-11-15 (lane/expdel2-34-neural): IDN_ATTN_ONE_FLAG_WAIT_OFF (unmeasured old path) is gone; the default is kept; code at e348b403be; restore: experiments/removed/MOJOLEARN_IDN_ATTN_ONE_FLAG_WAIT_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ATTN_SCAN_CACHE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ATTN_SCAN_CACHE_OFF"](), "removed 2026-11-15 (lane/expdel2-34-neural): IDN_ATTN_SCAN_CACHE_OFF (unmeasured old path) is gone; the default is kept; code at f8340c9fd3; restore: experiments/removed/MOJOLEARN_IDN_ATTN_SCAN_CACHE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ATTN_SCRATCH_CACHE_OFF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ATTN_SCRATCH_CACHE_OFF"](), "removed 2026-11-15 (lane/expdel2-34-neural): IDN_ATTN_SCRATCH_CACHE_OFF (unmeasured old path) is gone; the default is kept; code at 5f78ca38ed; restore: experiments/removed/MOJOLEARN_IDN_ATTN_SCRATCH_CACHE_OFF.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_ATTN_TILE_ORDER.patch
    comptime assert not is_defined["MOJOLEARN_IDN_ATTN_TILE_ORDER"](), "removed 2026-11-15 (lane/expdel2-34-neural): IDN_ATTN_TILE_ORDER (unmeasured grid arms) is gone; the default is kept; code at 85e5bc6d2e; restore: experiments/removed/MOJOLEARN_IDN_ATTN_TILE_ORDER.patch"
    # Restore: git apply experiments/removed/MOJOLEARN_NN18_DENSE_TILE_CONTROL.patch
    comptime assert not is_defined["MOJOLEARN_NN18_DENSE_TILE_CONTROL"](), "removed 2026-11-15 (lane/expdel2-34-neural): NN18_DENSE_TILE_CONTROL (unmeasured opt-in arm) is gone; the default is kept; code at 3e8269e098; restore: experiments/removed/MOJOLEARN_NN18_DENSE_TILE_CONTROL.patch"
    # Lane expdel2-37-series (2026-11-15): unmeasured arms and old paths deleted under "An unmeasured arm is deleted" (docs/plans/EXPERIMENT_DELETIONS_V2_2026-11-15.md, docs/TOMBSTONES.md).
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_GEMM_LEAF.patch
    comptime assert not is_defined["MOJOLEARN_IDN_GEMM_LEAF"](), "removed 2026-11-15 (lane/expdel2-37-series): IDN_GEMM_LEAF (opt-in arm, Neural (build-wide GEMM leaf arms 1-3); no A/B recorded) is gone; the default path is the only one; code recoverable at 27caa1d5fe; restore: experiments/removed/MOJOLEARN_IDN_GEMM_LEAF.patch"
    # ---- end lane expdel2-37-series
    # ---- lane fx-verdicts2 (2026-11-18): A/B verdicts of ab3 (NVIDIA L40S, wheel d652a4b0a1)
    # TOMBSTONE: MOJOLEARN_DOWNLOAD_STAGE_OFF (LOSER, A/B ab3 NVIDIA L40S, wheel d652a4b0a1, job nvc-ab-20261118T182558Z-cb6e02, one run per arm, digests equal) deleted 2026-11-18 by lane/fx-verdicts2; code recoverable at 85c36e324f.
    # Restore: git apply experiments/removed/MOJOLEARN_DOWNLOAD_STAGE_OFF.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_DOWNLOAD_STAGE_OFF"](), "removed 2026-11-18 (lane/fx-verdicts2, A/B ab3 NVIDIA L40S, wheel d652a4b0a1, job nvc-ab-20261118T182558Z-cb6e02, one run per arm, digests equal): the raw host-pointer download at every size was arm/base 1.016 geomean over 7 neural cells (lm-forward 1.13); the pinned stage from DOWNLOAD_STAGE_MIN floats is the route; recoverable at 85c36e324f"
    # TOMBSTONE: MOJOLEARN_MAMBA3_LEGACY_SETUP (LOSER, A/B ab3 NVIDIA L40S, wheel d652a4b0a1, job nvc-ab-20261118T182558Z-cb6e02, one run per arm, digests equal) deleted 2026-11-18 by lane/fx-verdicts2; code recoverable at 85c36e324f.
    # Restore: git apply experiments/removed/MOJOLEARN_MAMBA3_LEGACY_SETUP.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_MAMBA3_LEGACY_SETUP"](), "removed 2026-11-18 (lane/fx-verdicts2, A/B ab3 NVIDIA L40S, wheel d652a4b0a1, job nvc-ab-20261118T182558Z-cb6e02, one run per arm, digests equal): Mamba3 per-call setup without prefill sessions was arm/base 1.007 geomean over 5 cells (noise); sessions wherever the binding offers them; recoverable at 85c36e324f"
    # TOMBSTONE: MOJOLEARN_MAMBA3_RETAIN_WEIGHTS_OFF (LOSER, A/B ab3 NVIDIA L40S, wheel d652a4b0a1, job nvc-ab-20261118T182558Z-cb6e02, one run per arm, digests equal) deleted 2026-11-18 by lane/fx-verdicts2; code recoverable at 85c36e324f.
    # Restore: git apply experiments/removed/MOJOLEARN_MAMBA3_RETAIN_WEIGHTS_OFF.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_MAMBA3_RETAIN_WEIGHTS_OFF"](), "removed 2026-11-18 (lane/fx-verdicts2, A/B ab3 NVIDIA L40S, wheel d652a4b0a1, job nvc-ab-20261118T182558Z-cb6e02, one run per arm, digests equal): the per-call Mamba3 weight upload was arm/base 1.030 geomean over 5 cells (mamba3-forward 1.14); the session keeps its weights; recoverable at 85c36e324f"
    # TOMBSTONE: MOJOLEARN_OPTIMIZER_RESIDENT_OFF (LOSER, A/B ab3 NVIDIA L40S, wheel d652a4b0a1, job nvc-ab-20261118T182558Z-cb6e02, one run per arm, digests equal) deleted 2026-11-18 by lane/fx-verdicts2; code recoverable at 85c36e324f.
    # Restore: git apply experiments/removed/MOJOLEARN_OPTIMIZER_RESIDENT_OFF.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_OPTIMIZER_RESIDENT_OFF"](), "removed 2026-11-18 (lane/fx-verdicts2, A/B ab3 NVIDIA L40S, wheel d652a4b0a1, job nvc-ab-20261118T182558Z-cb6e02, one run per arm, digests equal): the per-call optimizer step was arm/base 1.129 geomean over 5 cells (samba-train-step 1.97); resident optimizer state is the single-device route; recoverable at 85c36e324f"
    # ---- end lane fx-verdicts2
    # ---- lane fx-verdicts3 (2026-11-19): A/B verdicts of round 1 (NVIDIA L40S, wheel 0c68d6cd27, job nvc-ab-20261119T125554Z-6f7afc)
    # MOJOLEARN_GEMM_FAST_SEQ_PACKED (sequence/gemm_packed_route.mojo): promoted to the default on the NVIDIA and AMD
    # columns; the old opt-in name is refused so a stale build line does not silently mean the default.
    comptime assert not is_defined["MOJOLEARN_GEMM_FAST_SEQ_PACKED"](), "promoted 2026-11-19 (lane/fx-verdicts3, A/B round 1 NVIDIA L40S, wheel 0c68d6cd27, job nvc-ab-20261119T125554Z-6f7afc, one run per arm, digests and quality equal on 8 cells, mlp-clf/mlp-reg fit 0.59-0.66, infer 0.43-0.68): large plain dense sequence GEMMs run the neural packed body with one leaf; drop -D MOJOLEARN_GEMM_FAST_SEQ_PACKED, use -D MOJOLEARN_GEMM_FAST_SEQ_PACKED_OFF for the 16 x 16 tile"
    # MOJOLEARN_LMHEAD_FAST_GEMM_PANEL (training/chunked_lm_head_panel_math.mojo): promoted to the profile of the
    # chunked LM head on every column and the host column (bits change together); the old opt-in name is refused.
    comptime assert not is_defined["MOJOLEARN_LMHEAD_FAST_GEMM_PANEL"](), "promoted 2026-11-19 (lane/fx-verdicts3, A/B round 1 NVIDIA L40S, wheel 0c68d6cd27, job nvc-ab-20261119T125554Z-6f7afc, one run per arm, train-chunked-lm-head-loss 33,016.9 -> 284.5 ms, FP64 differences smaller on loss and both gradients; bits change on every column together): GEMM panels and tree folds are the chunked LM head profile; drop -D MOJOLEARN_LMHEAD_FAST_GEMM_PANEL, use -D MOJOLEARN_LMHEAD_FAST_GEMM_PANEL_OFF for the scalar serial profile"
    # MOJOLEARN_MAMBA_FAST_DECODE_STAGE_ARENA (autograd/block_step_ops.mojo): promoted to the default; the old opt-in
    # name is refused so a stale build line does not silently mean the default.
    comptime assert not is_defined["MOJOLEARN_MAMBA_FAST_DECODE_STAGE_ARENA"](), "promoted 2026-11-19 (lane/fx-verdicts3, A/B round 1 NVIDIA L40S, wheel 0c68d6cd27, job nvc-ab-20261119T125554Z-6f7afc, one run per arm, digests equal on 6 cells, mamba3-decode 0.80, mamba2-decode 0.84, control mamba1-decode 0.92): a Mamba-2 / Mamba-3 decode token takes its stages as views of one held arena; drop -D MOJOLEARN_MAMBA_FAST_DECODE_STAGE_ARENA, use -D MOJOLEARN_MAMBA_FAST_DECODE_STAGE_ARENA_OFF for per-token stages"
    # MOJOLEARN_MAMBA3_FAST_S17_TAIL_TREE (autograd/kernels/mamba3.mojo and the generated host column): promoted to
    # the default on every column and the host column (bits change together); the old opt-in name is refused.
    comptime assert not is_defined["MOJOLEARN_MAMBA3_FAST_S17_TAIL_TREE"](), "promoted 2026-11-19 (lane/fx-verdicts3, A/B round 1 NVIDIA L40S, wheel 0c68d6cd27, job nvc-ab-20261119T125554Z-6f7afc, one run per arm, samba-train-step 50.37 -> 38.33 ms, loss equal to one float32 step; bits change on every column together): the Mamba-3 backward S17 chunk-end tail is 128 lane chains and the contract 7.2 tree; drop -D MOJOLEARN_MAMBA3_FAST_S17_TAIL_TREE, use -D MOJOLEARN_MAMBA3_FAST_S17_TAIL_TREE_OFF for the serial chain"
    # MOJOLEARN_LLAMA_FAST_REFUSE_WATERMARK (transformer/impl/llama/modeling_llama.mojo): promoted to the IDENTICAL
    # default; the old opt-in name is refused so a stale build line does not silently mean the default.
    comptime assert not is_defined["MOJOLEARN_LLAMA_FAST_REFUSE_WATERMARK"](), "promoted 2026-11-19 (lane/fx-verdicts3, A/B round 1 NVIDIA L40S, wheel 0c68d6cd27, job nvc-ab-20261119T125554Z-6f7afc, one run per arm, digests equal on 4 cells, four decode rows 0.91-0.94): a decode step of a granted cache scans the hidden states and the one appended key and value row; drop -D MOJOLEARN_LLAMA_FAST_REFUSE_WATERMARK, use -D MOJOLEARN_LLAMA_FAST_REFUSE_WATERMARK_OFF for the whole scans"
    # TOMBSTONE: MOJOLEARN_XCNN_FAST_TYPED_DOOR (NOISE, A/B round 1 NVIDIA L40S, wheel 0c68d6cd27, job nvc-ab-20261119T125554Z-6f7afc, one run per arm, digests equal) deleted 2026-11-19 by lane/fx-verdicts3; code recoverable at f96f23b7fb.
    # Restore: git apply experiments/removed/MOJOLEARN_XCNN_FAST_TYPED_DOOR.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_XCNN_FAST_TYPED_DOOR"](), "removed 2026-11-19 (lane/fx-verdicts3, A/B round 1 NVIDIA L40S, wheel 0c68d6cd27, job nvc-ab-20261119T125554Z-6f7afc, one run per arm, digests equal): the typed pool door was arm/base 0.92-1.08 over 12 pool cells, ten within 1 % (noise; no speed case for the typed boundary tables on these rows); the door is the entry; recoverable at f96f23b7fb"
    # TOMBSTONE: MOJOLEARN_GEMM_FAST_DX_CONCAT (NOISE, A/B round 1 NVIDIA L40S, wheel 0c68d6cd27, job nvc-ab-20261119T125554Z-6f7afc, one run per arm, quality equal) deleted 2026-11-19 by lane/fx-verdicts3; code recoverable at 913fc44cfa.
    # Restore: git apply experiments/removed/MOJOLEARN_GEMM_FAST_DX_CONCAT.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_GEMM_FAST_DX_CONCAT"](), "removed 2026-11-19 (lane/fx-verdicts3, A/B round 1 NVIDIA L40S, wheel 0c68d6cd27, job nvc-ab-20261119T125554Z-6f7afc, one run per arm, quality equal): the q/k/v fan-in dX as one concatenated product was arm/base 1.00 on lm-train-step and 0.99 on samba-train-step (noise) and changed bits; three products and (q + k) + v are the route; recoverable at 913fc44cfa"
    # ---- end lane fx-verdicts3
    # ---- lane fx-arm-cleanup (2026-11-19): the three A/B round 1 noise arms the verdict lane left in the tree
    # TOMBSTONE: MOJOLEARN_LLAMA_FAST_KV_STRIDE_CAP (NOISE, A/B round 1 NVIDIA L40S, wheel 0c68d6cd27, job nvc-ab-20261119T125554Z-6f7afc, one run per arm, ids, outputs and digests equal on 4 cells) deleted 2026-11-19 by lane/fx-arm-cleanup; code recoverable at a7c89e7558.
    # Restore: git apply experiments/removed/MOJOLEARN_LLAMA_FAST_KV_STRIDE_CAP.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_LLAMA_FAST_KV_STRIDE_CAP"](), "removed 2026-11-19 (lane/fx-arm-cleanup, A/B round 1 NVIDIA L40S, wheel 0c68d6cd27, job nvc-ab-20261119T125554Z-6f7afc, one run per arm, digests equal): the full-causal KV cache at stride max_tokens appended in place was arm/base 0.90 on causal-lm-generate from one pair with 3-8 percent drift and 1.00, 0.98, 1.02 on the other three decode rows (noise); the cache packed at stride cached_tokens is the layout; recoverable at a7c89e7558"
    # TOMBSTONE: MOJOLEARN_GEMM_FAST_WORKLIST (NOISE, A/B round 1 NVIDIA L40S, wheel 0c68d6cd27, job nvc-ab-20261119T125554Z-6f7afc, one run per arm, digests and quality equal) deleted 2026-11-19 by lane/fx-arm-cleanup; code recoverable at a7c89e7558.
    # Restore: git apply experiments/removed/MOJOLEARN_GEMM_FAST_WORKLIST.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_GEMM_FAST_WORKLIST"](), "removed 2026-11-19 (lane/fx-arm-cleanup, A/B round 1 NVIDIA L40S, wheel 0c68d6cd27, job nvc-ab-20261119T125554Z-6f7afc, one run per arm, digests and quality equal): independent GEMMs as one work-list launch of the packed body was arm/base 1.00 on lm-train-step, 1.00 on transformer-decode and 0.93 once on lm-forward (another base run of the row; noise); one launch per product is the route; recoverable at a7c89e7558"
    # TOMBSTONE: MOJOLEARN_GEMM_FAST_EPI, its transformer-backward use (NOISE, A/B round 1 NVIDIA L40S, wheel 0c68d6cd27, job nvc-ab-20261119T125554Z-6f7afc, one run per arm, digests and quality equal: lm-train-step 0.98 = the row's other bases, samba-train-step 1.05) deleted 2026-11-19 by lane/fx-arm-cleanup; code recoverable at a7c89e7558.
    # Restore: git apply experiments/removed/MOJOLEARN_GEMM_FAST_EPI-transformer-backward.patch; record in docs/TOMBSTONES.md.
    # NOT refused: the define still names the fused follower store of the packed sequence route (gemm/neural_dispatch.mojo
    # GEMM_LEAF_PTR_EPI, sequence/gemm_packed_route.mojo SEQ_GEMM_PACKED_EPI; owed row gemm-seq-packed-epi-mlp). On a
    # build with the define the transformer backward runs its default follower launches.
    # ---- end lane fx-arm-cleanup
    # ---- lane fx-old-arms-triage (2026-11-19): arms with no A/B in four weeks (AGENTS.md "An unmeasured arm is deleted")
    # TOMBSTONE: MOJOLEARN_CLIP_PACKED (opt-in arm, no A/B recorded in four weeks) deleted 2026-11-19 by lane/fx-old-arms-triage; code recoverable at a7c89e7558.
    # Restore: git apply experiments/removed/MOJOLEARN_CLIP_PACKED.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_CLIP_PACKED"](), "removed 2026-11-19 (lane/fx-old-arms-triage, opt-in arm, no A/B recorded in four weeks): the packed grad-norm clip on one optimizer device (run-time switch since 2026-09-30, build define since 2026-11-15) had no A/B in four weeks; the in-place clip stays, the packed buffer stays for several optimizer devices; code at a7c89e7558; restore: experiments/removed/MOJOLEARN_CLIP_PACKED.patch"
    # TOMBSTONE: MOJOLEARN_DEVICE_ARENA (NVIDIA/AMD opt-in arm, no A/B recorded in four weeks) deleted 2026-11-19 by lane/fx-old-arms-triage; code recoverable at a7c89e7558.
    # Restore: git apply experiments/removed/MOJOLEARN_DEVICE_ARENA.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_DEVICE_ARENA"](), "removed 2026-11-19 (lane/fx-old-arms-triage, NVIDIA/AMD opt-in arm, no A/B recorded in four weeks): session arenas on NVIDIA/AMD (run-time switch since 2026-10-01, build define since 2026-11-15) had no A/B in four weeks; arenas stay on the Apple column, -D MOJOLEARN_DEVICE_ARENA_OFF stays; code at a7c89e7558; restore: experiments/removed/MOJOLEARN_DEVICE_ARENA.patch"
    # TOMBSTONE: MOJOLEARN_SWIGLU_FUSED (opt-in arm, no A/B recorded in four weeks) deleted 2026-11-19 by lane/fx-old-arms-triage; code recoverable at a7c89e7558.
    # Restore: git apply experiments/removed/MOJOLEARN_SWIGLU_FUSED.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_SWIGLU_FUSED"](), "removed 2026-11-19 (lane/fx-old-arms-triage, opt-in arm, no A/B recorded in four weeks): the forward-only fused SiLU*gate launch (run-time switch since 2026-09-30, build define since 2026-11-15) had no A/B in four weeks; the S20/S21 launches stay; code at a7c89e7558; restore: experiments/removed/MOJOLEARN_SWIGLU_FUSED.patch"
    # TOMBSTONE: MOJOLEARN_TRANSFORMER_NORM_DW_OWN_WS (old path, no A/B recorded in four weeks) deleted 2026-11-19 by lane/fx-old-arms-triage; code recoverable at a7c89e7558.
    # Restore: git apply experiments/removed/MOJOLEARN_TRANSFORMER_NORM_DW_OWN_WS.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_TRANSFORMER_NORM_DW_OWN_WS"](), "removed 2026-11-19 (lane/fx-old-arms-triage, old path, no A/B recorded in four weeks): the norm dW through identical_gemm (run-time switch since 2026-10-20, build define since 2026-11-15) had no A/B in four weeks; the routed workspace.run stays; code at a7c89e7558; restore: experiments/removed/MOJOLEARN_TRANSFORMER_NORM_DW_OWN_WS.patch"
    # TOMBSTONE: MOJOLEARN_BYTE_LM_KEEP_CONTEXT (opt-in arm, no A/B recorded in four weeks) deleted 2026-11-19 by lane/fx-old-arms-triage; code recoverable at a7c89e7558.
    # Restore: git apply experiments/removed/MOJOLEARN_BYTE_LM_KEEP_CONTEXT.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_BYTE_LM_KEEP_CONTEXT"](), "removed 2026-11-19 (lane/fx-old-arms-triage, opt-in arm, no A/B recorded in four weeks): the byte LM keeper context (run-time switch since 2026-10-21, build define since 2026-11-15) had no A/B in four weeks; the process-lifetime neural context stays; code at a7c89e7558; restore: experiments/removed/MOJOLEARN_BYTE_LM_KEEP_CONTEXT.patch"
    # ---- end lane fx-old-arms-triage
    # ---- lane fx-verdicts4 (2026-11-19): verdicts of A/B round 2 (NVIDIA L40S, wheel 0c68d6cd27, job nvc-ab-20261119T154232Z-0642a5)
    # TOMBSTONE: MOJOLEARN_BYTE_LM_FAST_LOGITS_HUGEPAGE (LOSS, A/B round 2 NVIDIA L40S, wheel 0c68d6cd27, job nvc-ab-20261119T154232Z-0642a5, one run per arm, digests equal) deleted 2026-11-19 by lane/fx-verdicts4; code recoverable at 8f3f639b13.
    # Restore: git apply experiments/removed/MOJOLEARN_BYTE_LM_FAST_LOGITS_HUGEPAGE.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_BYTE_LM_FAST_LOGITS_HUGEPAGE"](), "removed 2026-11-19 (lane/fx-verdicts4, A/B round 2 NVIDIA L40S, wheel 0c68d6cd27, job nvc-ab-20261119T154232Z-0642a5, one run per arm, digests equal): the huge-page advice on a fresh logits store was arm/base 1.17 on lm-forward (21.96 -> 25.73 ms; loss); the unfilled small-page store stays; code at 8f3f639b13, experiments/removed/MOJOLEARN_BYTE_LM_FAST_LOGITS_HUGEPAGE.patch"
    # TOMBSTONE: MOJOLEARN_GEMM_FAST_LOADAHEAD, MOJOLEARN_GEMM_FAST_LOADAHEAD_STEPS (depth 2 NOISE in A/B round 1, depth 4 LOSS in A/B round 2; NVIDIA L40S, wheel 0c68d6cd27, jobs nvc-ab-20261119T125554Z-6f7afc and nvc-ab-20261119T154232Z-0642a5, digests equal) deleted 2026-11-19 by lane/fx-verdicts4; code recoverable at 8f3f639b13.
    # Restore: git apply experiments/removed/MOJOLEARN_GEMM_FAST_LOADAHEAD.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_GEMM_FAST_LOADAHEAD"](), "removed 2026-11-19 (lane/fx-verdicts4, A/B rounds 1 and 2 NVIDIA L40S, wheel 0c68d6cd27, one run per arm, digests equal): the packed GEMM accumulate with the operands of 2 or 4 steps loaded ahead was arm/base 0.97 gemm, 1.00 lm-train-step, 1.06 lm-forward at depth 2 (noise) and 1.00, 1.03, 1.40 at depth 4 (loss); the per-step loads stay; code at 8f3f639b13, experiments/removed/MOJOLEARN_GEMM_FAST_LOADAHEAD.patch"
    comptime assert not is_defined["MOJOLEARN_GEMM_FAST_LOADAHEAD_STEPS"](), "removed 2026-11-19 (lane/fx-verdicts4, A/B rounds 1 and 2 NVIDIA L40S, wheel 0c68d6cd27, one run per arm, digests equal): the packed GEMM accumulate with the operands of 2 or 4 steps loaded ahead was arm/base 0.97 gemm, 1.00 lm-train-step, 1.06 lm-forward at depth 2 (noise) and 1.00, 1.03, 1.40 at depth 4 (loss); the per-step loads stay; code at 8f3f639b13, experiments/removed/MOJOLEARN_GEMM_FAST_LOADAHEAD.patch"
    # TOMBSTONE: MOJOLEARN_SAMBA_FAST_STEP_SCRATCH_HELD (NOISE, A/B round 2 NVIDIA L40S, wheel 0c68d6cd27, job nvc-ab-20261119T154232Z-0642a5, one run per arm, digests equal) deleted 2026-11-19 by lane/fx-verdicts4; code recoverable at 8f3f639b13.
    # Restore: git apply experiments/removed/MOJOLEARN_SAMBA_FAST_STEP_SCRATCH_HELD.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_SAMBA_FAST_STEP_SCRATCH_HELD"](), "removed 2026-11-19 (lane/fx-verdicts4, A/B round 2 NVIDIA L40S, wheel 0c68d6cd27, job nvc-ab-20261119T154232Z-0642a5, one run per arm, digests equal): the held Mamba-3 backward arena and one-fill forward state of the Samba resident step was arm/base 1.00 on samba-train-step (48.73 -> 48.57 ms; noise); code at 8f3f639b13, experiments/removed/MOJOLEARN_SAMBA_FAST_STEP_SCRATCH_HELD.patch"
    # MOJOLEARN_LLAMA_FAST_STAGE_KEEP (transformer/impl/llama/modeling_llama.mojo): promoted to the default for the
    # stateless block forward; its stack-layer use is deleted (TOMBSTONE in autograd/block_step_ops.mojo, patch
    # experiments/removed/MOJOLEARN_LLAMA_FAST_STAGE_KEEP-stack-layer.patch). The old opt-in name is refused.
    comptime assert not is_defined["MOJOLEARN_LLAMA_FAST_STAGE_KEEP"](), "promoted 2026-11-19 (lane/fx-verdicts4, A/B round 2 NVIDIA L40S, wheel 0c68d6cd27, job nvc-ab-20261119T154232Z-0642a5, one run per arm, digests equal on 9 cells, transformer-forward 0.953, TransformerBlock.forward at four driver shapes 0.87-0.97): a stateless forward-only block run keeps its stage bytes by default; its stack-layer use was 1.03 on samba-forward and is deleted. Remove -D MOJOLEARN_LLAMA_FAST_STAGE_KEEP; -D MOJOLEARN_LLAMA_FAST_STAGE_KEEP_OFF restores the reset before every call"
    # TOMBSTONE: MOJOLEARN_CNN_FAST_FLAT_UPDATE (NOISE, A/B round 2 NVIDIA L40S, wheel 0c68d6cd27, job nvc-ab-20261119T154232Z-0642a5, one run per arm, digests equal) deleted 2026-11-19 by lane/fx-verdicts4; code recoverable at 8f3f639b13.
    # Restore: git apply experiments/removed/MOJOLEARN_CNN_FAST_FLAT_UPDATE.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_CNN_FAST_FLAT_UPDATE"](), "removed 2026-11-19 (lane/fx-verdicts4, A/B round 2 NVIDIA L40S, wheel 0c68d6cd27, job nvc-ab-20261119T154232Z-0642a5, one run per arm, digests equal): Program.fit's flat one-launch optimizer update was arm/base 0.97 on cnn-clf fit (187.1 -> 181.4 ms) where the row's two bases are 12 % apart (noise); code at 8f3f639b13, experiments/removed/MOJOLEARN_CNN_FAST_FLAT_UPDATE.patch"
    # MOJOLEARN_SEQ_FAST_RNN_STEP_GEMM (autograd/rules_recurrent.mojo): split by site. The forward step's GEMM launch is
    # the IDENTICAL default (RNN_STEP_GEMM, _OFF restores the fused launch); the backward step and the tile-routed weight
    # gradients are the opt-in arm MOJOLEARN_SEQ_FAST_RNN_BPTT_GEMM; the weight-gradient GEMM at split-K shapes is
    # deleted (TOMBSTONE beside the constant, experiments/removed/MOJOLEARN_SEQ_FAST_RNN_STEP_GEMM.patch). The old name
    # is refused so a stale build line does not silently mean one of the three.
    comptime assert not is_defined["MOJOLEARN_SEQ_FAST_RNN_STEP_GEMM"](), "split 2026-11-19 (lane/fx-verdicts4, A/B round 2 NVIDIA L40S, wheel 0c68d6cd27, job nvc-ab-20261119T154232Z-0642a5, one run per arm, quality equal): gru / lstm predict 0.35-0.39 (the forward step: now the default, -D MOJOLEARN_SEQ_FAST_RNN_STEP_GEMM_OFF restores the fused launch); fit 2.3-3.4x slower at hidden 64 from the weight-gradient GEMM at split-K shapes (deleted); the backward step and tile-routed weight gradients are -D MOJOLEARN_SEQ_FAST_RNN_BPTT_GEMM (default off, A/B owed). Remove -D MOJOLEARN_SEQ_FAST_RNN_STEP_GEMM"
    # ---- end lane fx-verdicts4
    # ---- lane fx-verdicts5 (2026-11-20): verdicts of A/B round 3 (NVIDIA L40S, wheel 28a780604c, job nvc-ab-20261120T063324Z-c630ad)
    # MOJOLEARN_CNN_FAST_OP_NOWAIT (x_cnn/device.mojo): promoted to the default; the old opt-in name is refused.
    comptime assert not is_defined["MOJOLEARN_CNN_FAST_OP_NOWAIT"](), "promoted 2026-11-20 (lane/fx-verdicts5, A/B round 3 NVIDIA L40S, wheel 28a780604c, job nvc-ab-20261120T063324Z-c630ad, one run per arm, cnn-clf fit 0.894, driver widths 0.887 and 0.886, digests equal): it is the default; drop the define, or build the old form with -D MOJOLEARN_CNN_FAST_OP_NOWAIT_OFF"
    # MOJOLEARN_SAMBA_FAST_STEP_REGISTRY_RESIDENT (core/weight_epoch.mojo): promoted to the default; the old opt-in name is refused.
    comptime assert not is_defined["MOJOLEARN_SAMBA_FAST_STEP_REGISTRY_RESIDENT"](), "promoted 2026-11-20 (lane/fx-verdicts5, A/B round 3 NVIDIA L40S, wheel 28a780604c, job nvc-ab-20261120T063324Z-c630ad, one run per arm, samba-train-step 0.965, driver steps 0.957, 0.961, 0.937, first forward after a step 0.79-0.87, digests equal): it is the default; drop the define, or build the old form with -D MOJOLEARN_SAMBA_FAST_STEP_REGISTRY_RESIDENT_OFF"
    # TOMBSTONE: MOJOLEARN_IDN_MAMBA_BACKWARD_GENERIC (NOISE, A/B round 3 NVIDIA L40S, wheel 28a780604c, job nvc-ab-20261120T063324Z-c630ad, one run per arm: driver, median of 20, Mamba1Block.backward 1.010, Mamba2 0.992; third noise reading) deleted 2026-11-20 by lane/fx-verdicts5; code recoverable at e25e35dab8.
    # Restore: git apply experiments/removed/MOJOLEARN_IDN_MAMBA_BACKWARD_GENERIC.fx-verdicts5.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_IDN_MAMBA_BACKWARD_GENERIC"](), "removed 2026-11-20 (lane/fx-verdicts5, A/B round 3 NVIDIA L40S, wheel 28a780604c, job nvc-ab-20261120T063324Z-c630ad, one run per arm; NOISE: driver, median of 20, Mamba1Block.backward 1.010, Mamba2 0.992; third noise reading): the tape graph as the block step VJP; the hand chain is the VJP; see docs/TOMBSTONES.md"
    # TOMBSTONE: MOJOLEARN_SEQ_POOL_OFF (NOISE, A/B round 3 NVIDIA L40S, wheel 28a780604c, job nvc-ab-20261120T063324Z-c630ad, one run per arm: sequential-fit 0.991, sequential-predict 1.023; round 2 0.98, 0.95) deleted 2026-11-20 by lane/fx-verdicts5; code recoverable at e25e35dab8.
    # Restore: git apply experiments/removed/MOJOLEARN_SEQ_POOL_OFF.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_SEQ_POOL_OFF"](), "removed 2026-11-20 (lane/fx-verdicts5, A/B round 3 NVIDIA L40S, wheel 28a780604c, job nvc-ab-20261120T063324Z-c630ad, one run per arm; NOISE: sequential-fit 0.991, sequential-predict 1.023; round 2 0.98, 0.95): fresh buffers for every shared sequence executor; the pool stays; see docs/TOMBSTONES.md"
    # TOMBSTONE: MOJOLEARN_SEQ_FAST_RNN_STEP_GEMM_OFF (old path, promotion confirmed twice, A/B round 3 NVIDIA L40S, wheel 28a780604c, job nvc-ab-20261120T063324Z-c630ad, one run per arm: gru and lstm fits 1.19-1.32, predicts 3.4-4.5 with the fused forward launch) deleted 2026-11-20 by lane/fx-verdicts5; code recoverable at e25e35dab8.
    # Restore: git apply experiments/removed/MOJOLEARN_SEQ_FAST_RNN_STEP_GEMM_OFF.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_SEQ_FAST_RNN_STEP_GEMM_OFF"](), "removed 2026-11-20 (lane/fx-verdicts5, A/B round 3 NVIDIA L40S, wheel 28a780604c, job nvc-ab-20261120T063324Z-c630ad, one run per arm; old path, promotion confirmed twice: gru and lstm fits 1.19-1.32, predicts 3.4-4.5 with the fused forward launch): the fused forward launch above the step rule; see docs/TOMBSTONES.md"
    # TOMBSTONE: MOJOLEARN_SEQ_FAST_RNN_BPTT_GEMM (SPLIT, A/B round 3 NVIDIA L40S, wheel 28a780604c, job nvc-ab-20261120T063324Z-c630ad, one run per arm: lstm fits 0.95-0.96, gru fits 1.05-1.07, quality equal) deleted 2026-11-20 by lane/fx-verdicts5; code recoverable at e25e35dab8.
    # Restore: git apply experiments/removed/MOJOLEARN_SEQ_FAST_RNN_BPTT_GEMM.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_SEQ_FAST_RNN_BPTT_GEMM"](), "removed 2026-11-20 (lane/fx-verdicts5, A/B round 3 NVIDIA L40S, wheel 28a780604c, job nvc-ab-20261120T063324Z-c630ad, one run per arm; SPLIT: lstm fits 0.95-0.96, gru fits 1.05-1.07, quality equal): the LSTM backward step GEMM is the default (old form: -D MOJOLEARN_SEQ_FAST_RNN_BPTT_LSTM_GEMM_OFF); the GRU backward step GEMM and the non-split weight-gradient GEMM are deleted; see docs/TOMBSTONES.md"
    # TOMBSTONE: MOJOLEARN_GEMM_FAST_SEQ_PACKED_OFF (old path, promotion confirmed twice, A/B round 3 NVIDIA L40S, wheel 28a780604c, job nvc-ab-20261120T063324Z-c630ad, one run per arm: driver MLP fit with the 16 x 16 tile 1.12, 1.67, 1.02, control 1.005) deleted 2026-11-20 by lane/fx-verdicts5; code recoverable at e25e35dab8.
    # Restore: git apply experiments/removed/MOJOLEARN_GEMM_FAST_SEQ_PACKED_OFF.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_GEMM_FAST_SEQ_PACKED_OFF"](), "removed 2026-11-20 (lane/fx-verdicts5, A/B round 3 NVIDIA L40S, wheel 28a780604c, job nvc-ab-20261120T063324Z-c630ad, one run per arm; old path, promotion confirmed twice: driver MLP fit with the 16 x 16 tile 1.12, 1.67, 1.02, control 1.005): the build switch of the sequence packed route; the tile stays for products the rule does not select; see docs/TOMBSTONES.md"
    # TOMBSTONE: MOJOLEARN_LLAMA_FAST_REFUSE_WATERMARK_OFF (old path, promotion confirmed twice, A/B round 3 NVIDIA L40S, wheel 28a780604c, job nvc-ab-20261120T063324Z-c630ad, one run per arm: four decode rows 1.013-1.030 with the whole scans, digests equal) deleted 2026-11-20 by lane/fx-verdicts5; code recoverable at e25e35dab8.
    # Restore: git apply experiments/removed/MOJOLEARN_LLAMA_FAST_REFUSE_WATERMARK_OFF.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_LLAMA_FAST_REFUSE_WATERMARK_OFF"](), "removed 2026-11-20 (lane/fx-verdicts5, A/B round 3 NVIDIA L40S, wheel 28a780604c, job nvc-ab-20261120T063324Z-c630ad, one run per arm; old path, promotion confirmed twice: four decode rows 1.013-1.030 with the whole scans, digests equal): the whole-cache refusal scans at every step of an IDENTICAL build; see docs/TOMBSTONES.md"
    # TOMBSTONE: MOJOLEARN_LLAMA_FAST_STAGE_KEEP_OFF (old path, promotion confirmed twice, A/B round 3 NVIDIA L40S, wheel 28a780604c, job nvc-ab-20261120T063324Z-c630ad, one run per arm: transformer-forward 1.108, driver shapes 1.00, 1.10, 1.06, 1.04 with the reset, digests equal) deleted 2026-11-20 by lane/fx-verdicts5; code recoverable at e25e35dab8.
    # Restore: git apply experiments/removed/MOJOLEARN_LLAMA_FAST_STAGE_KEEP_OFF.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_LLAMA_FAST_STAGE_KEEP_OFF"](), "removed 2026-11-20 (lane/fx-verdicts5, A/B round 3 NVIDIA L40S, wheel 28a780604c, job nvc-ab-20261120T063324Z-c630ad, one run per arm; old path, promotion confirmed twice: transformer-forward 1.108, driver shapes 1.00, 1.10, 1.06, 1.04 with the reset, digests equal): the stage reset before a stateless block forward on kept stages; see docs/TOMBSTONES.md"
    # TOMBSTONE: MOJOLEARN_MAMBA_FAST_DECODE_STAGE_ARENA_OFF (old path, promotion confirmed twice, A/B round 3 NVIDIA L40S, wheel 28a780604c, job nvc-ab-20261120T063324Z-c630ad, one run per arm: mamba3-decode 1.243, mamba2-decode 1.173, samba-decode 1.091 with per-token stages, control mamba1-decode 1.007) deleted 2026-11-20 by lane/fx-verdicts5; code recoverable at e25e35dab8.
    # Restore: git apply experiments/removed/MOJOLEARN_MAMBA_FAST_DECODE_STAGE_ARENA_OFF.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_MAMBA_FAST_DECODE_STAGE_ARENA_OFF"](), "removed 2026-11-20 (lane/fx-verdicts5, A/B round 3 NVIDIA L40S, wheel 28a780604c, job nvc-ab-20261120T063324Z-c630ad, one run per arm; old path, promotion confirmed twice: mamba3-decode 1.243, mamba2-decode 1.173, samba-decode 1.091 with per-token stages, control mamba1-decode 1.007): the build switch of the per-token Mamba decode stages; see docs/TOMBSTONES.md"
    # TOMBSTONE: MOJOLEARN_LMHEAD_FAST_GEMM_PANEL_OFF (old path, promotion confirmed twice, A/B round 3 NVIDIA L40S, wheel 28a780604c, job nvc-ab-20261120T063324Z-c630ad, one run per arm: train-chunked-lm-head-loss 277.9 -> 33,009 ms with the scalar profile) deleted 2026-11-20 by lane/fx-verdicts5; code recoverable at e25e35dab8.
    # Restore: git apply experiments/removed/MOJOLEARN_LMHEAD_FAST_GEMM_PANEL_OFF.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_LMHEAD_FAST_GEMM_PANEL_OFF"](), "removed 2026-11-20 (lane/fx-verdicts5, A/B round 3 NVIDIA L40S, wheel 28a780604c, job nvc-ab-20261120T063324Z-c630ad, one run per arm; old path, promotion confirmed twice: train-chunked-lm-head-loss 277.9 -> 33,009 ms with the scalar profile): the scalar serial LM-head profile; see docs/TOMBSTONES.md"
    # TOMBSTONE: MOJOLEARN_MAMBA3_FAST_S17_TAIL_TREE_OFF (old path, promotion confirmed twice, A/B round 3 NVIDIA L40S, wheel 28a780604c, job nvc-ab-20261120T063324Z-c630ad, one run per arm: driver SambaStack.train_step with the serial chain 1.48, 1.89, 1.32) deleted 2026-11-20 by lane/fx-verdicts5; code recoverable at e25e35dab8.
    # Restore: git apply experiments/removed/MOJOLEARN_MAMBA3_FAST_S17_TAIL_TREE_OFF.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_MAMBA3_FAST_S17_TAIL_TREE_OFF"](), "removed 2026-11-20 (lane/fx-verdicts5, A/B round 3 NVIDIA L40S, wheel 28a780604c, job nvc-ab-20261120T063324Z-c630ad, one run per arm; old path, promotion confirmed twice: driver SambaStack.train_step with the serial chain 1.48, 1.89, 1.32): the serial S17 chunk-end chain; see docs/TOMBSTONES.md"
    # ---- end lane fx-verdicts5
    # ---- lane fx-verdicts6 (2026-11-20): verdicts of A/B abf2 (NVIDIA L40S host 79f003b8e8ae, wheel e41b5148b24d, job nvc-ab-20261120T224801Z-458a59)
    # MOJOLEARN_GEMM_FAST_MATRIX_UNITS=<kind> (gemm/fast_matrix_units.mojo, checks/mode_policy.mojo): kind 3 (every neural-dispatch caller) is the default route of a fast-rows build on NVIDIA and AMD; the tiers are deleted (TOMBSTONE beside the route constant, experiments/removed/MOJOLEARN_GEMM_FAST_MATRIX_UNITS.tiers.patch). The old opt-in name is refused so a stale build line does not silently mean a tier.
    comptime assert not is_defined["MOJOLEARN_GEMM_FAST_MATRIX_UNITS"](), "promoted 2026-11-20 (lane/fx-verdicts6, A/B abf2 NVIDIA L40S, wheel e41b5148b24d, job nvc-ab-20261120T224801Z-458a59, one run per arm, arm fast-matrix-units-tf32-all: 12 of 13 cells faster, 0 slower, median 0.93, mlp fit 0.96-0.97, lm-train-step 0.82, quality equal): the matrix-units route is the default of a fast build (-D MOJOLEARN_MODE_FAST_ROWS) on NVIDIA and AMD and has no tiers (value 1, median 0.96, is deleted); drop the define, or build the old exact route with -D MOJOLEARN_GEMM_FAST_MATRIX_UNITS_OFF"
    # TOMBSTONE: MOJOLEARN_GEMM_FAST_BF16_UNITS (DOMINATED, A/B abf2 NVIDIA L40S, wheel e41b5148b24d, job nvc-ab-20261120T224801Z-458a59, one run per arm: median 0.96 but mlp-clf/taxi infer 8.49 -> 15.82 ms, 1.86, gemm max_rel_err 2.2e-3; slower than the TF32 route on every cell) deleted 2026-11-20 by lane/fx-verdicts6; code recoverable at 06d1e8ff34.
    # Restore: git apply experiments/removed/MOJOLEARN_GEMM_FAST_BF16_UNITS.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_GEMM_FAST_BF16_UNITS"](), "removed 2026-11-20 (lane/fx-verdicts6, A/B abf2 NVIDIA L40S, wheel e41b5148b24d, job nvc-ab-20261120T224801Z-458a59, one run per arm; DOMINATED: median 0.96 but mlp-clf/taxi infer 8.49 -> 15.82 ms, 1.86, gemm max_rel_err 2.2e-3; slower than the TF32 route on every cell): the bf16 matrix-unit products of a fast build; the TF32 matrix-units route is the default; see docs/TOMBSTONES.md"
    # ---- end lane fx-verdicts6
    # ---- lane inv-bf16-kernel (2026-11-20): MOJOLEARN_GEMM_FAST_BF16_RESIDENT (gemm/fast_bf16_resident.mojo: bf16 matrix-unit products over resident bf16 operands; default off, A/B owed on the kernel-clock rows) is a relaxation of `fast` only.
    comptime assert (not is_defined["MOJOLEARN_GEMM_FAST_BF16_RESIDENT"]()) or (is_defined["MOJOLEARN_MODE_FAST_ROWS"]() and not is_defined["MOJOLEARN_GEMM_FAST_MATRIX_UNITS_OFF"]()), "MOJOLEARN_GEMM_FAST_BF16_RESIDENT is legal only in a fast build with the matrix-units route (-D MOJOLEARN_MODE_FAST_ROWS, without MOJOLEARN_GEMM_FAST_MATRIX_UNITS_OFF); a build that promises identical bits never takes bf16 operands"
    # ---- end lane inv-bf16-kernel
    # ---- lane fx-verdicts7 (2026-11-21): verdicts of A/B abn9 (NVIDIA L40S host 79f003b8e8ae, wheel 3880db30a6af, job nvc-ab-20261121T003518Z-d38466)
    # TOMBSTONE: MOJOLEARN_PROGRAM_RUN_ENTRY (LOSER, A/B abn9 NVIDIA L40S, wheel 3880db30a6af, job nvc-ab-20261121T003518Z-d38466, one run per arm: neural/program-run 0.551 -> 0.603 ms, 1.10 arm/base) deleted 2026-11-21 by lane/fx-verdicts7; code recoverable at 519388d23a.
    # Restore: git apply experiments/removed/MOJOLEARN_PROGRAM_RUN_ENTRY.fx-verdicts7.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_PROGRAM_RUN_ENTRY"](), "removed 2026-11-21 (lane/fx-verdicts7, A/B abn9 NVIDIA L40S, wheel 3880db30a6af, job nvc-ab-20261121T003518Z-d38466, one run per arm; LOSER: neural/program-run 0.551 -> 0.603 ms, 1.10 arm/base): the switch is gone; a fixed-shape Program.run takes run_program, and dynamic_shapes, optimize and profile keep program_execute_run. Remove -D MOJOLEARN_PROGRAM_RUN_ENTRY; see docs/TOMBSTONES.md."
    # TOMBSTONE: MOJOLEARN_SESSION_CALL_DESCRIPTOR (LOSER, A/B abn9 NVIDIA L40S, wheel 3880db30a6af, job nvc-ab-20261121T003518Z-d38466, one run per arm: neural/ops-session-step 2.63 -> 2.93 ms, 1.11 arm/base; program-run 1.00) deleted 2026-11-21 by lane/fx-verdicts7; code recoverable at 519388d23a.
    # Restore: git apply experiments/removed/MOJOLEARN_SESSION_CALL_DESCRIPTOR.fx-verdicts7.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_SESSION_CALL_DESCRIPTOR"](), "removed 2026-11-21 (lane/fx-verdicts7, A/B abn9 NVIDIA L40S, wheel 3880db30a6af, job nvc-ab-20261121T003518Z-d38466, one run per arm; LOSER: neural/ops-session-step 2.63 -> 2.93 ms, 1.11 arm/base; program-run 1.00): the switch is gone; a same-image Session call is the direct typed call, and a Session of another image keeps the descriptor. Remove -D MOJOLEARN_SESSION_CALL_DESCRIPTOR; see docs/TOMBSTONES.md."
    # TOMBSTONE: MOJOLEARN_SAMBA_PROGRAM_FORWARD (NOISE, A/B abn9 NVIDIA L40S, wheel 3880db30a6af, job nvc-ab-20261121T003518Z-d38466, one run per arm: neural/samba-forward 4.55 -> 4.55 ms, 1.00 arm/base) deleted 2026-11-21 by lane/fx-verdicts7; code recoverable at 519388d23a.
    # Restore: git apply experiments/removed/MOJOLEARN_SAMBA_PROGRAM_FORWARD.fx-verdicts7.patch; record in docs/TOMBSTONES.md.
    comptime assert not is_defined["MOJOLEARN_SAMBA_PROGRAM_FORWARD"](), "removed 2026-11-21 (lane/fx-verdicts7, A/B abn9 NVIDIA L40S, wheel 3880db30a6af, job nvc-ab-20261121T003518Z-d38466, one run per arm; NOISE: neural/samba-forward 4.55 -> 4.55 ms, 1.00 arm/base): the switch is gone; the stateless SambaStack.forward is the resident forward, and the stack program serves the stateful forward, the step and a stack the resident route does not take. Remove -D MOJOLEARN_SAMBA_PROGRAM_FORWARD; see docs/TOMBSTONES.md."
    # ---- end lane fx-verdicts7
    return True


comptime NEURAL_GUARDS_OK = _check_configuration()
