#!/bin/sh
# MOJOLEARN_GEMM_LEG_EXTRA wrapper for lane/gpu-confirm-never-launched, leg 2.
#
# THE NON-par HALF OF THE LANES ADDED SINCE v0.8.8, AT ALL NINE PARTS.
# `git diff v0.8.8..HEAD -- tools/identity_break.py` names 26 new lanes. The
# thirteen below are the ones a one-device NVIDIA column can state.
#
# WHY NINE PARTS AND NOT FIVE. Every one of these lanes already shows a
# `nvidia` cell in gpu_coverage, which is what made them look covered. But
# every existing gap column was recorded at identity_break's DEFAULTS --
# train, infer, model, batch, rlpair -- and the other four parts are opt-in
# flags. So `stepfull`, `batchgrad`, `batchscale` and `ragged` have never
# been asked on this silicon for any of them. MOJOLEARN_GAP_PARTS is the
# knob added for exactly that (a959b73e5).
#
# TWO LANES ARE HELD OUT AND NAMED, NOT RUN.
# `language-model-config` and `saved-model-host-infer` are DEGENERATE on
# nvidia-1gpu: tools/lane_applicability.py reports that the lane's
# arithmetic IS the CPU host route (LanguageModelInference; HostForest /
# host_predict / host_predict_proba), so a cell recorded here would measure
# this box's CPU and say nothing whatever about cuda. They read gpu=[] in
# the matrix and will keep reading gpu=[] after this leg, because no run on
# any GPU can close them -- that is a property of the lanes, not a gap. They
# go to MOJOLEARN_GAP_DEGENERATE so the refusal comes home in
# lane_applicability's own words, checked ON THIS BOX.
MOJOLEARN_GAP_LANES=linalg-qr,linalg-eigh,linalg-svdvals,hf-causal-lm,hf-checkpoint,hf-tokenizer,lowbit-conversions,grad-accumulation,gbdt-bfa-quantile,gbdt-border-types,gbdt-catboost-defaults,gbdt-ordered,gbdt-ordered-bayesian-noise
MOJOLEARN_GAP_DEGENERATE=language-model-config,saved-model-host-infer
MOJOLEARN_GAP_SLUG=new-since-088-nine-parts-2026-09-20
MOJOLEARN_GAP_COMMIT_DIR=bench/results/identity_break/2026-09-20_gpu-confirm-never-launched/
MOJOLEARN_GAP_PARTS="--step-full --batch-grad --batch-scale --ragged"
MOJOLEARN_COMPILE_JOBS=16
export MOJOLEARN_GAP_LANES MOJOLEARN_GAP_DEGENERATE MOJOLEARN_GAP_SLUG \
       MOJOLEARN_GAP_COMMIT_DIR MOJOLEARN_GAP_PARTS MOJOLEARN_COMPILE_JOBS
exec sh /root/mojolearn/tools/gap_column_leg.sh
