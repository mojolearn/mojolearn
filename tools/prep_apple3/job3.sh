#!/bin/sh
# lane prep-apple3, steward speed job 3 (one job, one Mac). Commits, each with its own build:
#   BASE the merged base; C0 the defaults after job 2; C1 + x_prep/fastexact.mojo; C2 + the host-side
#   download; the tree + the no-copy runner. FAST, the TargetEncoder quality pair, then IDENTICAL.
BASE=6856b5f8f
C0=d6ecbae56
C1=a2dc8d742
C2=00a29f144
OPT=MOJOLEARN_XPREP_R3_ON=all,MOJOLEARN_XPREP_EXACT=1,MOJOLEARN_XPREP_II_SYM=1
BIG=onehot,label-binarizer,multilabel-binarizer,poly-features,spline,robust-scaler,normalizer,power-transformer,target-encoder
sh bench/x_prep_ab.sh $BASE fast 2 all \
    "base||" \
    "at-$C0-def||" \
    "at-$C0-labels||MOJOLEARN_XPREP_R3_ON=label_buffers|--only label-encoder,label-binarizer,multilabel-binarizer" \
    "at-$C0-work2||MOJOLEARN_XPREP_R3_ON=work2|--only power-transformer,target-encoder,onehot,label-binarizer,multilabel-binarizer" \
    "at-$C1-exact||MOJOLEARN_XPREP_EXACT=1" \
    "at-$C1-iisym||MOJOLEARN_XPREP_II_SYM=1|--only iterative-imputer" \
    "at-$C1-tefast||MOJOLEARN_XPREP_TE_FAST=1|--only target-encoder" \
    "at-$C1-all||$OPT" \
    "at-$C1-alltouch||$OPT|--touch --only $BIG" \
    "at-$C2-memcpy||$OPT,MOJOLEARN_XPREP_DOWNLOAD=memcpy" \
    "at-$C2-threads||$OPT,MOJOLEARN_XPREP_DOWNLOAD=threads" \
    "at-$C2-threadstouch||$OPT,MOJOLEARN_XPREP_DOWNLOAD=threads|--touch --only $BIG" \
    "nocopy||$OPT,MOJOLEARN_XPREP_NOCOPY=1|--touch --only $BIG" \
    "nocopynotouch||$OPT,MOJOLEARN_XPREP_NOCOPY=1|--only $BIG" \
    "at-$C2-prof||$OPT,MOJOLEARN_XPREP_DOWNLOAD=threads,MOJOLEARN_XPREP_PROFILE=1|--reps 1 --only onehot,label-binarizer,poly-features,power-transformer,iterative-imputer,target-encoder,categorical-nb,qda,lda,mutual-info-classif,multilabel-binarizer,label-encoder,select-f-classif,select-f-regression,gaussian-nb,bernoulli-nb"
# the paired quality check of te_global's FAST fold (scikit-learn float64 is the reference)
[ -d $HOME/skl313/sklearn ] || { /usr/bin/python3 -m pip install -q --target $HOME/skl313 --python-version 3.13 --only-binary=:all: --ignore-requires-python --platform macosx_14_0_arm64 scikit-learn==1.7.2 scipy joblib threadpoolctl; rm -rf $HOME/skl313/numpy $HOME/skl313/numpy-*; }
XPREP_AB_SCRIPT=bench/x_prep_quality.py sh bench/x_prep_ab.sh $BASE fast 1 all \
    "at-$C1-q0||PYTHONPATH=$HOME/skl313,MOJOLEARN_XPREP_TE_FAST=0|--arm tefast0 --only target-encoder" \
    "at-$C1-q1||PYTHONPATH=$HOME/skl313,MOJOLEARN_XPREP_TE_FAST=1|--arm tefast1 --only target-encoder"
sh bench/x_prep_ab.sh $BASE identical 2 all \
    "base||" \
    "at-$C0-def||" \
    "at-$C0-opt||MOJOLEARN_XPREP_R3_ON=all" \
    "at-$C2-threads||MOJOLEARN_XPREP_R3_ON=all,MOJOLEARN_XPREP_DOWNLOAD=threads" \
    "new||MOJOLEARN_XPREP_R3_ON=all"
