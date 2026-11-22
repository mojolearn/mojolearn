#!/bin/sh
# lane prep-apple3, steward speed job 2 (one job, one Mac): the merged base, the tree without the
# radix sort (6ff627c6c: the in/out fix, TargetEncoder parallel buckets on, the host-side changes
# behind MOJOLEARN_XPREP_R3_ON) and the tree (the radix sort, opt-in), FAST then IDENTICAL.
BASE=6856b5f8f
SAFE=6ff627c6c
SORTS=robust-scaler,quantile-transformer,kbins,onehot,ordinal,simple-imputer
BIG=onehot,label-binarizer,poly-features,spline,robust-scaler,gaussian-nb,normalizer
ALL=MOJOLEARN_XPREP_R3_ON=all
sh bench/x_prep_ab.sh $BASE fast 2 all \
    "base||" \
    "at-$SAFE-py||" \
    "at-$SAFE-r3||$ALL" \
    "radix||$ALL,MOJOLEARN_XPREP_SORT_RADIX=1" \
    "radix512||$ALL,MOJOLEARN_XPREP_SORT_RADIX=1,MOJOLEARN_XPREP_SORT_CHUNK=512|--only $SORTS" \
    "radix1k||$ALL,MOJOLEARN_XPREP_SORT_RADIX=1,MOJOLEARN_XPREP_SORT_CHUNK=1024|--only $SORTS" \
    "radix4k||$ALL,MOJOLEARN_XPREP_SORT_RADIX=1,MOJOLEARN_XPREP_SORT_CHUNK=4096|--only $SORTS" \
    "radix16k||$ALL,MOJOLEARN_XPREP_SORT_RADIX=1,MOJOLEARN_XPREP_SORT_CHUNK=16384|--only $SORTS" \
    "at-$SAFE-mapped||MOJOLEARN_XPREP_R3_ON=mapped|--only $BIG" \
    "at-$SAFE-mapview||MOJOLEARN_XPREP_R3_ON=mapped+view|--only $BIG" \
    "at-$SAFE-work||MOJOLEARN_XPREP_R3_ON=work|--only robust-scaler,kbins,lda,quantile-transformer,ordinal" \
    "at-$SAFE-nosort||MOJOLEARN_XPREP_R3_ON=imputer_nosort|--only simple-imputer,simple-imputer-mean,iterative-imputer" \
    "at-$SAFE-tearr||MOJOLEARN_XPREP_R3_ON=te_arrays|--only target-encoder" \
    "at-$SAFE-pbucket0||MOJOLEARN_XPREP_TE_PBUCKET=0|--only target-encoder" \
    "prof||$ALL,MOJOLEARN_XPREP_SORT_RADIX=1,MOJOLEARN_XPREP_PROFILE=1|--reps 1 --only robust-scaler,onehot,label-binarizer,poly-features,spline,power-transformer,iterative-imputer,target-encoder,categorical-nb,qda,lda,mutual-info-classif,multilabel-binarizer,label-encoder" \
    "profbase||MOJOLEARN_XPREP_PROFILE=1|--reps 1 --only robust-scaler,onehot,label-binarizer,spline"
sh bench/x_prep_ab.sh $BASE identical 2 all \
    "base||" \
    "at-$SAFE-py||" \
    "at-$SAFE-r3||$ALL" \
    "new||$ALL,MOJOLEARN_XPREP_SORT_RADIX=1"
