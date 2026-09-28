#!/bin/sh
# trees-apple3, batch 1: ONE steward speed job on a Mac with Istella staged
# (m3ultra-b). Submitted with --mode fast; the IDENTICAL part sets its own mode.
#
#   1. FAST timing, arms a1 / before / a2 (a1 first: it compiles everything).
#      Each change reaches its own cells only:
#        gbdt-lossguide cells     <- MOJOLEARN_GBDT_LG_EXACT_BATCH (a1 width 32, a2 width 16)
#        rf / dt / bagging cells  <- MOJOLEARN_RF_FAST_BATCH16K (a1), 32K (a2)
#        dart / adaboost:taxi     <- MOJOLEARN_FOREST_SESSION=1 (a1), share (a2)
#      gbdt-depthwise, gbdt-symmetric and adaboost:taxireg are controls.
#   2. IDENTICAL digests, session off and on (the defines are FAST only).
#   3. FAST quality where the Lossguide leaf budget binds, before and after.
#   4. FAST quality of the boosted members, session off against shared tables.
set -u
EBF="-D MOJOLEARN_GBDT_LG_EXACT_BATCH"
TAB_BUILDS="rf gbdt" TAB_ROUNDS=2 \
TAB_ARMS="a1=$EBF -D MOJOLEARN_RF_FAST_BATCH16K|MOJOLEARN_FOREST_SESSION=1;before=|MOJOLEARN_FOREST_SESSION=0;a2=$EBF -D MOJOLEARN_GBDT_LG_EXACT_BATCH16 -D MOJOLEARN_RF_FAST_BATCH32K|MOJOLEARN_FOREST_SESSION=share" \
TAP_CELLS="gbdt:gbdt-lossguide:taxi gbdt:gbdt-lossguide:istella gbdt:gbdt-depthwise:taxi gbdt:gbdt-depthwise:istella gbdt:gbdt-symmetric:taxi gbdt:gbdt-symmetric:istella rf:taxi rf:taxireg rf:istella rf:istellareg xt:dt:taxi xt:dt:taxireg xt:dt:istellareg xt:bagging:taxi xt:dart:taxi xt:dart:taxireg xt:adaboost:taxi xt:adaboost:taxireg" \
    sh tools/trees_apple_ab.sh

MOJOLEARN_NUMERIC_MODE=identical TAB_BUILDS="rf gbdt" TAB_ROUNDS=1 \
TAB_ARMS="ibefore=|MOJOLEARN_FOREST_SESSION=0;iafter=$EBF -D MOJOLEARN_RF_FAST_BATCH16K|MOJOLEARN_FOREST_SESSION=1" \
TAP_CELLS="gbdt:gbdt-lossguide:taxi gbdt:gbdt-depthwise:taxi gbdt:gbdt-symmetric:taxi rf:taxi rf:taxireg rf:istellareg xt:dt:taxireg xt:dt:istellareg xt:bagging:taxi xt:dart:taxi xt:dart:taxireg xt:adaboost:taxi xt:adaboost:taxireg" \
    sh tools/trees_apple_ab.sh

TAB_BUILDS="gbdt" TAB_ROUNDS=1 \
TAB_ARMS="qbefore=|TAP_LABEL=before;qafter=$EBF|TAP_LABEL=after" \
TAP_CELLS="lgq:taxi,istellareg:5" \
    sh tools/trees_apple_ab.sh

TAB_BUILDS="rf" TAB_ROUNDS=1 \
TAB_ARMS="mbefore=|MOJOLEARN_FOREST_SESSION=0 TAP_LABEL=before;mshare=|MOJOLEARN_FOREST_SESSION=share TAP_LABEL=share" \
TAP_CELLS="mq:dart:taxi,taxireg,istellareg:5 mq:adaboost:taxi,istella:5" \
    sh tools/trees_apple_ab.sh
