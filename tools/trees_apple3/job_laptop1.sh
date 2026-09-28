#!/bin/sh
# trees-apple3 on the laptop M4, run 1 (tools/trees_apple3/laptop_ab.sh):
# every arm this lane lost with m3ultra-b, plus the three later changes.
#
# FAST arms (each change reaches its own cells only):
#   all     EBF width 32 + inherited partition | RF batch 16K + node split zero-after-read | session share
#   before  nothing                                                                       | session off
#   a1      EBF width 32                        | RF batch 16K                             | session exact
#   a2      EBF width 16                        | RF batch 32K                             | session share
#   gbdt-lossguide  <- EBF (a1, a2), EBF + inherited partition (all)
#   gbdt-depthwise  <- inherited partition (all); a1 and a2 are controls
#   gbdt-symmetric  <- nothing (control)
#   rf, dt, bagging <- RF batch (a1 16K, a2 32K), 16K + zero-after-read (all)
#   dart, adaboost  <- data session: exact (a1), share (a2, all); adaboost:taxireg
#                      is the device row gather (exact in a1, a2, all)
# Parts: A taxi-shaped cells at 1,000,000 rows; B Istella cells at 500,000
# rows (16 GB laptop); C IDENTICAL digests, session off and on; D quality.
set -u
LAB="sh tools/trees_apple3/laptop_ab.sh"
EBF="-D MOJOLEARN_GBDT_LG_EXACT_BATCH"
ALL="$EBF -D MOJOLEARN_GBDT_NS_INHERIT_PARTITION -D MOJOLEARN_RF_FAST_BATCH16K -D MOJOLEARN_RF_NODESPLIT_ZERO_AFTER_READ"
A1="$EBF -D MOJOLEARN_RF_FAST_BATCH16K"
A2="$EBF -D MOJOLEARN_GBDT_LG_EXACT_BATCH16 -D MOJOLEARN_RF_FAST_BATCH32K"
ARMS="all=$ALL|MOJOLEARN_FOREST_SESSION=share;before=|MOJOLEARN_FOREST_SESSION=0;a1=$A1|MOJOLEARN_FOREST_SESSION=1;a2=$A2|MOJOLEARN_FOREST_SESSION=share"
part() { echo "######## PART $1 $(date -u +%H:%M:%SZ)"; }

case " ${LAB_PARTS:-A B C D} " in *" A "*)
part A
MOJOLEARN_NUMERIC_MODE=fast TAB_BUILDS="rf gbdt" TAB_ROUNDS=2 TAB_ARMS="$ARMS" TAP_ROWS=1000000 \
TAP_CELLS="gbdt:gbdt-lossguide:taxi gbdt:gbdt-depthwise:taxi gbdt:gbdt-symmetric:taxi rf:taxi rf:taxireg xt:dt:taxi xt:dt:taxireg xt:bagging:taxi xt:dart:taxi xt:dart:taxireg xt:adaboost:taxi xt:adaboost:taxireg" \
    $LAB || exit 1 ;;
esac

case " ${LAB_PARTS:-A B C D} " in *" B "*)
part B
MOJOLEARN_NUMERIC_MODE=fast TAB_BUILDS="rf gbdt" TAB_ROUNDS=2 TAB_ARMS="$ARMS" TAP_ROWS=500000 \
TAP_CELLS="gbdt:gbdt-lossguide:istella gbdt:gbdt-depthwise:istella rf:istellareg xt:dt:istellareg" \
    $LAB ;;
esac

case " ${LAB_PARTS:-A B C D} " in *" C "*)
part C
MOJOLEARN_NUMERIC_MODE=identical TAB_BUILDS="x_trees rf gbdt" TAB_ROUNDS=1 TAP_ROWS=1000000 \
TAB_ARMS="ibefore=|MOJOLEARN_FOREST_SESSION=0;iafter=|MOJOLEARN_FOREST_SESSION=1" \
TAP_CELLS="gbdt:gbdt-lossguide:taxi gbdt:gbdt-depthwise:taxi gbdt:gbdt-symmetric:taxi rf:taxi rf:taxireg xt:dt:taxi xt:dt:taxireg xt:bagging:taxi xt:dart:taxi xt:dart:taxireg xt:adaboost:taxi xt:adaboost:taxireg" \
    $LAB ;;
esac

case " ${LAB_PARTS:-A B C D} " in *" D "*)
part D
MOJOLEARN_NUMERIC_MODE=fast TAB_BUILDS="rf gbdt" TAB_ROUNDS=1 TQ_LOAD_ROWS=1000000 \
TAB_ARMS="qbefore=|MOJOLEARN_FOREST_SESSION=0 TAP_LABEL=before;qall=$ALL|MOJOLEARN_FOREST_SESSION=share TAP_LABEL=all" \
TAP_CELLS="lgq:taxi:5 lgq:taxi:5:Depthwise lgq:istellareg:3 mq:dart,adaboost:taxi,taxireg:5" \
    $LAB ;;
esac
echo "######## DONE $(date -u +%H:%M:%SZ)"
