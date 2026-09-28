#!/bin/sh
# trees-apple3 on m2pro (brief update 21:40Z: only what does not fit the
# laptop's 16 GB): the Istella cells at the full timing floor, 1,000,000
# rows x 220 columns, FAST. Arms as tools/trees_apple3/job_laptop1.sh:
#   gbdt-lossguide <- exact batch (a1), exact batch + inherited partition (all)
#   gbdt-depthwise <- inherited partition (all); gbdt-symmetric is a control
#   rf, dt         <- RF node batch 16K (a1), 16K + node split zero-after-read (all)
set -u
EBF="-D MOJOLEARN_GBDT_LG_EXACT_BATCH"
ALL="$EBF -D MOJOLEARN_GBDT_NS_INHERIT_PARTITION -D MOJOLEARN_RF_FAST_BATCH16K -D MOJOLEARN_RF_NODESPLIT_ZERO_AFTER_READ"
A1="$EBF -D MOJOLEARN_RF_FAST_BATCH16K"
TAB_BUILDS="rf gbdt" TAB_ROUNDS=2 TAP_ROWS=1000000 \
TAB_ARMS="all=$ALL;before=;a1=$A1" \
TAP_CELLS="gbdt:gbdt-lossguide:istella gbdt:gbdt-depthwise:istella gbdt:gbdt-symmetric:istella rf:istellareg xt:dt:istellareg" \
    sh tools/trees_apple_ab.sh
