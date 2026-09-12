#!/bin/sh
# AMD CONFIRMATION FOR DEVIATION 2649 (the byte LM step glue flip), MI300X.
#
# 2649 flipped the NVIDIA column's step glue default to
# `optskip_noshadow_rows16` on an H100 leg. The AMD column was NOT touched:
# `step_glue_default_arm_for` returns the winner for COLUMN_NVIDIA and
# `shipped` for every other column, so Apple and AMD still launch the 128
# thread RMSNorm rows and still take the shadow copy and the in-place
# optimizer step (BRIEF_step_glue section 11).
#
# Two questions, kept apart:
#
#   a. DID THE MERGE DISTURB AMD? The shipped build must still produce the
#      same step witnesses it did before, and the glue check must still pass
#      on this board. A BITS question, and the one that matters. Note the
#      check builds WITH the trial define by design, so on AMD it exercises
#      the arms without the column having flipped.
#   b. SHOULD AMD FLIP TOO? The NVIDIA winner is priced against `shipped` on
#      both corpora. The flip rule (ENGINEERING_RULES section 9) wants the
#      geometric mean of the two lean step ratios below 1 with every step
#      witness equal. Nothing flips here; the number decides, and a losing
#      number is a result.
#
# ONE ARM, NOT FIVE. The H100 leg priced the whole family to find a winner.
# This is a confirmation of that one winner on a second vendor inside a
# 60-minute Hot Aisle lease (60 is the maximum and an extension is refused),
# so the arm list is the winner alone and the timers build is skipped. The
# reference `lean-glue-shipped` still runs from the SAME binary as the arm,
# which is what every verdict is measured against.
#
# A MOJOLEARN_GEMM_LEG_EXTRA body for tools/hotaisle_leg.sh.
# POSIX sh only.
set -u
cd /root/mojolearn || exit 9
OUTROOT=/root/gemm_leg_out

MOJOLEARN_STEP_GLUE_LEG_ARMS=optskip_noshadow_rows16 \
MOJOLEARN_STEP_GLUE_LEG_SKIP_TIMERS=1 \
MOJOLEARN_COMPILE_JOBS=8 \
    sh tools/step_glue_leg.sh
g=$?
echo "step_glue_leg_exit=$g" >> $OUTROOT/leg.txt
echo "amd_glue_arm=optskip_noshadow_rows16 amd_column_default=shipped" >> $OUTROOT/leg.txt
exit "$g"
