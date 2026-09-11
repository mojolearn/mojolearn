#!/bin/sh
# Lane forest-finish, batch G: build and re-gate the FLIPPED DEFAULT.
#
# A default that ships is a binary that was built with that default and
# fingerprinted, not a trial binary that happened to carry the same define. So
# when DEVIATION 2663's default moves in extratrees/estimator.mojo, this
# rebuilds `dflt` from the edited source with NO defines at all, fingerprints
# it, diffs it against the lane set, and re-times one cell per dataset to
# confirm it lands where the trial width landed.
#
# INERT until the lane pushes the edited source and creates
# /root/trees_out/GO_BATCH_G.
cd /root/mojolearn || exit 9
AB="sh tools/trees_identical_ab.sh"
OUT=/root/trees_out
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical
LANES=rf-clf,rf-reg,et-clf,et-reg,iforest
while [ ! -f $OUT/GO_BATCH_G ]; do sleep 15; done
echo "batchG start $(date -u +%T)"

rm -rf /root/bins/dflt; mkdir -p /root/bins/dflt; cp /root/bins/rowmajor/*.so /root/bins/dflt/
if $AB build dflt trees; then
  sha256sum /root/bins/dflt/_mojolearn_trees.so | tee -a $OUT/ab.txt
else
  rm -f /root/bins/dflt/_mojolearn_trees.so
  echo "DFLT BUILD FAILED" | tee -a $OUT/ab.txt
  exit 9
fi

# Identity of the shipped default, against the lane set and against the trial
# binary it is supposed to equal in behavior.
$AB ib dflt $LANES
$AB diff rowmajor dflt
echo "G1_IDENTITY_DONE $(date -u +%T)" | tee -a $OUT/ab.txt
: > $OUT/phase.G1_IDENTITY_DONE

# One ours-only cell per dataset: the default must land where its trial did.
for ds in taxi istella; do
  MOJOLEARN_SPEED_TAG=dflt $AB speed dflt et $ds 1000000 3 ours
done
echo "G2_CONFIRM_DONE $(date -u +%T)" | tee -a $OUT/ab.txt
: > $OUT/phase.G2_CONFIRM_DONE
echo "batchG end $(date -u +%T)"
