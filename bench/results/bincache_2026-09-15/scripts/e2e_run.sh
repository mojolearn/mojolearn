#!/bin/sh
set -u
SP=/private/tmp/claude-501/-Users-andrewhendel-CascadeProjects-mojolearn/4e8829df-5da1-4c85-8680-6b52df16132c/scratchpad
E=$SP/bincache/e2e; W=$SP/wt-bincache; T=$E/mojolearn
echo "== clean start: $(sh $E/clean.sh)"
rm -rf $E/outA $E/outB $E/box; mkdir -p $E/box/root
mk() { rm -rf "$T"; python3 -c "import sys; sys.path.insert(0,'$W/tools'); import test_bincache as t; t.make_repo('$T')"; }
box_build() { env -u MOJOLEARN_BINCACHE MOJOLEARN_BINCACHE_MAP=$E/box/root/.mojolearn_bincache/urls.tsv MOJOLEARN_BINCACHE_OUT=$1 BINCACHE_TEST_MARKER=$T/ran.txt nice -n 19 python3 $W/tools/bincache.py build $T/bindings/build_fake.sh; }
echo "== stage 1"; PATH=$E/shim:$PATH sh $W/tools/bincache_leg.sh stage "fake-target" "local-e2e-selftest" | tail -1
mk; echo "== box A build"; box_build $E/outA; cat $E/outA/provenance.tsv | cut -f3,4 ; ls $T/ran.txt
SO_A=$(shasum -a 256 $T/python/mojolearn/identical/_mojolearn_fake.so | cut -c1-64)
echo "== promote"; sh $W/tools/bincache_leg.sh promote $E/outA | tail -2
echo "== stage 2"; PATH=$E/shim:$PATH sh $W/tools/bincache_leg.sh stage "fake-target" "local-e2e-selftest" | tail -1
mk; echo "== box B build"; box_build $E/outB; cut -f3,4 $E/outB/provenance.tsv; ls $T/ran.txt 2>&1 | tail -1
SO_B=$(shasum -a 256 $T/python/mojolearn/identical/_mojolearn_fake.so | cut -c1-64)
echo "A=$SO_A"; echo "B=$SO_B"; [ "$SO_A" = "$SO_B" ] && echo SAME-DIGEST
echo "== credential leak scan over provenance dirs (count of files naming a signature or the key id)"
( . "$HOME/.mojolearn_r2"; grep -rl -e "X-Amz-Signature" -e "$R2_ACCESS_KEY_ID" -e "$R2_SECRET_ACCESS_KEY" $E/outA $E/outB | wc -l )
echo "== map file mode"; ls -l $E/box/root/.mojolearn_bincache/urls.tsv | cut -c1-10
echo "== clean end: $(sh $E/clean.sh)"
