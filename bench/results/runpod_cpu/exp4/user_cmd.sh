set -u
# Whole-file reproducibility, release shape: the same checkout path every time,
# the Mojo cache wiped before each round, the 23 gfx942 IDENTICAL bindings built
# 8 at a time with one compiler worker each (packaging/linux/build_sets.sh's
# gfx942 setting). Six rounds.
cd /root/mojolearn
O=$LEG_OUT/exp4; mkdir -p $O
MH=$(pixi run printenv MODULAR_HOME); echo "MODULAR_HOME=$MH" > $O/notes.txt
BIND="linalg mixture solver tsa core arima byte_lm embedding estimators gbdt gp hdbscan ivf kernel_methods mamba metrics preprocessing resample rf svm training transformer trees"
one() {
  b=$1; sc=bindings/build_$b.sh; [ $b = core ] && sc=bindings/build.sh
  MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=1 MOJOLEARN_GPU_ARCHS=gfx942 \
    MOJOLEARN_TARGET_COLUMN=amd MOJOLEARN_LINUX_CPU=x86-64-v3 bash $sc > /root/r4log-$b.txt 2>&1
  echo "$b $?" >> /root/r4rc.txt
}
export -f one
for r in 1 2 3 4 5 6; do
  rm -rf "$MH/cache" "$MH/.mojo_cache"; rm -rf python/mojolearn/identical/_mojolearn*.so; : > /root/r4rc.txt
  t=$(date +%s)
  printf '%s\n' $BIND | xargs -P 8 -I{} bash -c 'one {}'
  echo "round $r $(( $(date +%s)-t ))s rc: $(sort /root/r4rc.txt | tr '\n' ' ')" >> $O/notes.txt
  mkdir -p $O/sets/round-$r; cp python/mojolearn/identical/_mojolearn*.so $O/sets/round-$r/
  (cd $O/sets/round-$r && sha256sum *.so) > $O/sha-round-$r.txt
done
python3 - $O > $O/summary.txt <<'PY'
import sys,glob,os,collections
O=sys.argv[1]; d=collections.defaultdict(list)
for f in sorted(glob.glob(O+'/sha-round-*.txt')):
    for l in open(f):
        h,n=l.split(); d[n].append(h[:16])
for n,hs in sorted(d.items()):
    c=collections.Counter(hs); print(f"{n}\trounds={len(hs)}\tdistinct={len(c)}\t{dict(c)}")
PY
