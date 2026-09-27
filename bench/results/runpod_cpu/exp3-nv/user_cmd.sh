set -u
# Cold-build reproducibility of the gfx942 IDENTICAL bindings: each replicate is
# its own source copy and its own empty MODULAR_HOME (no Mojo cache), -j 1.
cd /root/mojolearn
O=$LEG_OUT/exp3; mkdir -p $O
ENVP=$(pixi run printenv CONDA_PREFIX)
REALPIXI=$(command -v pixi)
mkdir -p /root/shim
cat > /root/shim/pixi <<SHIM
#!/bin/bash
if [ "\$1" = run ]; then shift; exec $REALPIXI run env MODULAR_HOME="\$MOJOLEARN_EXP_MH" "\$@"; fi
exec $REALPIXI "\$@"
SHIM
chmod +x /root/shim/pixi
bindings_for() { echo "linalg mixture solver tsa core arima byte_lm embedding estimators gbdt gp hdbscan ivf kernel_methods mamba metrics preprocessing resample rf svm training transformer trees"; }
NREP=3
mkrep() {  # variant rep
  D=/root/rep/$1-$2; rm -rf $D; mkdir -p $D
  tar -C /root/mojolearn --exclude=./.pixi --exclude=./bench/results -cf - . | tar -C $D -xf -
  ln -s /root/mojolearn/.pixi $D/.pixi
}
runrep() {  # variant rep
  v=$1; r=$2; D=/root/rep/$v-$r; H=/root/mh/$v-$r; rm -rf $H; mkdir -p $H
  cp $ENVP/share/max/modular.cfg $H/ 2>/dev/null; cp -r $ENVP/share/max/crashdb $H/ 2>/dev/null
  for b in $(bindings_for $v); do
    sc=bindings/build_$b.sh; so=_mojolearn_$b.so; if [ $b = core ]; then sc=bindings/build.sh; so=_mojolearn.so; fi
    t=$(date +%s)
    ( cd $D && PATH=/root/shim:$PATH MOJOLEARN_EXP_MH=$H MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 \
      MOJOLEARN_COMPILE_JOBS=2 MOJOLEARN_GPU_ARCHS=$v MOJOLEARN_TARGET_COLUMN=nvidia MOJOLEARN_LINUX_CPU=x86-64-v3 \
      bash $sc > /root/rep/log-$v-$r-$b.txt 2>&1 )
    rc=$?
    f=$D/python/mojolearn/identical/$so
    sha=$(sha256sum $f 2>/dev/null | cut -c1-16)
    printf '%s\t%s\t%s\t%s\t%s\t%s\n' $v $r $b $rc $(( $(date +%s)-t )) "$sha" >> $O/builds.tsv
    mkdir -p $O/sets/$v-$r; cp $f $O/sets/$v-$r/ 2>/dev/null
    rm -rf $H/cache $H/.mojo_cache   # each binding cold too
  done
}
for v in sm_90a sm_89; do for r in $(seq 1 $NREP); do mkrep $v $r; done; done
for v in sm_90a sm_89; do for r in $(seq 1 $NREP); do runrep $v $r & done; done; wait
ls $ENVP/share/max/cache > $O/shared_cache_ls.txt 2>&1
du -sh $ENVP/share/max/cache >> $O/shared_cache_ls.txt 2>&1
grep -l -i 'error' /root/rep/log-* > $O/logs_with_error.txt 2>/dev/null
mkdir -p $O/errlogs; for f in $(cat $O/logs_with_error.txt | head -5); do tail -30 $f > $O/errlogs/$(basename $f); done
# summary: distinct sha per (variant, binding)
python3 - $O/builds.tsv > $O/summary.txt <<'PY'
import sys,collections
d=collections.defaultdict(list)
for l in open(sys.argv[1]):
    v,r,b,rc,t,sha=(l.rstrip('\n').split('\t')+[''])[:6]
    d[(v,b)].append(sha if rc=='0' else 'FAIL')
for (v,b),s in sorted(d.items()):
    c=collections.Counter(s); print(f"{v}\t{b}\tbuilds={len(s)}\tdistinct={len(c)}\t{dict(c)}")
PY
