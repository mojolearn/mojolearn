# workstream D on a GPU box: build the six touched bindings, the seven surface test modules, the five checks
set -u
cd /root/mojolearn 2>/dev/null || cd "$(pwd)"
OUT=/root/gemm_leg_out; mkdir -p "$OUT"
[ -n "${MOJOLEARN_COMMIT:-}" ] && echo "$MOJOLEARN_COMMIT" > commit.txt
for b in build_gp build_kernel_methods build_mixture build_hdbscan build_resample build; do
  echo "== $b"; env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 sh bindings/$b.sh > "$OUT/$b.log" 2>&1; echo "exit $?"; grep -v "warning:\|^ *\^\|^    var\|^Imported" "$OUT/$b.log" | grep -E "error|built" | tail -2
done
ls python/mojolearn/identical/
cd python
for t in cholesky kernel_methods mixture hdbscan resample training_primitives kmeans_metric; do
  echo "== test_${t}_surface"; env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python pixi run python -m mojolearn.tests.test_${t}_surface > "$OUT/test_${t}_surface.log" 2>&1; echo "exit $?"; tail -3 "$OUT/test_${t}_surface.log" | cut -c1-200
done
cd /root/mojolearn
for c in check-cholesky check-kernel-methods check-mixture check-hdbscan check-resample; do
  echo "== $c"; pixi run $c > "$OUT/$c.log" 2>&1; echo "exit $?"; grep -v "warning:\|^ *\^\|^    var\|^Imported" "$OUT/$c.log" | tail -3 | cut -c1-200
done
exit 0
