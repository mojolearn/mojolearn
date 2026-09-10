#!/bin/sh
# H100 leg 2026-09-10 (lane/nvidia-identical-trees-0910): boundary-tax NVIDIA
# IDENTICAL qualification (fingerprints vs Sep 9, native gates), ours-only
# timing at 1M/2M, and the Sep 9 A/B that never got its H100 timing.
# Ours IDENTICAL only; no opponent is re-measured. Phases in priority order.
cd /root/mojolearn || exit 9
AB="sh tools/trees_identical_ab.sh"
OUT=/root/trees_out
GATES=$OUT/gates
mkdir -p $GATES /root/gates
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical
while [ ! -f $OUT/setup.done ]; do sleep 20; done
echo "batchI start $(date -u +%T)"; cat $OUT/setup.txt
mark() { echo "$1 $(date -u +%T)" | tee -a $OUT/ab.txt; : > "$OUT/phase.$1"; }

# ---------- PHASE_A: baseline snapshot, fingerprints, diff vs the Sep 9 H100 set
mkdir -p /root/bins/baseline && cp python/mojolearn/identical/*.so /root/bins/baseline/
sha256sum /root/bins/baseline/*.so | tee $OUT/baseline_so_sha256.txt
$AB ib baseline
$AB diff sep9_h100_baseline baseline
mark PHASE_A_DONE

# ---------- PHASE_B: ours-only timing, baseline, 1M and 2M (never below 1M)
$AB speed baseline rf higgs 1000000 5 ours
$AB speed baseline et higgs 1000000 5 ours
$AB speed baseline rf higgs 2000000 5 ours
$AB speed baseline et higgs 2000000 5 ours
$AB speed baseline rf higgs 1000000 1 stage
mark PHASE_B_DONE

# ---------- PHASE_C: RF candidates (the open loss), build, fingerprints, timing
$AB build rf2010 rf "-D MOJOLEARN_2010_ROWS_SORTED=1"
$AB build rf2011 rf "-D MOJOLEARN_2011_HIST_ITEMS4=1"
$AB build rf2012 rf "-D MOJOLEARN_2012_SMEM_COPIES4=1"
for s in rf2010 rf2011 rf2012; do $AB ib $s rf-clf,rf-reg; $AB diff baseline $s; done
for s in rf2010 rf2011 rf2012; do $AB speed $s rf higgs 1000000 5 ours; done
for s in rf2010 rf2011 rf2012; do $AB speed $s rf higgs 2000000 5 ours; done
mark PHASE_C_DONE

# ---------- PHASE_E: boundary-tax native gates, IDENTICAL, NVIDIA
gate() {  # <name> <source> [extra mojo args]
    _n="$1"; _src="$2"; shift 2
    echo "gate $_n build $(date -u +%T)"
    # shellcheck disable=SC2086
    timeout -k 30 1500 pixi run mojo build -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 $* "$_src" -o /root/gates/$_n > $GATES/$_n.build.log 2>&1
    _b=$?
    if [ "$_b" = 0 ]; then
        timeout -k 30 1200 /root/gates/$_n > $GATES/$_n.run.log 2>&1; _r=$?
    else _r=NOBUILD; fi
    echo "gate $_n build_exit=$_b run_exit=$_r $(date -u +%T)" | tee -a $OUT/ab.txt $GATES/summary.txt
    tail -2 $GATES/$_n.run.log 2>/dev/null
}
$AB use baseline
gate wp4_oob_check ensemble/checks/oob_check.mojo
gate wp1_borrowed_upload_check extratrees/checks/borrowed_upload_check.mojo
gate wp8_stage_upload_bytes_check extratrees/checks/stage_upload_bytes_check.mojo
gate wp2_forest_export_protocol checks/forest_export_protocol.mojo -I bindings
echo "gate wp2_forest_export_public $(date -u +%T)"
MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=/root/mojolearn/python timeout -k 30 1200 python3 checks/forest_export_public.py --mode identical --vendor cuda > $GATES/wp2_forest_export_public.run.log 2>&1
echo "gate wp2_forest_export_public run_exit=$? $(date -u +%T)" | tee -a $OUT/ab.txt $GATES/summary.txt
gate wp3_forest_inference_separate checks/forest_inference_model.mojo
gate wp3_forest_inference_packed checks/forest_inference_model.mojo -D MOJOLEARN_FOREST_PACKED_NODES=1
gate wp5_gbdt_cindex_staging checks/gbdt_cindex_staging_check.mojo
gate wp5_nan_mode_check checks/nan_mode_check.mojo
$AB rfgate src
mark PHASE_E_DONE

# ---------- PHASE_D: symmetric A/B (fold is already in baseline: 4ccccde6 is an ancestor)
$AB build fused gbdt "-D MOJOLEARN_2030_FUSED_EST_MOVE=1"
$AB build sp gbdt "-D MOJOLEARN_2030_FUSED_EST_MOVE=1 -D MOJOLEARN_IDENTICAL_SINGLE_PASS_PARTITION=1"
$AB build spnf gbdt "-D MOJOLEARN_IDENTICAL_SINGLE_PASS_PARTITION=1"
for s in fused sp spnf; do $AB ib $s gbdt-symmetric,gbdt-depthwise,gbdt-lossguide,gbdt-rmse; $AB diff baseline $s; done
for s in baseline fused sp spnf; do $AB speed $s gbdt-symmetric higgs 1000000 7 ours; done
for s in baseline fused sp spnf; do $AB speed $s gbdt-symmetric higgs 2000000 5 ours; done
for s in baseline sp; do $AB speed $s gbdt-symmetric higgsreg 1000000 7 ours; done
mark PHASE_D_DONE
echo "BATCHI_DONE $(date -u +%T)" | tee -a $OUT/ab.txt
: > $OUT/batchI.done
