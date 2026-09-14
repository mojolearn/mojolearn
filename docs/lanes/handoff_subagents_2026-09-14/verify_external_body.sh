# item 7: validate docs/VERIFY_EXTERNALLY.md + tools/verify_external.sh as an OUTSIDER
# on a box that did not build the wheel: the pod's own python3/pip/git, pixi
# removed from PATH, a fresh clone of the public repository (NOT the shipped
# archive, which carries no bench/results), the PyPI wheel.
set -u
OUT=/root/gemm_leg_out; mkdir -p "$OUT"
PATH=$(printf '%s' "$PATH" | tr ':' '\n' | grep -v -i pixi | paste -sd: -); export PATH
{ echo "python3=$(command -v python3) $(python3 --version 2>&1)"; echo "git=$(command -v git)"; echo "pixi on PATH: $(command -v pixi || echo none)"; } | tee "$OUT/outsider_env.txt"
python3 -m pip install --quiet numpy 2>&1 | tail -2
cd /root
GIT_TERMINAL_PROMPT=0 git clone --quiet https://github.com/mojolearn/mojolearn.git outsider > "$OUT/clone.log" 2>&1 || { echo "clone FAILED"; cat "$OUT/clone.log"; exit 0; }
cd /root/outsider && git log --oneline -1 | tee "$OUT/outsider_head.txt"
RECORD=bench/results/identity_break/2026-09-14_47-lanes
sh tools/verify_external.sh nvidia-h100-sm_90a "$RECORD" > "$OUT/verify_external.log" 2>&1
RC=$?
echo "verify_external exit $RC" | tee -a "$OUT/verify_external.log"
cp -v verify_external_out/*.json verify_external_out/*.txt "$OUT/" 2>&1 | tail -5
tail -40 "$OUT/verify_external.log"
exit 0
