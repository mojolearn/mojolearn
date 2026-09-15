# LANE_STATUS lane/r2-binding-cache (bincache), 2026-09-15

**State.** The cache is implemented, OFF by default, and tested locally: unit
tests, sabotage arms, closure compiles, and a live R2 end-to-end run with fake
bindings. See `bench/results/bincache_2026-09-15/README.md`. No GPU box was
rented (Andrew's Sep 15 rule). The box proof below is OWED to the next release
record. Do not flip the default before that proof exists.

## How to use it (while the default is OFF)

1. On the Mac, set `MOJOLEARN_BINCACHE=1` when launching any of the three runners. The runner stages presigned URLs after the R2 dataset step; look for `BINCACHE STAGED partition=<arch>/<image> ... entries=N` in the leg log. Qualification legs of `gemm_remote_leg.sh` never stage.
2. In the body, build each binding through the wrapper instead of `sh "$s"`:
   `python3 tools/bincache.py build "$s"` (same environment, same exit code, stdout and stderr merged).
3. After the fetch the runner prints `BINCACHE PROMOTED n`. The box's record is `remote/bincache/provenance.tsv`, with one row per build. Each row gives the outcome (`hit`, `miss+built-uploaded`, `refused:sabotage:...`, `rejected:<why>+built-uploaded`), the key and the sha256 of every file placed or built.
4. `MOJOLEARN_BINCACHE=0` in a body turns the wrapper into plain `sh <script>`.
5. A hit runs no `pixi run`, so the pixi environment must already be installed. All three runners install it before the extra body; a hand-rolled box must run `pixi install` first.

## OWED: the fresh-then-cached proof on real boxes

The goal is one NVIDIA and one AMD pair, each a fresh leg then a cached leg on
the same image and arch. Evidence per pair:
- the `provenance.tsv` of both legs (miss+built-uploaded, then hit)
- `package.bindings` sha256 in both identity_break JSONs, which must be equal
- the diff of the three lanes, which must read IDENTICAL
- the build seconds saved (leg 1 build rows minus leg 2)

Andrew's rule allows one record per vendor at a release, so the second leg of
each pair needs his explicit OK, or it waits for the following release on the
same image.

Body (bake the commit, since RunPod passes no environment). Save it as
`$SP/bincache_pair_body.sh.in`. The commands below substitute the commit and
check that it landed:

```sh
# bincache pair: 3 bindings through the cache, 3 lanes; the same body for the fresh and the cached leg
MOJOLEARN_COMMIT=@COMMIT@; export MOJOLEARN_COMMIT
set -u
cd /root/mojolearn || exit 9
echo "$MOJOLEARN_COMMIT" > commit.txt
OUT=/root/gemm_leg_out/identity; mkdir -p "$OUT/logs"
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
  _cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
  case "$_cc" in 9.0) A=sm_90a ;; 12.0) A=sm_120a ;; *) A="sm_$(echo "$_cc" | tr -d .)" ;; esac
else A=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-z]+'); fi
export MOJOLEARN_GPU_ARCHS=$A
for s in bindings/build.sh bindings/build_preprocessing.sh bindings/build_tsa.sh bindings/build_arima.sh; do
  t0=$(date +%s)
  env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=8 \
    python3 tools/bincache.py build "$s" > "$OUT/logs/$(basename "$s" .sh).log" 2>&1
  echo "$(basename "$s" .sh)	$?	$(( $(date +%s) - t0 ))" >> "$OUT/status.tsv"
done
sha256sum python/mojolearn/identical/*.so > "$OUT/so_sha256.txt"
pixi run python tools/identity_break.py --lanes standard-scaler,holtwinters,arima \
  --json "$OUT/identity_break.bincache.json" > "$OUT/logs/identity_break.log" 2>&1
echo "identity_break_exit=$?" >> "$OUT/status.tsv"
exit 0
```

`bindings/build.sh` (the base binding) is in the list in case `import mojolearn`
needs it. That has not been checked on a box. Adjust the `pixi run python`
line to however the current record bodies launch identity_break.

Commands (NVIDIA; for AMD use `tools/do_extra_leg.sh amd` with
`MOJOLEARN_GPU_ARCHS=gfx942`, or `tools/gemm_remote_leg.sh amd`), run twice
with the same body:

```sh
cd <your worktree>
sed "s/@COMMIT@/$(git rev-parse HEAD)/" $SP/bincache_pair_body.sh.in > $SP/bincache_pair_body.sh
grep -c "$(git rev-parse HEAD)" $SP/bincache_pair_body.sh     # must print 1
MOJOLEARN_BINCACHE=1 MOJOLEARN_GEMM_LEG_LOCAL_CARD=<existing apple.card> \
MOJOLEARN_GEMM_LEG_EXTRA=$SP/bincache_pair_body.sh \
MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-nvidia-rtx4090-bincache-fresh \
  bash tools/gemm_remote_leg.sh nvidia
# then the same with ...-bincache-cached; then compare:
cut -f3,4,6 bench/results/e1g/*-bincache-fresh/remote/bincache/provenance.tsv
cut -f3,4,6 bench/results/e1g/*-bincache-cached/remote/bincache/provenance.tsv
python3 tools/identity_break.py --diff bench/results/e1g/*-bincache-fresh/remote/identity/identity_break.bincache.json \
  bench/results/e1g/*-bincache-cached/remote/identity/identity_break.bincache.json
```

A stronger variant for the cached leg: after the hits, build the same four
into a scratch copy of the tree without the wrapper, then `cmp` each `.so`.
That proves a cached binding equals what this box would have compiled, not
only what the first box did.

## Not done, by design

- The wheel and release pipeline (`packaging/linux/build_sets.sh`) never calls the cache.
- Hot Aisle `--spec 2gpu` bodies (`/root/leg-a`) are not staged.
- Existing record bodies were not edited; they build from source until one switches to the wrapper.
- The default flip is a later commit, after the box proof.
