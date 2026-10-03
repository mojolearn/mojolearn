#!/bin/bash
# af_batch_run.sh <tag> <command...>     M3 wrapper (lane apple-fast-batch)
# If ~/mq/prebuilt-ab/<tag>/A.so and B.so exist (built once on the M2 by
# tools/af_batch_build.sh), copy them into the A/B tool's out dir and run the
# command with AFT_SKIP_BUILD=1 AFC_SKIP_BUILD=1, so the M3 only times.
# Otherwise run the command unchanged (the tool builds as before).
# Out dir, as the tools compute it:
#   tools/aft_ab.sh      ${AFT_OUT:-$HOME/aft-ab/<binding>}  (AFT_OUT=... in the command)
#   tools/afc_ab_def.sh  $HOME/afc-def/<its TAG argument>
#   tools/afc_ab.sh      builds nothing: run unchanged
set -u
tag=$1; shift
[ $# -ge 1 ] || { sed -n 2,11p "$0"; exit 2; }
pre=$HOME/mq/prebuilt-ab/$tag
out=""; args=("$@"); n=${#args[@]}
for ((i = 0; i < n; i++)); do
  t=${args[$i]}
  case "$t" in
    AFT_OUT=*) aft_out=${t#AFT_OUT=} ;;
    tools/aft_ab.sh) out=${aft_out:-$HOME/aft-ab/${args[$((i + 1))]}} ;;
    tools/afc_ab_def.sh) out=$HOME/afc-def/${args[$((i + 1))]} ;;
  esac
done
out=${out/#\$HOME/$HOME}; out=${out/#\~/$HOME}
if [ -n "$out" ] && [ -f "$pre/A.so" ] && [ -f "$pre/B.so" ]; then
  mkdir -p "$out" && cp "$pre/A.so" "$out/A.so" && cp "$pre/B.so" "$out/B.so"
  echo "AF-BATCH-RUN $tag prebuilt -> $out"
  export AFT_SKIP_BUILD=1 AFC_SKIP_BUILD=1
else
  echo "AF-BATCH-RUN $tag no prebuilt (out=${out:-none}): tool builds"
fi
exec env "$@"
