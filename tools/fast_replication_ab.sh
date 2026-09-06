#!/bin/sh
# DEVIATION 2040 A/B: does capping FAST's histogram replication make FAST faster?
#
# TWO FAST BUILDS, not FAST against IDENTICAL. The price harness answers a
# different question (what the numeric contract costs); this one asks whether
# the FAST arm is mis-tuned on a high-core-count device, and both arms here
# are FAST so the contract is held constant.
#
# WHY THIS IS A LEGITIMATE SPEED SWITCH. On the gbdt lane's AMD dispatch the
# histogram is `launch_hist2_8bit`, which quantizes PER ROW and sums in Int32,
# so any partition of rows into blocks gives the same histogram bits. The
# harness prints each arm's output hash and REFUSES to report a ratio if they
# differ -- a replication change that moved bits would be a tradeoff, not a
# win, and must not be read off this script.
#
#   MOJOLEARN_AB_OUT=<dir> sh tools/fast_replication_ab.sh
set -u
OUT="${MOJOLEARN_AB_OUT:-bench/results/fast_replication_ab/$(date +%Y-%m-%d_%H%M%S)}"
ROUNDS="${MOJOLEARN_AB_ROUNDS:-3}"
LANES="${MOJOLEARN_AB_LANES:-gbdt rf et}"
ROWS="${MOJOLEARN_AB_ROWS:-1000000}"
# WHICH CANDIDATE THIS RUN TESTS, and it is assigned HERE, above the env block
# that prints it. It was first assigned down beside the build, while the env
# block near the top already referenced it, so `set -u` killed every arm at
# "AB_DEFINE: parameter not set" before a single build started.
# 2040 (histogram replication) is REFUTED, measured inert on an MI325X.
# 2041 pins the partition-stats chunk count; 2042 takes FAST off the
# decoupled-lookback partition. One per run: combining them makes a null
# result unattributable.
AB_DEFINE="${MOJOLEARN_AB_DEFINE:-MOJOLEARN_2040_FAST_REPLICATION_PIN}"
mkdir -p "$OUT/bin"

echo "== DEVIATION 2040 A/B: rows $ROWS, lanes [$LANES], $ROUNDS rounds -> $OUT =="
{
  echo "date $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "head $(git rev-parse HEAD 2>/dev/null || echo nogit)"
  echo "host $(hostname -s)"
  echo "rows $ROWS"
  echo "ab_define $AB_DEFINE"
  if [ "$(uname -s)" = "Darwin" ]; then
    echo "mem $(sysctl -n vm.swapusage 2>/dev/null)"
  else
    echo "mem $(free -m 2>/dev/null | awk '/^Mem:/{printf "used %sM avail %sM; ",$3,$7} /^Swap:/{printf "swap used %sM",$3}')"
  fi
} > "$OUT/env.txt"

echo "== build BASE (FAST, replication as ported) =="
pixi run mojo build -I . bench/lanes_price_main.mojo -o "$OUT/bin/base" \
  > "$OUT/build.base.log" 2>&1 || { echo "!! BASE build failed"; tail -20 "$OUT/build.base.log"; exit 1; }
echo "== build PIN  (FAST, -D ${AB_DEFINE}=1) =="
pixi run mojo build -I . -D ${AB_DEFINE}=1 \
  bench/lanes_price_main.mojo -o "$OUT/bin/pin" \
  > "$OUT/build.pin.log" 2>&1 || { echo "!! PIN build failed"; tail -20 "$OUT/build.pin.log"; exit 1; }

printf 'lane\tbase_med_s\tpin_med_s\tratio\tbase_hash\tpin_hash\tverdict\n' > "$OUT/ab.tsv"
for lane in $LANES; do
  case "$lane" in
    gbdt) SZ="MOJOLEARN_LANES_PRICE_GBDT_ROWS=$ROWS" ;;
    rf)   SZ="MOJOLEARN_LANES_PRICE_RF_ROWS=$ROWS" ;;
    et)   SZ="MOJOLEARN_LANES_PRICE_ET_ROWS=$ROWS" ;;
    *)    SZ="" ;;
  esac
  # ALTERNATE THE ARMS round by round, so a drift that scales both cancels.
  for r in $(seq 1 "$ROUNDS"); do
    for arm in base pin; do
      env $SZ MOJOLEARN_LANES_PRICE_LANE="$lane" MOJOLEARN_LANES_PRICE_ROUNDS=1 \
        "$OUT/bin/$arm" >> "$OUT/$lane.$arm.log" 2>&1
    done
  done
  # LPRICE <lane> <mode> <round|warmup> <size> <seconds> <hash16>
  #   $1      $2     $3     $4              $5     $6        $7
  # The round label is $4, the seconds are $6 and the hash is $7. The first
  # spelling of these two read $5 as the seconds (it is the SIZE), $6 as the
  # hash, and filtered warmup on $3 (it is the MODE, so every line survived).
  med() { grep -h "^LPRICE $1 " "$OUT/$2" 2>/dev/null | awk '$4!="warmup"{print $6}' | sort -n | awk '{a[NR]=$1} END{if(NR)print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'; }
  hsh() { grep -h "^LPRICE $1 " "$OUT/$2" 2>/dev/null | awk '$4!="warmup"{print $7}' | sort -u | tr '\n' ',' ; }
  b=$(med "$lane" "$lane.base.log"); p=$(med "$lane" "$lane.pin.log")
  bh=$(hsh "$lane" "$lane.base.log"); ph=$(hsh "$lane" "$lane.pin.log")
  if [ -z "$b" ] || [ -z "$p" ]; then
    printf '%s\t-\t-\t-\t%s\t%s\tNO DATA\n' "$lane" "$bh" "$ph" >> "$OUT/ab.tsv"; continue
  fi
  if [ "$bh" != "$ph" ]; then
    v="BITS MOVED -- NOT A SPEED RESULT"
  else
    v="bits equal"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$lane" "$b" "$p" \
    "$(awk -v a="$p" -v c="$b" 'BEGIN{if(c>0)printf "%.3f", a/c; else print "-"}')" \
    "$bh" "$ph" "$v" >> "$OUT/ab.tsv"
done

echo "== A/B table (ratio = PIN / BASE; below 1.0 means the cap is FASTER) =="
cat "$OUT/ab.tsv"
echo
echo "A ratio here is only a speed result on a row whose two hashes are EQUAL."
