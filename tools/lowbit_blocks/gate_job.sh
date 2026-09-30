#!/bin/bash
# lane/lowbit-blocks gate job (NVIDIA pod). Phases:
#  1. the block check under the profile (check-transformer-int15), plain build
#  2. the same program with -D MOJOLEARN_LOWBIT_SABOTAGE=1: MUST report failures
#  3. THE DEFAULT GATE: transformer/checks/transformer_check.mojo built from this
#     tree and from a clean worktree at the merge base (whose block sources are
#     main's byte for byte), run on the same GPU; their
#     outputs and identity cards must be identical.
set -u
cd /root/mojolearn-lowbit-blocks
OUT=/root/lb/gate_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$OUT"
B=/root/lb/base
if [ ! -d "$B" ]; then git worktree add --detach "$B" 45464ced2 >/dev/null 2>&1 || { echo "cannot make the base worktree"; exit 2; }; fi
MJ="pixi run -e default mojo"
echo "== phase 1 build" ; $MJ build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . transformer/checks/transformer_int15_check.mojo -o "$OUT/int15_check" 2>&1 | tail -20
"$OUT/int15_check" > "$OUT/int15_check.log" 2>&1; echo "phase 1 exit $?"; grep -E "failures|MOVED" "$OUT/int15_check.log" | head
echo "== phase 2 build (sabotage)" ; $MJ build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_LOWBIT_SABOTAGE=1 -I . transformer/checks/transformer_int15_check.mojo -o "$OUT/int15_check_sab" 2>&1 | tail -20
"$OUT/int15_check_sab" > "$OUT/int15_check_sab.log" 2>&1; echo "phase 2 exit $? (must be non-zero)"; grep -cE "^  profile .*MOVED" "$OUT/int15_check_sab.log" | sed 's/^/profile cases moved under sabotage: /'; grep -cE "^  default .*MOVED" "$OUT/int15_check_sab.log" | sed 's/^/default cases moved under sabotage (must be 0): /'; grep -E "failures" "$OUT/int15_check_sab.log"
echo "== phase 3 default gate"
$MJ build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . transformer/checks/transformer_check.mojo -o "$OUT/tcheck_branch" 2>&1 | tail -20
pixi run -e default bash -c "cd $B && mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . transformer/checks/transformer_check.mojo -o $OUT/tcheck_base" 2>&1 | tail -20
MOJOLEARN_IDENTITY_TRACE="$OUT/card_branch.card" "$OUT/tcheck_branch" > "$OUT/tcheck_branch.log" 2>&1; echo "branch transformer_check exit $?"
(cd "$B" && MOJOLEARN_IDENTITY_TRACE="$OUT/card_base.card" "$OUT/tcheck_base" > "$OUT/tcheck_base.log" 2>&1; echo "base transformer_check exit $?")
# tcmalloc's mbind warning and the card's own path are the only lines that may differ.
grep -v -e tcmalloc -e '^card: ' "$OUT/tcheck_branch.log" | sed -E 's/[0-9]+\.[0-9]+ ?(ms|s)\b/T/g' > "$OUT/a.txt"; grep -v -e tcmalloc -e '^card: ' "$OUT/tcheck_base.log" | sed -E 's/[0-9]+\.[0-9]+ ?(ms|s)\b/T/g' > "$OUT/b.txt"
if cmp -s "$OUT/a.txt" "$OUT/b.txt"; then echo "DEFAULT GATE: transformer_check output identical ($(wc -l < "$OUT/a.txt") lines)"; else echo "DEFAULT GATE: OUTPUT DIFFERS"; diff "$OUT/b.txt" "$OUT/a.txt" | head -20; fi
if [ -f "$OUT/card_branch.card" ] && [ -f "$OUT/card_base.card" ]; then
  if cmp -s "$OUT/card_branch.card" "$OUT/card_base.card"; then echo "DEFAULT GATE: identity card identical ($(wc -l < "$OUT/card_base.card") lines, sha256 $(sha256sum < "$OUT/card_base.card" | cut -c1-16))"; else echo "DEFAULT GATE: CARD DIFFERS"; fi
else echo "DEFAULT GATE: a card is missing"; ls "$OUT"; fi
tail -3 "$OUT/tcheck_branch.log"
echo "out $OUT"
