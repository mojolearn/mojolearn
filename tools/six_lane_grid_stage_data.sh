#!/usr/bin/env bash
# six_lane_grid_stage_data.sh: put the grid's full inputs and the kit on a box via R2. Runs on the orchestrator's Mac.
#
#   bash tools/six_lane_grid_stage_data.sh push <kit-dir>                  # once: upload kit + full inputs to R2
#   bash tools/six_lane_grid_stage_data.sh box-script <nvidia|amd> <kit-dir> [secs] | ssh <box> bash -s
#                                                                          # per box: fetch, sha-verify, place
#
# The grid's saved recipes read the Oct 6 six-lane full inputs (big-*, reg-*, cls-*, cat-*, raw-*, tsvd-* npz+json,
# sha256-pinned in each recipe), NOT the board's rows-small / rows-full slices. The box needs them at their original
# paths under /root/six-lane-full-ab-20261006/data/ (tsvd inputs cannot be relocated at all). Same pattern as
# ~/mojolearn-evidence/lq/stage_lq_box.sh: credentials stay on this machine (~/.mojolearn_r2, the file
# tools/dataset_store.sh reads); the box only receives short-lived presigned GET URLs inside the piped script.
#
# R2 keys: grid-inputs/<sha256>/<name> (content-addressed: re-pushing is a no-op) and grid-inputs/kit-<sha256>.tar.gz.
# The box script skips a file already present with the right sha256, writes to <path>.part and renames after the
# sha check, untars the kit to /root/grid-kit and ends with one line: GRID_DATA_READY placed=.. present=.. missing=..
set -euo pipefail
CREDS=${MOJOLEARN_R2_CREDS:-$HOME/.mojolearn_r2}
usage() { sed -n '4,6p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }
[ $# -ge 2 ] || usage
CMD=$1; shift
load_creds() {
  [ -f "$CREDS" ] || { echo "no $CREDS" >&2; exit 1; }
  # shellcheck disable=SC1090
  . "$CREDS"
  ENDPOINT="https://$R2_ACCOUNT_ID.r2.cloudflarestorage.com"
  export AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" AWS_DEFAULT_REGION=auto
  command -v aws > /dev/null || { echo "the aws CLI is not installed" >&2; exit 1; }
}
sha() { shasum -a 256 "$1" | cut -d' ' -f1; }
exists() { aws s3api head-object --bucket "$R2_BUCKET" --key "$1" --endpoint-url "$ENDPOINT" > /dev/null 2>&1; }
kit_tar() {  # deterministic-enough tarball of the kit; prints its path
  local kit=$1 out
  out=$(mktemp -d)/grid-kit.tar.gz
  COPYFILE_DISABLE=1 tar -czf "$out" -C "$kit" .
  echo "$out"
}
# lines: vendor<TAB>sha256<TAB>name<TAB>original_path<TAB>laptop candidates (|-separated)
manifest_rows() {
  python3 -c '
import json, sys
for d in json.load(open(sys.argv[1] + "/data-manifest.json")):
    print("\t".join([d["vendor"], d["sha256"], d["name"], d["original_path"], "|".join(d.get("laptop_candidates", []))]))
' "$1"
}

case $CMD in
push)
  KIT=$1; load_creds
  tarball=$(kit_tar "$KIT"); tsha=$(sha "$tarball")
  key=grid-inputs/kit-$tsha.tar.gz
  exists "$key" || aws s3 cp "$tarball" "s3://$R2_BUCKET/$key" --endpoint-url "$ENDPOINT" --only-show-errors
  echo "kit $key"
  echo "$tsha" > "$KIT/.r2-kit-sha256"
  pushed=0 present=0 missing=0
  while IFS=$'\t' read -r vendor fsha name orig cands; do
    key=grid-inputs/$fsha/$name
    if exists "$key"; then present=$((present + 1)); continue; fi
    src=
    IFS='|' read -r -a list <<< "$cands"
    for c in "${list[@]}"; do [ -f "$c" ] && [ "$(sha "$c")" = "$fsha" ] && { src=$c; break; }; done
    if [ -z "$src" ]; then echo "MISSING $vendor $name ${fsha:0:12} (no laptop copy with this sha; box path $orig)"; missing=$((missing + 1)); continue; fi
    aws s3 cp "$src" "s3://$R2_BUCKET/$key" --endpoint-url "$ENDPOINT" --only-show-errors
    pushed=$((pushed + 1))
  done < <(manifest_rows "$KIT" | sort -u -t$'\t' -k2,2)
  echo "push: pushed=$pushed already_in_r2=$present missing=$missing"
  ;;
box-script)
  VENDOR=$1; KIT=$2; SECS=${3:-21600}; load_creds
  [ -f "$KIT/.r2-kit-sha256" ] || { echo "run push first" >&2; exit 1; }
  tsha=$(cat "$KIT/.r2-kit-sha256")
  presign() { aws s3 presign "s3://$R2_BUCKET/$1" --expires-in "$SECS" --endpoint-url "$ENDPOINT"; }
  echo 'set -uo pipefail; placed=0; present=0; missing=0'
  echo 'fetch() {  # fetch <url> <sha256> <path>'
  echo '  if [ -f "$3" ] && [ "$(sha256sum "$3" | cut -d" " -f1)" = "$2" ]; then present=$((present + 1)); return; fi'
  echo '  mkdir -p "$(dirname "$3")"; cfg=$(mktemp); printf "url = \"%s\"\n" "$1" > "$cfg"'
  echo '  if curl -fsS --retry 3 -C - -K "$cfg" -o "$3.part" && [ "$(sha256sum "$3.part" | cut -d" " -f1)" = "$2" ]; then'
  echo '    mv "$3.part" "$3"; placed=$((placed + 1)); else echo "FAILED $3"; missing=$((missing + 1)); fi; rm -f "$cfg"; }'
  printf 'fetch %q %q %q\n' "$(presign "grid-inputs/kit-$tsha.tar.gz")" "$tsha" /root/grid-kit.tar.gz
  echo 'mkdir -p /root/grid-kit && tar -xzf /root/grid-kit.tar.gz -C /root/grid-kit'
  while IFS=$'\t' read -r vendor fsha name orig _c; do
    [ "$vendor" = "$VENDOR" ] || continue
    if exists "grid-inputs/$fsha/$name"; then
      printf 'fetch %q %q %q\n' "$(presign "grid-inputs/$fsha/$name")" "$fsha" "$orig"
    else
      printf 'echo %q; missing=$((missing + 1))\n' "NOT IN R2: $orig (${fsha:0:12})"
    fi
  done < <(manifest_rows "$KIT")
  echo 'echo "GRID_DATA_READY placed=$placed present=$present missing=$missing"'
  ;;
*) usage;;
esac
