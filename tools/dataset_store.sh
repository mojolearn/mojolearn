#!/bin/sh
# tools/dataset_store.sh -- the pinned datasets and corpora, held once in
# Cloudflare R2 instead of refetched per box.
#
# WHY THIS EXISTS. Every rented box used to rebuild Istella-S from
# library.istella.it: about 6 minutes for the 472 MB tarball on a good day
# (40+ on a bad one, and one run blew a 2400 s timeout at 464 of 472 MB) and
# then about 18 minutes to decode it into the 2.25 GB npz. On 2026-09-12 a
# GBDT leg spent 21 minutes on setup to produce 4.5 minutes of measurement.
# R2 stores about 2.9 GB for pennies a month and charges NO egress, so a pod
# pulls a pre-decoded npz at datacenter speed and the decode never happens
# again.
#
# WHAT IT DOES NOT DO. It edits no pinned artifact. `speed_gbdt_arm.py`
# already returns early when its npz exists, and tools/fetch_corpus_*.sh
# already honour MOJOLEARN_CORPUS_SOURCE_DIR, so this script only has to put
# bytes where that code already looks. The corpus fetchers keep verifying
# their own sha256/md5/length pins; nothing here replaces a check.
#
# CREDENTIALS NEVER REACH A RENTED BOX. `presign` mints a short-lived URL on
# this machine; the box receives only that URL and fetches it with plain
# resumable curl. The secret stays in ~/.mojolearn_r2, is never passed in
# argv and is never printed.
#
#   sh tools/dataset_store.sh manifest              # hash local files, write the pins
#   sh tools/dataset_store.sh push [key...]         # upload (default: all)
#   sh tools/dataset_store.sh list
#   sh tools/dataset_store.sh presign <key> [secs]  # URL to hand to a box
#   sh tools/dataset_store.sh pull <key> [dest]     # fetch here, then verify
#   sh tools/dataset_store.sh verify <key> [dest]   # size + sha256 against the pins
#   sh tools/dataset_store.sh box-cmd <key>         # print the curl+verify a pod should run
#
# Needs the aws CLI (S3-compatible mode) for push/list/presign; pull/verify
# need only curl and sha256sum/shasum. POSIX sh.
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
MANIFEST="$ROOT/bench/results/dataset_store/manifest.tsv"
CREDS="${MOJOLEARN_R2_CREDS:-$HOME/.mojolearn_r2}"

# key <TAB> local path, relative to $HOME for data, $ROOT for corpora.
# Istella-S ships BOTH the decoded npz (what the arms read) and the source
# tarball (so a decode can be reproduced without the origin server).
catalog() {
    cat <<'EOF'
gbm-bench/taxi/taxi_speed.npz	HOME/datasets/gbm-bench/taxi/taxi_speed.npz
gbm-bench/istella/istella_speed.npz	HOME/datasets/gbm-bench/istella/istella_speed.npz
gbm-bench/istella/istella-s-letor.tar.gz	HOME/datasets/gbm-bench/istella/istella-s-letor.tar.gz
corpus/enwik8/input.txt	ROOT/training/corpus/enwik8/input.txt
corpus/pile_github/input.txt	ROOT/training/corpus/pile_github/input.txt
EOF
}

local_path_for() {
    _p=$(catalog | awk -F'\t' -v k="$1" '$1==k {print $2}')
    [ -n "$_p" ] || return 1
    case "$_p" in
        HOME/*) echo "$HOME/${_p#HOME/}" ;;
        ROOT/*) echo "$ROOT/${_p#ROOT/}" ;;
        *) echo "$_p" ;;
    esac
}

sha256_of() {
    if command -v sha256sum > /dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1
    else shasum -a 256 "$1" | cut -d' ' -f1; fi
}
size_of() { wc -c < "$1" | tr -d ' '; }

load_creds() {
    [ -f "$CREDS" ] || { echo "no $CREDS (see the header of this script)" >&2; return 1; }
    # shellcheck disable=SC1090
    . "$CREDS"
    for v in R2_ACCOUNT_ID R2_ACCESS_KEY_ID R2_SECRET_ACCESS_KEY R2_BUCKET; do
        eval "_val=\${$v:-}"
        [ -n "$_val" ] || { echo "$CREDS is missing $v" >&2; return 1; }
    done
    ENDPOINT="https://$R2_ACCOUNT_ID.r2.cloudflarestorage.com"
    # the aws CLI reads these from the environment, never from argv
    AWS_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
    AWS_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
    AWS_DEFAULT_REGION=auto
    export AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_DEFAULT_REGION
}

need_aws() { command -v aws > /dev/null 2>&1 || { echo "the aws CLI is not installed" >&2; return 1; }; }

pinned() { awk -F'\t' -v k="$1" '$1==k {print $2"\t"$3}' "$MANIFEST" 2>/dev/null; }

cmd_manifest() {
    mkdir -p "$(dirname "$MANIFEST")"
    _tmp="$MANIFEST.new"
    : > "$_tmp"
    _missing=0
    # keys come through `cut -f1`, NOT through IFS: a literal tab in an
    # `IFS=<tab> read` did not survive editing here and silently swallowed
    # every line into $key, so local_path_for matched nothing, `|| continue`
    # skipped all five files and this wrote an EMPTY manifest and exited 0.
    # An empty pin file is worse than no pin file, because `verify` then
    # reports "no pin" instead of a mismatch.
    for key in $(catalog | cut -f1); do
        _lp=$(local_path_for "$key") || { echo "unknown key in catalog: $key" >&2; _missing=1; continue; }
        if [ ! -f "$_lp" ]; then echo "MISSING locally, skipped: $key ($_lp)" >&2; _missing=1; continue; fi
        _sz=$(size_of "$_lp"); _sh=$(sha256_of "$_lp")
        printf '%s\t%s\t%s\n' "$key" "$_sz" "$_sh" >> "$_tmp"
        printf '%-44s %14s  %s\n' "$key" "$_sz" "$_sh"
    done
    _rows=$(wc -l < "$_tmp" | tr -d ' ')
    if [ "$_rows" = 0 ]; then
        rm -f "$_tmp"
        echo "REFUSING to write an empty manifest: nothing in the catalog resolved" >&2
        return 1
    fi
    mv "$_tmp" "$MANIFEST"
    echo "wrote $MANIFEST ($_rows pinned)"
    [ "$_missing" = 0 ] || echo "NOTE: some catalog entries were skipped (see above); push only what is pinned" >&2
}

cmd_push() {
    need_aws || return 1
    load_creds || return 1
    [ -f "$MANIFEST" ] || { echo "run 'manifest' first so pushes are pinned" >&2; return 1; }
    if [ "$#" -gt 0 ]; then _keys="$*"; else _keys=$(catalog | cut -f1); fi
    for key in $_keys; do
        _lp=$(local_path_for "$key") || { echo "unknown key: $key" >&2; return 1; }
        [ -f "$_lp" ] || { echo "missing locally: $_lp" >&2; return 1; }
        echo "pushing $key ($(size_of "$_lp") bytes) ..."
        aws s3 cp "$_lp" "s3://$R2_BUCKET/$key" --endpoint-url "$ENDPOINT" --only-show-errors \
            || { echo "push FAILED: $key" >&2; return 1; }
        echo "  ok"
    done
}

cmd_list() {
    need_aws || return 1; load_creds || return 1
    aws s3 ls "s3://$R2_BUCKET/" --recursive --endpoint-url "$ENDPOINT"
}

cmd_presign() {
    need_aws || return 1; load_creds || return 1
    key="${1:?usage: presign <key> [seconds]}"; secs="${2:-7200}"
    aws s3 presign "s3://$R2_BUCKET/$key" --expires-in "$secs" --endpoint-url "$ENDPOINT"
}

cmd_verify() {
    key="${1:?usage: verify <key> [dest]}"
    dest="${2:-$(local_path_for "$key")}"
    _pin=$(pinned "$key")
    [ -n "$_pin" ] || { echo "no pin for $key in $MANIFEST" >&2; return 1; }
    _wsz=$(printf '%s' "$_pin" | cut -f1); _wsh=$(printf '%s' "$_pin" | cut -f2)
    [ -f "$dest" ] || { echo "missing: $dest" >&2; return 1; }
    _sz=$(size_of "$dest")
    [ "$_sz" = "$_wsz" ] || { echo "$key size $_sz, pinned $_wsz" >&2; return 1; }
    _sh=$(sha256_of "$dest")
    [ "$_sh" = "$_wsh" ] || { echo "$key sha256 $_sh, pinned $_wsh" >&2; return 1; }
    echo "ok: $key $_sz bytes sha256 $_sh"
}

cmd_pull() {
    key="${1:?usage: pull <key> [dest]}"
    dest="${2:-$(local_path_for "$key")}"
    mkdir -p "$(dirname "$dest")"
    url=$(cmd_presign "$key" 7200) || return 1
    # -C - so a partial file resumes instead of restarting, the failure mode
    # that cost this project two legs on 2026-09-11 and 2026-09-12
    curl -sS -C - --retry 8 --retry-delay 5 --retry-all-errors -o "$dest" "$url" || return 1
    cmd_verify "$key" "$dest"
}

# What a rented box should run. Takes the presigned URL as $2 so no secret
# ever reaches the box; prints a self-verifying fetch.
cmd_box_cmd() {
    key="${1:?usage: box-cmd <key>}"
    _pin=$(pinned "$key")
    [ -n "$_pin" ] || { echo "no pin for $key" >&2; return 1; }
    _wsz=$(printf '%s' "$_pin" | cut -f1); _wsh=$(printf '%s' "$_pin" | cut -f2)
    _lp=$(local_path_for "$key")
    case "$_lp" in "$HOME"/*) _rp="/root/${_lp#"$HOME"/}" ;; *) _rp="/root/$(basename "$key")" ;; esac
    cat <<EOF
# run ON THE BOX; \$URL is a presigned URL minted on the Mac (no credentials here)
mkdir -p "\$(dirname $_rp)"
curl -sS -C - --retry 8 --retry-delay 5 --retry-all-errors -o "$_rp" "\$URL"
sz=\$(wc -c < "$_rp" | tr -d ' ')
[ "\$sz" = "$_wsz" ] || { echo "size \$sz, pinned $_wsz" >&2; exit 1; }
sh=\$(sha256sum "$_rp" | cut -d' ' -f1)
[ "\$sh" = "$_wsh" ] || { echo "sha256 \$sh, pinned $_wsh" >&2; exit 1; }
echo "ok $_rp \$sz \$sh"
EOF
}

case "${1:-}" in
    manifest) shift; cmd_manifest "$@" ;;
    push)     shift; cmd_push "$@" ;;
    list)     shift; cmd_list "$@" ;;
    presign)  shift; cmd_presign "$@" ;;
    pull)     shift; cmd_pull "$@" ;;
    verify)   shift; cmd_verify "$@" ;;
    box-cmd)  shift; cmd_box_cmd "$@" ;;
    *) sed -n '2,40p' "$0"; exit 2 ;;
esac
