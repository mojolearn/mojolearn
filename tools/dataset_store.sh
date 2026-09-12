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
#   sh tools/dataset_store.sh stage "<ssh flags+target>" <key>...
#                                                   # THE ONE A LEG CALLS: put
#                                                   # keys on a rented box,
#                                                   # verified, no credential
#                                                   # leaving this machine
#   sh tools/dataset_store.sh manifest              # hash local files, write the pins
#   sh tools/dataset_store.sh push [key...]         # upload (default: all)
#   sh tools/dataset_store.sh list
#   sh tools/dataset_store.sh presign <key> [secs]  # GET URL to hand to a box
#   sh tools/dataset_store.sh presign-put <key> [secs]
#                                                   # WRITE URL, so a box can
#                                                   # upload straight to R2
#   sh tools/dataset_store.sh pull <key> [dest]     # fetch here, then verify
#   sh tools/dataset_store.sh verify <key> [dest]   # size + sha256 against the pins
#   sh tools/dataset_store.sh box-cmd <key>         # print the curl+verify a pod should run
#
# The keys, as pinned in bench/results/dataset_store/manifest.tsv:
#   gbm-bench/taxi/taxi_speed.npz               419,757,252
#   gbm-bench/istella/istella_speed.npz       2,248,281,826   (decoded; skips the 18 min parse)
#   gbm-bench/istella/istella-s-letor.tar.gz    472,129,615   (source, so a decode is reproducible)
#   corpus/enwik8/input.txt                     100,000,000   (neural, English kind)
#   corpus/pile_github/input.txt                 97,124,565   (neural, source-code kind)
#
# And one MULTI-SHARD group, whose members are keys in their own right:
#   corpus/fineweb-edu-10BT/NNN_00000.parquet  28,518,193,415 total, 14 shards
#                                                 (FineWeb-Edu sample-10BT, the
#                                                  corpus for a real LM run)
#
# BYTES NEED NOT PASS THROUGH THIS MACHINE. The FineWeb shards went
# HuggingFace -> rented box -> R2 directly: the box fetched each shard,
# computed its sha256 and PUT it with a presigned write URL minted here, then
# deleted it locally. At this Mac's ~8.9 MB/s upstream, pushing 28.5 GB from
# here would have cost about an hour of saturated uplink for bytes no local
# code reads. That is what `presign-put` exists for.
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

# A MULTI-SHARD GROUP: one logical corpus held as many objects. The catalog
# above maps one key to exactly one file, which a 14-shard parquet corpus
# breaks, so a group names a key prefix, the directory its shards live in
# locally, and the shard file names. `push`, `verify` and `presign-put` take
# either a plain key or a group name; a group expands to one key per shard, so
# every shard still gets its own size+sha256 pin and nothing about the five
# single-file keys changes.
groups() {
    cat <<'EOF'
corpus/fineweb-edu-10BT	ROOT/training/corpus/fineweb-edu-10BT	000_00000.parquet 001_00000.parquet 002_00000.parquet 003_00000.parquet 004_00000.parquet 005_00000.parquet 006_00000.parquet 007_00000.parquet 008_00000.parquet 009_00000.parquet 010_00000.parquet 011_00000.parquet 012_00000.parquet 013_00000.parquet
EOF
}

group_members() { groups | awk -F'\t' -v g="$1" '$1==g {print $3}'; }
group_dir_for() { groups | awk -F'\t' -v g="$1" '$1==g {print $2}'; }

# Expand group names to their member keys; a plain key expands to itself.
expand_keys() {
    for _a in "$@"; do
        _mem=$(group_members "$_a")
        if [ -n "$_mem" ]; then
            for _m in $_mem; do echo "$_a/$_m"; done
        else
            echo "$_a"
        fi
    done
}

local_path_for() {
    _p=$(catalog | awk -F'\t' -v k="$1" '$1==k {print $2}')
    if [ -z "$_p" ]; then
        # maybe a shard of a group: <group key>/<shard file name>
        _gd=$(group_dir_for "$(dirname "$1")")
        [ -n "$_gd" ] || return 1
        _p="$_gd/$(basename "$1")"
    fi
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
    _hashed=0
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
        _hashed=$((_hashed + 1))
    done
    # Rows this machine CANNOT hash are carried forward, not dropped. A
    # multi-shard corpus is pinned by the box that fetched it, and its bytes
    # deliberately never exist here; rebuilding the local pins must not
    # silently un-pin 28.5 GB that is sitting in R2. Only keys in the local
    # catalog are recomputed above.
    _carried=0
    if [ -f "$MANIFEST" ]; then
        _localkeys=$(catalog | cut -f1)
        while IFS= read -r _row; do
            _k=$(printf '%s' "$_row" | cut -f1)
            [ -n "$_k" ] || continue
            printf '%s\n' "$_localkeys" | grep -qx "$_k" && continue
            printf '%s\n' "$_row" >> "$_tmp"
            _carried=$((_carried + 1))
        done < "$MANIFEST"
    fi
    # the refusal below counts only LOCALLY HASHED rows on purpose: carried
    # rows must never be able to disguise a catalog that resolved nothing.
    _rows=$(wc -l < "$_tmp" | tr -d ' ')
    if [ "$_hashed" = 0 ]; then
        rm -f "$_tmp"
        echo "REFUSING to write an empty manifest: nothing in the catalog resolved" >&2
        return 1
    fi
    LC_ALL=C sort -o "$_tmp" "$_tmp"
    mv "$_tmp" "$MANIFEST"
    echo "wrote $MANIFEST ($_rows pinned: $_hashed hashed here, $_carried carried forward)"
    [ "$_missing" = 0 ] || echo "NOTE: some catalog entries were skipped (see above); push only what is pinned" >&2
}

cmd_push() {
    need_aws || return 1
    load_creds || return 1
    [ -f "$MANIFEST" ] || { echo "run 'manifest' first so pushes are pinned" >&2; return 1; }
    if [ "$#" -gt 0 ]; then _keys=$(expand_keys "$@"); else _keys=$(catalog | cut -f1); fi
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

# A presigned PUT, which `aws s3 presign` cannot mint: it signs GET only, so a
# box could read from R2 but never write to it, and every upload had to be run
# from this machine. That is the whole reason a 28.5 GB corpus looked like it
# needed an hour of Mac uplink. SigV4 is signed here in stdlib python3 (no
# boto3, so this works on a bare checkout); the secret is passed through the
# environment, never argv, and only the finished URL is printed.
#
# Hand the URL to a box INSIDE a piped script or a 0600 curl config file, the
# way cmd_stage does, so it never lands in the box's process list.
cmd_presign_put() {
    load_creds || return 1
    key="${1:?usage: presign-put <key|group> [seconds]}"; secs="${2:-7200}"
    for _k in $(expand_keys "$key"); do
        # load_creds only exports the AWS_* aliases, so the R2_* values are
        # handed to the child as an env prefix: visible to python, absent from
        # argv, and never exported into this shell's wider environment.
        R2_ACCOUNT_ID="$R2_ACCOUNT_ID" R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID" \
        R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY" R2_BUCKET="$R2_BUCKET" \
        PRESIGN_KEY="$_k" PRESIGN_SECS="$secs" python3 - <<'PY' || return 1
import datetime, hashlib, hmac, os, sys, urllib.parse

def sign(k, m):
    return hmac.new(k, m.encode(), hashlib.sha256).digest()

acct = os.environ["R2_ACCOUNT_ID"]
akid = os.environ["R2_ACCESS_KEY_ID"]
secret = os.environ["R2_SECRET_ACCESS_KEY"]
bucket = os.environ["R2_BUCKET"]
key = os.environ["PRESIGN_KEY"]
expires = int(os.environ["PRESIGN_SECS"])

host = "%s.r2.cloudflarestorage.com" % acct
now = datetime.datetime.now(datetime.timezone.utc)
amzdate = now.strftime("%Y%m%dT%H%M%SZ")
datestamp = now.strftime("%Y%m%d")
region, service = "auto", "s3"
scope = "%s/%s/%s/aws4_request" % (datestamp, region, service)
uri = "/" + urllib.parse.quote("%s/%s" % (bucket, key), safe="/~")
q = {"X-Amz-Algorithm": "AWS4-HMAC-SHA256",
     "X-Amz-Credential": "%s/%s" % (akid, scope),
     "X-Amz-Date": amzdate,
     "X-Amz-Expires": str(expires),
     "X-Amz-SignedHeaders": "host"}
cqs = "&".join("%s=%s" % (urllib.parse.quote(k2, safe="-_.~"),
                          urllib.parse.quote(v, safe="-_.~"))
               for k2, v in sorted(q.items()))
creq = "\n".join(["PUT", uri, cqs, "host:%s\n" % host, "host", "UNSIGNED-PAYLOAD"])
sts = "\n".join(["AWS4-HMAC-SHA256", amzdate, scope,
                 hashlib.sha256(creq.encode()).hexdigest()])
k = sign(("AWS4" + secret).encode(), datestamp)
for part in (region, service, "aws4_request"):
    k = sign(k, part)
sig = hmac.new(k, sts.encode(), hashlib.sha256).hexdigest()
sys.stdout.write("https://%s%s?%s&X-Amz-Signature=%s\n" % (host, uri, cqs, sig))
PY
    done
}

# Accepts a group name, in which case every shard is checked against its pin.
cmd_verify() {
    key="${1:?usage: verify <key|group> [dest]}"
    if [ -n "$(group_members "$key")" ]; then
        [ "$#" -le 1 ] || { echo "verify <group> takes no dest" >&2; return 1; }
        _rc=0
        for _k in $(expand_keys "$key"); do verify_one "$_k" || _rc=1; done
        return "$_rc"
    fi
    verify_one "$@"
}

verify_one() {
    key="$1"
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

# Put keys onto a rented box, verified, without any credential leaving here.
#
#   sh tools/dataset_store.sh stage "-p 11827 root@1.2.3.4" <key> [key...]
#
# For each key: mint a short-lived presigned URL locally, then pipe a
# self-verifying fetch script to the box over stdin. The URL travels INSIDE the
# piped script, not in argv, so it never appears in the box's process list; the
# R2 secret never leaves this machine at all. The box fetches with curl -C - and
# refuses the file unless its size AND sha256 equal the committed pins, so a
# truncated transfer cannot quietly become a dataset -- which is exactly how a
# pointwise leg once measured synthclf while believing it had Istella-S.
#
# Any leg can call this after renting. It deliberately does NOT edit the
# per-lane body scripts under bench/results/, because those are the record of
# what a given night actually ran.
cmd_stage() {
    target="${1:?usage: stage \"<ssh flags+target>\" <key> [key...]}"; shift
    [ "$#" -gt 0 ] || { echo "no keys given" >&2; return 1; }
    [ -f "$MANIFEST" ] || { echo "no $MANIFEST; run 'manifest' first" >&2; return 1; }
    _staged=0
    for key in $(expand_keys "$@"); do
        pinned "$key" > /dev/null || { echo "no pin for $key" >&2; return 1; }
        echo "staging $key ..."
        _staged=$((_staged + 1))
        _url=$(cmd_presign "$key" 7200) || return 1
        # shellcheck disable=SC2086
        { printf "URL='%s'\n" "$_url"; cmd_box_cmd "$key"; } | ssh -o StrictHostKeyChecking=no \
            -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR $target 'sh -s' \
            || { echo "staging FAILED: $key" >&2; return 1; }
    done
    echo "staged $_staged key(s); the box verified each against the pins"
}

case "${1:-}" in
    stage)    shift; cmd_stage "$@" ;;
    manifest) shift; cmd_manifest "$@" ;;
    push)     shift; cmd_push "$@" ;;
    list)     shift; cmd_list "$@" ;;
    presign)  shift; cmd_presign "$@" ;;
    presign-put) shift; cmd_presign_put "$@" ;;
    pull)     shift; cmd_pull "$@" ;;
    verify)   shift; cmd_verify "$@" ;;
    box-cmd)  shift; cmd_box_cmd "$@" ;;
    *) sed -n '2,40p' "$0"; exit 2 ;;
esac
