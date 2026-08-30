#!/usr/bin/env bash
# The release workflow's three artifact-handling shell blocks, RUN, on the Mac.
#
#   bash packaging/release_workflow_test.sh
#
# WHY THIS EXISTS. `.github/workflows/release-provenance.yml` decides what
# reaches PyPI, and until 0.3.0 every line of that decision was written unrun:
# the only way to exercise it was to dispatch a release. Two bugs were found
# by running it here, before it ever ran in CI, and one of them would have
# broken EVERY macOS-only release:
#
#   * `${{ env.MOJOLEARN_LINUX_WHEEL_DIR }}` cannot see a variable set in the
#     self-hosted runner's own environment; the override silently did nothing.
#   * under `set -o pipefail`, `ls *.whl | grep -v -- '-macosx_'` exits 1 when
#     there is no Linux wheel, which is the normal case. The Digest step would
#     have died on a macOS-only release.
#
# WHAT IT RUNS. Not a retyped copy. It parses the YAML and extracts the
# `run:` bodies of the `linux` (admit), `digest` and "What is about to be
# published" steps VERBATIM, then drives them against fabricated wheels in a
# temp directory. Nothing is compiled, downloaded, published or rented, and
# the real `python/dist/` is never touched.
#
# THE RULE THIS FILE LEARNED THE HARD WAY, twice. A refusal case that stops
# refusing must FAIL, not disappear: `run_admit "$B" || ok "..."` scores
# nothing at all when the check is removed, and the suite quietly went from
# nine cases to eight while still printing FAIL=0. And a non-zero exit is NOT
# proof the intended check fired: with the one-wheel check deleted the script
# still failed, downstream, on a `basename` of two lines. Every refusal case
# therefore asserts BOTH that the block refused AND that it named its own
# reason. All nine sabotages below are caught; verified 2026-08-30.
#
#   drop the one-wheel check            drop the sidecar-exists check
#   drop the version check              drop the sidecar-digest check
#   drop the manylinux-tag check        restore the pipefail bug
#   drop count-equals-manifest          drop `sha256sum -c`
#   drop the macOS digest check
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WF="$REPO/.github/workflows/release-provenance.yml"
SP="$(mktemp -d)"; trap 'rm -rf "$SP"' EXIT
python3 - "$WF" "$SP" <<'EXTRACT'
import sys, yaml, pathlib
wf, out = sys.argv[1], pathlib.Path(sys.argv[2])
d = yaml.safe_load(open(wf))
b = d["jobs"]["build"]["steps"]
(out / "admit.sh").write_text([s for s in b if s.get("id") == "linux"][0]["run"])
(out / "digest.sh").write_text([s for s in b if s.get("id") == "digest"][0]["run"])
(out / "publish.sh").write_text(
    [s for s in d["jobs"]["publish"]["steps"]
     if s.get("name", "").startswith("What is about")][0]["run"])
EXTRACT
[ -s "$SP/admit.sh" ] && [ -s "$SP/digest.sh" ] && [ -s "$SP/publish.sh" ] || {
  echo "could not extract the three run blocks from $WF"; exit 2; }
PASS=0; FAIL=0
ok(){ echo "  PASS  $1"; PASS=$((PASS+1)); }
no(){ echo "  FAIL  $1  ($2)"; FAIL=$((FAIL+1)); }

mkbox(){ # $1=version ; builds a fake workspace, echoes its path
  local V="$1" B; B="$(mktemp -d)"
  mkdir -p "$B/python/dist" "$B/stage"
  printf 'version = "%s"\n' "$V" > "$B/python/pyproject.toml"
  : > "$B/python/dist/mojolearn-$V-py3-none-macosx_11_0_arm64.whl"
  echo "macos body $V" > "$B/python/dist/mojolearn-$V-py3-none-macosx_11_0_arm64.whl"
  echo "$B"
}
stage(){ # $1=box $2=filename [$3=BAD to corrupt the sidecar]
  local B="$1" N="$2"
  echo "linux body" > "$B/stage/$N"
  if [ "${3:-}" = BAD ]; then echo "0000000000000000000000000000000000000000000000000000000000000000  $N" > "$B/stage/$N.sha256"
  elif [ "${3:-}" = NOSIDE ]; then :
  else shasum -a 256 "$B/stage/$N" | sed "s#$B/stage/##" > "$B/stage/$N.sha256"; fi
}
run_admit(){ ( cd "$1" && MOJOLEARN_LINUX_WHEEL_DIR="$1/stage" GITHUB_OUTPUT="$1/gho.txt" bash "$SP/admit.sh" ) >"$1/admit.out" 2>&1; }
run_digest(){ ( cd "$1" && GITHUB_OUTPUT="$1/gho2.txt" bash "$SP/digest.sh" ) >"$1/digest.out" 2>&1; }

echo "== admit =="
B=$(mkbox 0.3.0); stage "$B" "mojolearn-0.3.0-py3-none-manylinux_2_28_x86_64.whl"
run_admit "$B" && [ -f "$B/python/dist/mojolearn-0.3.0-py3-none-manylinux_2_28_x86_64.whl" ] \
  && ok "a good staged wheel is admitted" || no "a good staged wheel is admitted" "rc/copy"

B=$(mkbox 0.3.0)   # empty stage dir
run_admit "$B" && grep -q 'staged=none' "$B/gho.txt" \
  && ok "no stage dir is a macOS-only release, not a failure" || no "empty stage" "$(tail -1 "$B/admit.out")"

B=$(mkbox 0.3.0); stage "$B" "mojolearn-0.3.0-py3-none-manylinux_2_28_x86_64.whl" BAD
refuses(){ # $1=box $2=name $3=required error substring
  if run_admit "$1"; then no "$2" "IT WAS ADMITTED"
  elif ! grep -qF "$3" "$1/admit.out"; then no "$2" "refused, but not for this reason: $(grep -m1 '::error::' "$1/admit.out")"
  else ok "$2"; fi; }
refuses "$B" "a wheel whose digest does not match its sidecar is REFUSED" "sidecar says"

B=$(mkbox 0.3.0); stage "$B" "mojolearn-0.3.0-py3-none-manylinux_2_28_x86_64.whl" NOSIDE
refuses "$B" "a wheel with no sidecar is REFUSED" "has no .sha256 sidecar"

B=$(mkbox 0.3.0); stage "$B" "mojolearn-0.2.0-py3-none-manylinux_2_28_x86_64.whl"
refuses "$B" "a wheel of the WRONG VERSION is REFUSED" "is not version"

B=$(mkbox 0.3.0); stage "$B" "mojolearn-0.3.0-py3-none-linux_x86_64.whl"
refuses "$B" "an UNAUDITED linux_x86_64 tag is REFUSED" "carries no manylinux tag"

B=$(mkbox 0.3.0); stage "$B" "mojolearn-0.3.0-py3-none-manylinux_2_28_x86_64.whl"; stage "$B" "mojolearn-0.3.0-py3-none-manylinux_2_34_x86_64.whl"
refuses "$B" "TWO staged wheels is REFUSED rather than guessed" "stage exactly one"

echo "== digest =="
B=$(mkbox 0.3.0); run_digest "$B" \
  && [ "$(grep -c 'macosx' "$B/gho2.txt")" -ge 1 ] \
  && ok "macOS-only release still produces a manifest (the pipefail bug)" \
  || no "macOS-only digest" "$(tail -3 "$B/digest.out")"

B=$(mkbox 0.3.0); stage "$B" "mojolearn-0.3.0-py3-none-manylinux_2_28_x86_64.whl"
run_admit "$B" && run_digest "$B" \
  && [ "$(sed -n '/wheel_manifest<</,/MANIFEST_EOF/p' "$B/gho2.txt" | grep -c '\.whl$')" = 2 ] \
  && [ "$(sed -n '/wheel_manifest<</,/MANIFEST_EOF/p' "$B/gho2.txt" | grep -n 'macosx' | cut -d: -f1)" = 2 ] \
  && ok "two wheels, macOS FIRST in the manifest" \
  || no "two-wheel manifest" "$(sed -n '/wheel_manifest/,/MANIFEST_EOF/p' "$B/gho2.txt")"

echo "== publish =="
# The publish job runs on ubuntu and only ever sees `dist/` plus the three
# values the build job handed it. These cases fabricate that pair directly.
pub(){ # $1=dir $2=EXPECTED_WHEEL $3=EXPECTED_SHA256 $4=manifest text
  ( cd "$1" && EXPECTED_WHEEL="$2" EXPECTED_SHA256="$3" EXPECTED_MANIFEST="$4" \
      bash "$SP/publish.sh" ) >"$1/pub.out" 2>&1; }
pubbox(){ local D; D="$(mktemp -d)"; mkdir -p "$D/dist"
  echo "macos"  > "$D/dist/mojolearn-0.3.0-py3-none-macosx_11_0_arm64.whl"
  echo "linux"  > "$D/dist/mojolearn-0.3.0-py3-none-manylinux_2_28_x86_64.whl"
  ( cd "$D/dist" && shasum -a 256 mojolearn-0.3.0-py3-none-macosx_11_0_arm64.whl \
      mojolearn-0.3.0-py3-none-manylinux_2_28_x86_64.whl ) > "$D/man.txt"
  echo "$D"; }
prefuses(){ if pub "$1" "$2" "$3" "$4"; then no "$5" "IT WAS PUBLISHED"; else
  if grep -qF "$6" "$1/pub.out"; then ok "$5"; else no "$5" "refused for another reason: $(grep -m1 '::error::' "$1/pub.out")"; fi; fi; }

D=$(pubbox); M=$(cat "$D/man.txt"); MAC=$(grep macosx "$D/man.txt" | cut -d' ' -f1)
pub "$D" "mojolearn-0.3.0-py3-none-macosx_11_0_arm64.whl" "$MAC" "$M" \
  && ok "two matching wheels are published" || no "two matching wheels" "$(tail -3 "$D/pub.out")"

D=$(pubbox); M=$(cat "$D/man.txt"); MAC=$(grep macosx "$D/man.txt" | cut -d' ' -f1)
echo "stowaway" > "$D/dist/mojolearn-0.3.0-py3-none-manylinux_2_34_x86_64.whl"
prefuses "$D" "mojolearn-0.3.0-py3-none-macosx_11_0_arm64.whl" "$MAC" "$M" \
  "an EXTRA wheel nobody built is REFUSED" "the build job named"

D=$(pubbox); M=$(cat "$D/man.txt"); MAC=$(grep macosx "$D/man.txt" | cut -d' ' -f1)
echo "tampered" > "$D/dist/mojolearn-0.3.0-py3-none-manylinux_2_28_x86_64.whl"
prefuses "$D" "mojolearn-0.3.0-py3-none-macosx_11_0_arm64.whl" "$MAC" "$M" \
  "a TAMPERED Linux wheel is REFUSED" "does not match its recorded digest"

D=$(pubbox); M=$(cat "$D/man.txt"); MAC=$(grep macosx "$D/man.txt" | cut -d' ' -f1)
rm "$D/dist/mojolearn-0.3.0-py3-none-manylinux_2_28_x86_64.whl"
prefuses "$D" "mojolearn-0.3.0-py3-none-macosx_11_0_arm64.whl" "$MAC" "$M" \
  "a MISSING wheel is REFUSED" "the build job named"

D=$(pubbox); M=$(grep macosx "$D/man.txt"); MAC=$(echo "$M" | cut -d' ' -f1)
rm "$D/dist/mojolearn-0.3.0-py3-none-manylinux_2_28_x86_64.whl"
pub "$D" "mojolearn-0.3.0-py3-none-macosx_11_0_arm64.whl" "$MAC" "$M" \
  && ok "a macOS-only release still publishes" || no "macOS-only publish" "$(tail -3 "$D/pub.out")"

D=$(pubbox); M=$(cat "$D/man.txt")
prefuses "$D" "mojolearn-0.3.0-py3-none-macosx_11_0_arm64.whl" "0000000000000000000000000000000000000000000000000000000000000000" "$M" \
  "a macOS digest that disagrees with the build job is REFUSED" "does not match the digest the build job recorded"

echo
echo "PASS=$PASS FAIL=$FAIL"
[ "$FAIL" = 0 ]
