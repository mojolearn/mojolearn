#!/usr/bin/env bash
# Publish a packed, audited Linux wheel through the Trusted Publisher route:
# manifest, GitHub release, workflow dispatch, watch. One command.
#
#   bash tools/release_linux_publish.sh <final manylinux .whl> <tag> <none|testpypi|pypi> <workdir>
#
# Optional installed qualification (run it when numerics changed or the
# release is paper evidence; docs/RELEASE_CHECKLIST.md step 4):
#
#   MOJOLEARN_QUAL_SM89=<sm_89 release-build dir> MOJOLEARN_QUAL_SM90A=<sm_90a release-build dir> \
#   MOJOLEARN_QUAL_GFX942=<gfx942 release-build dir> MOJOLEARN_QUAL_PROOFS=<dir with cuda-sm_89.json cuda-sm_90a.json hip-gfx942.json> \
#   bash tools/release_linux_publish.sh ...
#
# With those set, the three columns are staged, the admission checker runs,
# and the archive is attached to the release and bound in the manifest.
# Without them the manifest carries the wheel alone and the workflow's
# verifier checks payload, RECORD, metadata and hashes only.
#
# <tag> is created at HEAD if it does not exist. The workflow checks out that
# tag, so HEAD must be the commit the wheel was built from or one whose
# native inventory is unchanged. The inventory (native_inventory in
# tools/check_linux_release_qualification.py) is every .mojo file, the .py
# and .sh files under bindings/, packaging/linux/ and python/mojolearn/,
# pixi.toml, pixi.lock and tools/linux_surface_qualification.sh. Docs,
# other tools/*.sh and tools/*.py do not count; tools/*.py join the
# qualification-sources set only when installed qualification is attached.
# Create the tag explicitly at the build commit when main has moved on.
set -euo pipefail
WHL=$(cd "$(dirname "${1:?final wheel}")" && pwd)/$(basename "$1")
TAG="${2:?tag, e.g. alpha-api-0.7.0-20260909}"
PUBLISH="${3:?none|testpypi|pypi}"
WORK="${4:?workdir}"
case "$PUBLISH" in none|testpypi|pypi) ;; *) echo "publish must be none, testpypi or pypi" >&2; exit 2 ;; esac
case "$TAG" in alpha-api-*) ;; *) echo "the workflow's alpha route requires an alpha-api-* tag" >&2; exit 2 ;; esac
case "$WHL" in *manylinux*) ;; *) echo "REFUSING: $WHL is not the auditwheel-repaired wheel" >&2; exit 2 ;; esac
REPO="${MOJOLEARN_REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO"
VERSION=$(sed -n 's/^__version__ = "\(.*\)"$/\1/p' python/mojolearn/_version.py)
mkdir -p "$WORK"; WORK=$(cd "$WORK" && pwd)
ART="$WORK/artifact"; rm -rf "$ART"; mkdir -p "$ART"; cp "$WHL" "$ART/"
WSHA=$(shasum -a 256 "$WHL" | cut -d' ' -f1)

TAR=""; TSHA=""
if [ -n "${MOJOLEARN_QUAL_SM89:-}" ]; then
  echo "== stage the three columns and run the admission check =="
  Q="$WORK/qualification"; rm -rf "$Q"; mkdir -p "$Q/build-proofs" "$Q/cuda" "$Q/hip"
  cp "$MOJOLEARN_QUAL_PROOFS"/cuda-sm_89.json "$MOJOLEARN_QUAL_PROOFS"/cuda-sm_90a.json "$MOJOLEARN_QUAL_PROOFS"/hip-gfx942.json "$Q/build-proofs/"
  stage() { rsync -a --exclude venv --exclude tools-venv --exclude '*.whl' --exclude '.DS_Store' "$1/" "$2/"; }
  stage "$MOJOLEARN_QUAL_SM89" "$Q/cuda/sm_89"
  stage "$MOJOLEARN_QUAL_SM90A" "$Q/cuda/sm_90a"
  stage "$MOJOLEARN_QUAL_GFX942" "$Q/hip/gfx942"
  python3 tools/check_linux_release_qualification.py "$WHL" --qualification-root "$Q" \
    --source-root "$REPO" --profile release-linux3 | tee "$WORK/admission.json"
  python3 -c "import json,sys; sys.exit(0 if json.load(open('$WORK/admission.json')).get('status')=='PASSED' else 1)" \
    || { echo "ADMISSION FAILED; read $WORK/admission.json"; exit 1; }
  TAR="$WORK/linux-qualification.tar.gz"
  COPYFILE_DISABLE=1 tar -C "$Q" -czf "$TAR" --no-xattrs build-proofs cuda hip
  TSHA=$(shasum -a 256 "$TAR" | cut -d' ' -f1)
fi

echo "== manifest =="
python3 - "$ART/alpha-manifest.json" "$VERSION" "$(basename "$WHL")" "$WSHA" "$TSHA" <<'EOF'
import json, sys
out, version, wheel, wsha, tsha = sys.argv[1:]
doc = {"schema": "mojolearn.alpha-release.v1", "version": version, "release_profile": "alpha-api",
       "files": {wheel: wsha}}
if tsha:
    doc["linux_qualification"] = {"file": "linux-qualification.tar.gz", "sha256": tsha, "wheel": wheel}
open(out, "w").write(json.dumps(doc, indent=2, sort_keys=True) + "\n")
EOF
MSHA=$(shasum -a 256 "$ART/alpha-manifest.json" | cut -d' ' -f1)
if [ -n "$TAR" ]; then
  python3 packaging/verify_alpha_artifacts.py "$ART" --manifest-sha256 "$MSHA" --qualification-archive "$TAR" --source-root "$REPO" | tee "$WORK/file-admission.json"
else
  python3 packaging/verify_alpha_artifacts.py "$ART" --manifest-sha256 "$MSHA" | tee "$WORK/file-admission.json"
fi

echo "== GitHub release $TAG =="
HEAD_SHA=$(git rev-parse HEAD)
git rev-parse -q --verify "refs/tags/$TAG" >/dev/null || { git tag -a "$TAG" -m "mojolearn $VERSION Linux wheel" "$HEAD_SHA"; git push origin "refs/tags/$TAG"; }
if ! gh release view "$TAG" >/dev/null 2>&1; then
  NOTES="Linux x86-64 wheel with CUDA sm_89, CUDA sm_90a and HIP gfx942 sets in fast, deterministic and identical modes, built from $HEAD_SHA. See CHANGELOG.md."
  [ -n "$TAR" ] && NOTES="$NOTES linux-qualification.tar.gz is the install-and-test record on each architecture." \
                || NOTES="$NOTES Installed per-architecture qualification was not run for this release."
  # shellcheck disable=SC2086
  gh release create "$TAG" --prerelease --target "$HEAD_SHA" --title "mojolearn $VERSION Linux" --notes "$NOTES" \
    "$ART/$(basename "$WHL")" "$ART/alpha-manifest.json" ${TAR:+"$TAR"}
fi

echo "== dispatch release-provenance.yml publish=$PUBLISH =="
gh workflow run release-provenance.yml --ref "$TAG" -f publish="$PUBLISH" -f alpha_candidate_tag="$TAG" -f alpha_manifest_sha256="$MSHA"
sleep 20
RUN=$(gh run list --workflow release-provenance.yml --limit 1 --json databaseId --jq '.[0].databaseId')
echo "run $RUN"
gh run watch "$RUN" --exit-status || { echo "WORKFLOW FAILED run=$RUN"; exit 1; }
echo "DONE publish=$PUBLISH tag=$TAG wheel_sha256=$WSHA manifest_sha256=$MSHA run=$RUN"
