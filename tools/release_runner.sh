#!/bin/sh
# Bring up ONE ephemeral GitHub Actions runner on this Mac, for exactly one
# job, then exit. Nothing is left listening.
#
# WHY THIS EXISTS. The repository has no registered Actions runners, and a
# GitHub-hosted macOS runner builds a wheel with no Metal kernels in it (see
# the header of .github/workflows/release-provenance.yml: that wheel shipped
# once, as TestPyPI 0.1.0a2). A release wheel has to be built on a real Apple
# GPU, which means this machine. The release workflow's build job asks for
# `runs-on: [self-hosted, macos, arm64, metal, real-gpu]` and this script is
# the only thing that ever provides such a runner.
#
# WHY EPHEMERAL. `config.sh --ephemeral` registers a runner that accepts one
# job and then deregisters itself; `run.sh` exits when that job is done. So
# the window in which this laptop executes code from GitHub is the length of
# one deliberately dispatched release run, not "whenever the laptop is open".
#
# WHY A SELF-HOSTED RUNNER ON A PUBLIC REPOSITORY IS ACCEPTABLE HERE, and only
# under all of these conditions together:
#   * Every job that targets these labels is `workflow_dispatch` only. No
#     `push` and no `pull_request` trigger reaches this runner, so a fork
#     cannot send it code (verified 2026-08-23 against every file under
#     .github/workflows/; re-check with
#       grep -n "runs-on\|^on:\|push\|pull_request" .github/workflows/*.yml
#     before relying on it).
#   * The runner registers for ONE job and exits; it is never left online.
#   * The organization setting "Require approval for all outside
#     collaborators" (Actions -> General -> Fork pull request workflows)
#     should be ON. This script cannot read it; check it at
#       https://github.com/organizations/mojolearn/settings/actions
#   * The registration token is short-lived, requested from the API at the
#     moment of use, passed to config.sh on its command line, and never
#     written to a file by this script. (config.sh writes its own runner
#     credential under the install directory; it is removed on exit below.)
#
# USAGE
#   tools/release_runner.sh            bring up the runner and wait for a job
#   tools/release_runner.sh --dry-run  install and verify the runner package,
#                                      print what would be done, register
#                                      nothing, request no token
#   MOJOLEARN_RUNNER_DIR=<dir>         install somewhere other than
#                                      $HOME/.mojolearn-runner
#
# Then, from ANOTHER terminal, dispatch exactly one run:
#   gh workflow run release-provenance.yml -f publish=none      (build only)
#   gh workflow run release-provenance.yml -f publish=testpypi
#   gh workflow run release-provenance.yml -f publish=pypi
# The full procedure is docs/PYPI_RELEASE.md.
set -eu

REPO=mojolearn/mojolearn
RUNNER_DIR="${MOJOLEARN_RUNNER_DIR:-$HOME/.mojolearn-runner}"
RUNNER_NAME="m4-real-gpu-$(date +%s)"
LABELS=macos,arm64,metal,real-gpu   # `self-hosted` is added by GitHub itself
DRY_RUN=0
[ "${1:-}" = "--dry-run" ] && DRY_RUN=1

say() { printf '%s\n' "$*"; }
die() { printf 'release_runner: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------- gh auth ----
command -v gh >/dev/null 2>&1 || die "gh is not installed (brew install gh)"
gh auth status >/dev/null 2>&1 || die "gh is not authenticated; run: gh auth login"
command -v shasum >/dev/null 2>&1 || die "shasum not found"
command -v curl >/dev/null 2>&1 || die "curl not found"
[ "$(uname -s)" = Darwin ] || die "this script is for the Apple silicon Mac only"
[ "$(uname -m)" = arm64 ] || die "this script is for arm64 only (got $(uname -m))"

# ------------------------------------------------------ install the runner ----
# Fetch the latest actions/runner osx-arm64 tarball and verify it against the
# SHA-256 the release notes publish for that exact asset. If the checksum
# cannot be found in the notes, STOP: an unverified runner binary is the one
# piece of this that must never be installed on a guess.
if [ ! -x "$RUNNER_DIR/config.sh" ] || [ ! -x "$RUNNER_DIR/run.sh" ]; then
    say "installing the GitHub Actions runner under $RUNNER_DIR"
    rel=repos/actions/runner/releases/latest
    tag=$(gh api "$rel" -q .tag_name) || die "could not read actions/runner latest release"
    [ -n "$tag" ] || die "latest actions/runner release has no tag_name"
    asset=$(gh api "$rel" \
        -q '.assets[] | select(.name | test("^actions-runner-osx-arm64-[0-9.]+\\.tar\\.gz$")) | .name' \
        | head -1)
    [ -n "$asset" ] || die "no actions-runner-osx-arm64-*.tar.gz asset in release $tag"
    url=$(gh api "$rel" \
        -q '.assets[] | select(.name == "'"$asset"'") | .browser_download_url' | head -1)
    [ -n "$url" ] || die "no download url for $asset"
    # The release body lists one line per asset of the form
    #   - actions-runner-osx-arm64-X.Y.Z.tar.gz <!-- BEGIN SHA osx-arm64 -->HEX<!-- END SHA osx-arm64 -->
    # Take the 64-hex token on the line that names our asset and nothing else.
    body=$(gh api "$rel" -q .body)
    expected=$(printf '%s\n' "$body" | grep -F -- "$asset" \
        | grep -oE '[0-9a-f]{64}' | head -1 || true)
    [ -n "$expected" ] || die "could not parse a SHA-256 for $asset from the release notes of $tag; refusing to install an unverified runner. Read the notes at https://github.com/actions/runner/releases/tag/$tag and fix the parser in this script."
    say "runner release: $tag"
    say "asset:          $asset"
    say "expected sha:   $expected"

    tmp=$(mktemp -d)
    curl -fsSL -o "$tmp/$asset" "$url" || die "download failed: $url"
    actual=$(shasum -a 256 "$tmp/$asset" | cut -d' ' -f1)
    say "actual sha:     $actual"
    [ "$actual" = "$expected" ] || { rm -rf "$tmp"; die "SHA-256 MISMATCH for $asset; not installing"; }
    mkdir -p "$RUNNER_DIR"
    tar xzf "$tmp/$asset" -C "$RUNNER_DIR" || { rm -rf "$tmp"; die "untar failed"; }
    rm -rf "$tmp"
    [ -x "$RUNNER_DIR/config.sh" ] || die "unpacked runner has no config.sh"
    say "runner installed"
else
    say "runner already installed under $RUNNER_DIR (rm -rf it to reinstall)"
fi

# ------------------------------------------------------------- cleanup ----
# Remove the registration from GitHub if it is still there (an ephemeral
# runner that finished its job is already gone; one that never got a job, or
# was interrupted, is not), and drop the local credential config.sh wrote so
# nothing on disk can re-attach under this name.
CLEANED=0
cleanup() {
    [ "$CLEANED" = 1 ] && return 0
    CLEANED=1
    [ "$DRY_RUN" = 1 ] && return 0
    ids=$(gh api --paginate "repos/$REPO/actions/runners" \
            -q '.runners[] | select(.name == "'"$RUNNER_NAME"'") | .id' 2>/dev/null || true)
    if [ -n "$ids" ]; then
        for id in $ids; do
            if gh api -X DELETE "repos/$REPO/actions/runners/$id" >/dev/null 2>&1; then
                say "removed runner $RUNNER_NAME (id $id) from GitHub"
            else
                say "WARNING: could not remove runner $RUNNER_NAME (id $id); remove it at https://github.com/$REPO/settings/actions/runners" >&2
            fi
        done
    else
        say "runner $RUNNER_NAME is not registered on GitHub (already gone)"
    fi
    rm -f "$RUNNER_DIR/.runner" "$RUNNER_DIR/.credentials" "$RUNNER_DIR/.credentials_rsaparams"
}
trap 'cleanup' EXIT
trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

# ---------------------------------------------------------- what to run ----
say ""
say "runner name:   $RUNNER_NAME"
say "labels:        self-hosted (automatic), $LABELS"
say "repository:    https://github.com/$REPO"
say ""
say "From ANOTHER terminal, dispatch exactly one run:"
say "  gh workflow run release-provenance.yml -f publish=none       # build only"
say "  gh workflow run release-provenance.yml -f publish=testpypi"
say "  gh workflow run release-provenance.yml -f publish=pypi"
say "then: gh run watch"
say ""

if [ "$DRY_RUN" = 1 ]; then
    say "DRY RUN. Would now:"
    say "  TOKEN=\$(gh api -X POST repos/$REPO/actions/runners/registration-token -q .token)"
    say "  cd $RUNNER_DIR"
    say "  ./config.sh --url https://github.com/$REPO --token \"\$TOKEN\" --name $RUNNER_NAME \\"
    say "      --labels $LABELS --ephemeral --unattended --replace --work _work"
    say "  ./run.sh          # foreground; exits after one job (--ephemeral)"
    say "  then remove the registration from GitHub if still present"
    say "No token was requested and nothing was registered."
    exit 0
fi

# ------------------------------------------------------------- register ----
TOKEN=$(gh api -X POST "repos/$REPO/actions/runners/registration-token" -q .token) \
    || die "could not get a registration token (needs admin on $REPO)"
[ -n "$TOKEN" ] || die "empty registration token"

cd "$RUNNER_DIR"
./config.sh --url "https://github.com/$REPO" --token "$TOKEN" \
    --name "$RUNNER_NAME" --labels "$LABELS" \
    --ephemeral --unattended --replace --work _work
unset TOKEN

say ""
say "runner $RUNNER_NAME is registered and waiting for ONE job; dispatch it now."
say "Ctrl-C here removes the registration."
say ""
# Foreground. Exits after the single job because of --ephemeral.
rc=0
./run.sh || rc=$?
say "run.sh exited with $rc"
cleanup
exit "$rc"
