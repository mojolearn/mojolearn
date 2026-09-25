# shellcheck shell=sh
# tools/route_overlay_lib.sh -- THE ROUTE OVERLAY ON A BUILD BOX (2026-09-25).
#
# A release ships its frozen source commit, and the release tooling runs from
# a separate, newer checkout (tools/release_tooling.py). A build leg ships the
# SOURCE archive to its box; when the tooling's copy of a box-side tool
# (tools/release_tooling.py BOX_OVERLAY: the remote build driver, the serial
# guards, the binding cache) differs from the source's, tools/release.py hands
# the leg a tarball of the tooling copies:
#
#   MOJOLEARN_ROUTE_OVERLAY=<route-overlay.tgz>  MOJOLEARN_ROUTE_OVERLAY_SHA256=<its sha256>
#
# and the leg, after the source is unpacked on the box, extracts it over the
# source tree and records every file's sha256 before and after in
# <leg out>/route-overlay.txt. Nothing in the build's source inventory may be
# in it (a Mojo file, bindings/, packaging/, python/, tokenizer/, pixi.toml,
# pixi.lock, tools/linux_surface_qualification.sh): ro_prepare refuses such a
# tarball before anything is rented, and the source inventory the build proof
# binds is therefore the frozen commit's, byte for byte.
#
#   ro_prepare <tarball> <recorded sha256> <scratch dir>   sets RO_FILES, RO_SHA, RO_TMP
#   ro_remote_cmd <source dir on the box>                  the box script (stdin: the tarball)
#   ro_verify <route-overlay.txt>                          every file on the box is the overlaid bytes
# POSIX sh.

ro_sha256() { { shasum -a 256 "$1" 2>/dev/null || sha256sum "$1"; } | cut -d' ' -f1; }

ro_prepare() {
    RO_TGZ=$1; RO_WANT=$2; RO_TMP=$3
    [ -f "$RO_TGZ" ] || { echo "route overlay: no tarball at $RO_TGZ"; return 1; }
    RO_SHA=$(ro_sha256 "$RO_TGZ")
    if [ -z "$RO_WANT" ] || [ "$RO_SHA" != "$RO_WANT" ]; then
        echo "route overlay: sha256 $RO_SHA is not the recorded ${RO_WANT:-(none)}"; return 1
    fi
    rm -rf "$RO_TMP/route-overlay"
    mkdir -p "$RO_TMP/route-overlay" && tar -xzf "$RO_TGZ" -C "$RO_TMP/route-overlay" \
        || { echo "route overlay: $RO_TGZ does not unpack"; return 1; }
    if [ -n "$(cd "$RO_TMP/route-overlay" && find . ! -type f ! -type d)" ]; then
        echo "route overlay: refusing a link or special file"; return 1
    fi
    RO_FILES=$(cd "$RO_TMP/route-overlay" && find . -type f | sed 's|^\./||' | LC_ALL=C sort | tr '\n' ' ')
    [ -n "$RO_FILES" ] || { echo "route overlay: empty"; return 1; }
    for f in $RO_FILES; do
        case "$f" in
            *.mojo|*.mojopkg|bindings/*|packaging/*|python/*|tokenizer/*|pixi.toml|pixi.lock|tools/linux_surface_qualification.sh|/*|*..*)
                echo "route overlay: REFUSING $f, it is in the build's source inventory"; return 1 ;;
        esac
    done
    return 0
}

ro_remote_cmd() {
    printf 'cd %s || exit 1\n' "$1"
    printf 'for f in %s; do if [ -f "$f" ]; then printf "before %%s %%s\\n" "$f" "$(sha256sum "$f" | cut -c1-64)"; else printf "before %%s absent\\n" "$f"; fi; done\n' "$RO_FILES"
    printf 'tar -xzf - || exit 1\n'
    printf 'for f in %s; do printf "after %%s %%s\\n" "$f" "$(sha256sum "$f" | cut -c1-64)"; done\n' "$RO_FILES"
}

ro_verify() {
    for f in $RO_FILES; do
        want=$(ro_sha256 "$RO_TMP/route-overlay/$f")
        grep -qx "after $f $want" "$1" || { echo "route overlay: $f on the box is not the overlaid bytes"; return 1; }
    done
    printf 'overlay_sha256=%s\n' "$RO_SHA" >> "$1"
    return 0
}
