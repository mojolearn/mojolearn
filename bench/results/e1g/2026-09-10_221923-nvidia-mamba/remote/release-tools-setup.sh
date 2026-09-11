#!/bin/sh
set -eu
system_python=$1
destination=$2
if command -v patchelf >/dev/null 2>&1; then
    tool_source=preexisting
else
    tool_source=pinned-private-venv
    test ! -e "$destination/tools-venv" && test ! -L "$destination/tools-venv"
    "$system_python" -m venv "$destination/tools-venv"
    "$destination/tools-venv/bin/python" -m pip install --disable-pip-version-check \
        --only-binary=:all: --retries 1 --timeout 20 'patchelf==0.17.2.4'
    PATH="$destination/tools-venv/bin:$PATH"
    export PATH
fi
tool_path=$(command -v patchelf)
{
    printf 'source=%s\npath=%s\n' "$tool_source" "$tool_path"
    "$tool_path" --version
} > "$destination/release-tools-provenance.txt"
printf '%s\n' "$tool_path" > "$destination/release-patchelf-path.txt"
