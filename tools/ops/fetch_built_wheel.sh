#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# tools/ops/fetch_built_wheel.sh WHEEL_DIR
#
# THE ONE FETCH of a measurement build's packed core wheel on a GitHub runner, shared by verify-smoke.yml and
# fast-trial.yml. release-linux-build.yml uploads no wheel (the wheels are packed on the Mac by
# `tools/release.py candidate`), so tools/ops/verify_smoke_dispatch.py uploads the core wheel to R2 and dispatches
# the workflow with a presigned GET, the wheel's name and its sha256. This downloads it into WHEEL_DIR and checks
# the sha256. Environment (the workflow's inputs): WHEEL_URL, WHEEL_NAME, WHEEL_SHA256, REF (the wheel's source
# commit, 40 hex).
set -eu
[ $# -eq 1 ] || { echo "usage: $0 WHEEL_DIR" >&2; exit 2; }
[[ "$REF" =~ ^[0-9a-f]{40}$ ]] || { echo "ref must be a full commit sha" >&2; exit 2; }
[[ "$WHEEL_NAME" =~ ^[A-Za-z0-9_.+-]+\.whl$ ]] || { echo "bad wheel name" >&2; exit 2; }
[[ "$WHEEL_SHA256" =~ ^[0-9a-f]{64}$ ]] || { echo "wheel_sha256 must be a sha256" >&2; exit 2; }
mkdir -p "$1"
curl -fsSL --retry 5 "$WHEEL_URL" -o "$1/$WHEEL_NAME"
echo "$WHEEL_SHA256  $1/$WHEEL_NAME" | sha256sum -c -
