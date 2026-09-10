#!/bin/bash
set -euo pipefail
cd /Users/andrewhendel/CascadeProjects/mojolearn
unset MACOSX_DEPLOYMENT_TARGET
mojo build -j 2 --emit shared-lib --target-cpu apple-m1 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_TRANSFORMER_CALLER_TRANSFER=1 -Xlinker -platform_version -Xlinker macos -Xlinker 11.0 -Xlinker "$(xcrun --sdk macosx --show-sdk-version)" -I . -I bindings bindings/_mojolearn_transformer.mojo -o /tmp/mojolearn-transformer-transfer-apple/caller.so
cp /tmp/mojolearn-transformer-transfer-apple/caller.so python/mojolearn/identical/_mojolearn_transformer.so
export MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH=python
python tools/transformer_transfer_check.py --output /tmp/mojolearn-transformer-transfer-apple/caller.json
python python/mojolearn/tests/test_transformer_surface.py
python python/mojolearn/tests/test_transformer_hd128.py
