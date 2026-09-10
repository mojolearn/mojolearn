#!/bin/bash
set -euo pipefail
cd /Users/andrewhendel/CascadeProjects/mojolearn
tools/with_identical_mode.sh pixi run mojo run -I . transformer/checks/transformer_fused_check.mojo > /tmp/mojolearn-transformer-exact-final-fused.log 2>&1
MOJOLEARN_IDENTITY_TRACE=/tmp/mojolearn-transformer-exact-final.card MOJOLEARN_TRANSFORMER_CHECK_CLAUSE_D=1 tools/with_identical_mode.sh pixi run check-transformer > /tmp/mojolearn-transformer-exact-final-forward.log 2>&1
cmp /tmp/mojolearn-transformer-exact-final.card bench/results/e1/2026-08-28_161700-MacBook-Air-1-terrabyte/lanes/transformer.identical.card
MOJOLEARN_IDENTITY_TRACE=/tmp/mojolearn-transformer-exact-final-backward.card tools/with_identical_mode.sh pixi run check-transformer-backward > /tmp/mojolearn-transformer-exact-final-backward.log 2>&1
cmp /tmp/mojolearn-transformer-exact-final-backward.card bench/results/apple_cards_2026-09-03/transformer-backward.identical.card
MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_transformer.sh > /tmp/mojolearn-transformer-exact-final-binding.log 2>&1
cd python
MOJOLEARN_NUMERIC_MODE=identical pixi run python -m mojolearn.tests.test_transformer_surface > /tmp/mojolearn-transformer-exact-final-surface.log 2>&1

MOJOLEARN_NUMERIC_MODE=identical pixi run python -m mojolearn.tests.test_transformer_hd128 > /tmp/mojolearn-transformer-exact-final-hd128.log 2>&1
