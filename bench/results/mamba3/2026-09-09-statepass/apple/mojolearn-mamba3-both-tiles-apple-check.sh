#!/bin/bash
set -euo pipefail
MOJOLEARN_IDENTITY_TRACE=/tmp/mojolearn-mamba3-both-tiles-apple.trace /tmp/mojolearn-mamba3-both-tiles-apple > /tmp/mojolearn-mamba3-both-tiles-apple.log 2>&1
cmp /tmp/mojolearn-mamba3-before.trace /tmp/mojolearn-mamba3-both-tiles-apple.trace
/tmp/mojolearn-mamba3-both-tiles-apple decode-cross > /tmp/mojolearn-mamba3-both-tiles-apple-decode.log 2>&1
/tmp/mojolearn-mamba3-both-tiles-apple continuation > /tmp/mojolearn-mamba3-both-tiles-apple-continuation.log 2>&1
/tmp/mojolearn-mamba3-both-tiles-apple refusal > /tmp/mojolearn-mamba3-both-tiles-apple-refusal.log 2>&1
MOJOLEARN_MAMBA3_CHECK_L=65 MOJOLEARN_MAMBA3_CHECK_DM=64 MOJOLEARN_MAMBA3_CHECK_B=2 MOJOLEARN_IDENTITY_TRACE=/tmp/mojolearn-mamba3-both-tiles-apple-long.trace /tmp/mojolearn-mamba3-both-tiles-apple > /tmp/mojolearn-mamba3-both-tiles-apple-long.log 2>&1
cmp /tmp/mojolearn-mamba3-before-long.trace /tmp/mojolearn-mamba3-both-tiles-apple-long.trace
