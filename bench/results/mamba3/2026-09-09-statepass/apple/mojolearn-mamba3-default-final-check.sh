#!/bin/bash
set -euo pipefail
MOJOLEARN_IDENTITY_TRACE=/tmp/mojolearn-mamba3-default-final.trace /tmp/mojolearn-mamba3-default-final > /tmp/mojolearn-mamba3-default-final.log 2>&1
cmp /tmp/mojolearn-mamba3-before.trace /tmp/mojolearn-mamba3-default-final.trace
/tmp/mojolearn-mamba3-default-final decode-cross > /tmp/mojolearn-mamba3-default-final-decode.log 2>&1
/tmp/mojolearn-mamba3-default-final continuation > /tmp/mojolearn-mamba3-default-final-continuation.log 2>&1
/tmp/mojolearn-mamba3-default-final refusal > /tmp/mojolearn-mamba3-default-final-refusal.log 2>&1
MOJOLEARN_MAMBA3_CHECK_L=65 MOJOLEARN_MAMBA3_CHECK_DM=64 MOJOLEARN_MAMBA3_CHECK_B=2 MOJOLEARN_IDENTITY_TRACE=/tmp/mojolearn-mamba3-default-final-long.trace /tmp/mojolearn-mamba3-default-final > /tmp/mojolearn-mamba3-default-final-long.log 2>&1
cmp /tmp/mojolearn-mamba3-before-long.trace /tmp/mojolearn-mamba3-default-final-long.trace
