#!/bin/sh
# lane/attention-replay-vendors: pile_github paired arms plus guarded.
set -eu
exec sh /root/mojolearn/tools/lm_attention_vendor_train_body.sh pile_github guarded
