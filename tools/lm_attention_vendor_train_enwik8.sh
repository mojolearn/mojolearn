#!/bin/sh
# lane/attention-replay-vendors: enwik8 paired arms plus released_legacy.
set -eu
exec sh /root/mojolearn/tools/lm_attention_vendor_train_body.sh enwik8 released_legacy
