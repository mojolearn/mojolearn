#!/bin/sh
# lane/attention-replay-vendors: enwik8 legacy/repaired pair only (post-merge confirmation).
set -eu
exec sh /root/mojolearn/tools/lm_attention_vendor_train_body.sh enwik8 none
