#!/bin/sh
export BYTE_LM_LIFETIME_CASES=stateless_x1,stateless_x2,stateless_x2_sync_teardown,stateless_x2_teardown_with_gil
exec sh /root/mojolearn/tools/byte_lm_lifetime_diag.sh
