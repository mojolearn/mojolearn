#!/bin/sh
export BYTE_LM_LIFETIME_CASES=stateless_x1,stateless_x2,resident_close_reopen
exec sh /root/mojolearn/tools/byte_lm_lifetime_diag.sh
