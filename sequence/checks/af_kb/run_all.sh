#!/bin/sh
# One program per bisect variant (sequence/checks/af_kb/vN.mojo (each self-contained)).
for v in 0 1 2 3 4 5 6 7 8 9 10 11; do
    out=$(pixi run -e default mojo run -I . sequence/checks/af_kb/v$v.mojo 2>&1)
    echo "$out" | grep -E "^V[0-9]" || echo "V$v: NO RESULT: $(echo "$out" | grep -E "error:" | head -2 | tr '\n' ' ')"
done
