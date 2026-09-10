# Mamba regime diagnostic smoke

Root ran the documented explicit retained helper/spec command with --order tiny
--passes1 --rounds2 --reference-json pointing to mamba-repeat/summary.json,
under nice19/shared lock with OMP/OPENBLAS threads2 and temporary validation
Python. Exit0. Both Metal outputs match the retained complete H100 tiny hash.
The first call385.85ms versus13.75ms on the second includes startup/first-touch;
two tiny calls cannot diagnose the H100 large-shape latency regimes or qualify
a throughput price. Raw resource counters, input/binary hashes and paths remain
in the log. No library rebuild or NVIDIA run was made for this smoke test.
