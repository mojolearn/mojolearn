# Rejected GEMM per-fragment exponent admission

The experiment checks the minimum biased exponent of each register
fragment. If the two minima sum to at least151, every exact product is
on the2^-149 lattice, as is the already-flushed accumulator. Consequently
an exact FMA cannot fall inside the half-subnormal-ulp interval below the
smallest normal value. Single hardware FTZ FMA is safe in that case;
all unproved fragments retain the corrected RN FMA plus FTZ multiply.
Zeros/subnormals are ignored after operand flushing and exceptional
operands force the strict path. The scalar admitted-fragment probe passed
262144 adversarial triples; NVIDIA's seven device gates also passed.

Despite numerical success, per-fragment checking and code-generation
costs regressed dense timing substantially. It was rejected; the patch
and boundary probe are retained here solely to prevent repeating it.
The boundary probe imports helpers supplied by the rejected patch.

Same dedicated H100580.126.09, seven-call means from gemm_tuned_probe.
The binary's two displayed arms are both its own current dispatcher,
so before/after prices come from baseline-price and admitted-price logs,
not either log's internal speedup column. Numerical evidence is the
independent boundary/device gates; no cross-binary SHA claim is made.
No opponent was timed.
