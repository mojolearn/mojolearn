# portable_log2_64 full-range gate

`pixi run check-portable-log2-64` executes 262,144 deterministic positive
finite inputs spanning every normal exponent band plus forced subnormals,
all 2,098 representable powers of two, 82 boundary-neighbor values, and
special-value checks. The ULP comparator rejects nonfinite finite-input
results and uses ordered unsigned keys to avoid signed subtraction overflow.

Apple ARM64 and H100-pod Linux x86-64 both passed with worst error 1 ULP
against their own host libm log2. Both recorded FNV-1a64 output hash
18138053008657378164. Every power of two returned its exact integer log2.
The gate admits at most 2 ULP; it does not claim correctly rounded log2 for
all binary64 inputs. Matching sampled hashes provide independent host
agreement evidence, not an exhaustive proof. No GPU arithmetic is used here.
Source: checks/portable_log2_64_check.mojo, final gate commit fd71584a.
