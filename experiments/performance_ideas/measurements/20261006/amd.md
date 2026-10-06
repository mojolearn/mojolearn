# Candidate A/B measurements

One excluded warmup and one scored sample. Identity and compilation are reused; no separate retests.
Component and public-caller fixtures retain their stated scope. Full-workload results and opponent comparisons require their own measurements. Default decisions are recorded beside source toggles; this board does not change them.

| Candidate | Mode | Measurement status | Captured pairs |
|---|---|---|---:|
| A01 | identical | PARTIAL_MEASUREMENTS_RETAINED | 60 |
| A02 | identical | PARTIAL_MEASUREMENTS_RETAINED | 21 |
| A03 | identical | PARTIAL_MEASUREMENTS_RETAINED | 42 |
| A04 | identical | PARTIAL_MEASUREMENTS_RETAINED | 3 |
| A05 | identical | PARTIAL_MEASUREMENTS_RETAINED | 21 |
| A06 | identical | NO_DISTINCT_RUNTIME_ARM | 0 |
| A07 | identical | PARTIAL_MEASUREMENTS_RETAINED | 3 |
| A08 | identical | PARTIAL_MEASUREMENTS_RETAINED | 12 |
| I01 | identical | PENDING_MEASUREMENT | 0 |
| I02 | identical | PARTIAL_MEASUREMENTS_RETAINED | 2 |
| I03 | identical | PARTIAL_MEASUREMENTS_RETAINED | 9 |
| I04 | identical | PARTIAL_MEASUREMENTS_RETAINED | 2 |
| I05 | identical | PARTIAL_MEASUREMENTS_RETAINED | 9 |
| I06 | identical | PARTIAL_MEASUREMENTS_RETAINED | 4 |
| I07 | identical | PARTIAL_MEASUREMENTS_RETAINED | 2 |
| I08 | identical | PARTIAL_MEASUREMENTS_RETAINED | 2 |
| I09 | identical | PARTIAL_MEASUREMENTS_RETAINED | 2 |
| I10 | identical | PARTIAL_MEASUREMENTS_RETAINED | 1 |
| I11 | identical | PARTIAL_MEASUREMENTS_RETAINED | 2 |
| I12 | identical | PARTIAL_MEASUREMENTS_RETAINED | 2 |
| I13 | identical | PARTIAL_MEASUREMENTS_RETAINED | 2 |
| I14 | identical | PARTIAL_MEASUREMENTS_RETAINED | 2 |
| I15 | identical | NO_DISTINCT_RUNTIME_ARM | 0 |
| I16 | identical | PARTIAL_MEASUREMENTS_RETAINED | 18 |
| I17 | identical | PARTIAL_MEASUREMENTS_RETAINED | 18 |
| I18 | identical | PARTIAL_MEASUREMENTS_RETAINED | 3 |
| I19 | identical | PARTIAL_MEASUREMENTS_RETAINED | 9 |
| I20 | identical | PARTIAL_MEASUREMENTS_RETAINED | 4 |
| I21 | identical | PARTIAL_MEASUREMENTS_RETAINED | 3 |
| I22 | identical | PARTIAL_MEASUREMENTS_RETAINED | 3 |
| I23 | identical | NO_DISTINCT_RUNTIME_ARM | 0 |
| I24 | identical | PARTIAL_MEASUREMENTS_RETAINED | 4 |
| N01 | identical | PENDING_MEASUREMENT | 0 |
| N02 | identical | PENDING_MEASUREMENT | 0 |
| N04 | identical | PENDING_MEASUREMENT | 0 |
| N05 | identical | PARTIAL_MEASUREMENTS_RETAINED | 27 |
| N06 | identical | PARTIAL_MEASUREMENTS_RETAINED | 1 |
| N07 | identical | PARTIAL_MEASUREMENTS_RETAINED | 1 |
| N08 | identical | PARTIAL_MEASUREMENTS_RETAINED | 1 |

## Captured evidence

| Candidate | Vendor / route | Case | Scope | Status | B/A time | Evidence |
|---|---|---|---|---|---:|---|
| A01 | amd/gfx942 | 63x66x385-op0-mfma16_off | component | MEASURED | 0.7897 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 63x66x385-op1-mfma16_off | component | MEASURED | 0.6730 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 63x66x385-op2-mfma16_off | component | MEASURED | 0.6104 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 63x66x513-op0-mfma16_off | component | MEASURED | 0.8795 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 63x66x513-op1-mfma16_off | component | MEASURED | 0.6417 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 63x66x513-op2-mfma16_off | component | MEASURED | 0.6022 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 63x66x1025-op0-mfma16_off | component | MEASURED | 0.8100 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 63x66x1025-op1-mfma16_off | component | MEASURED | 0.8451 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 63x66x1025-op2-mfma16_off | component | MEASURED | 0.6127 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 127x130x385-op0-mfma16_off | component | MEASURED | 0.7051 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 127x130x385-op1-mfma16_off | component | MEASURED | 1.1985 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 127x130x385-op2-mfma16_off | component | MEASURED | 1.0524 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 127x130x513-op0-mfma16_off | component | MEASURED | 1.0130 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 127x130x513-op1-mfma16_off | component | MEASURED | 1.1516 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 127x130x513-op2-mfma16_off | component | MEASURED | 1.0427 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 127x130x1025-op0-mfma16_off | component | MEASURED | 0.8064 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 127x130x1025-op1-mfma16_off | component | MEASURED | 1.0488 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 127x130x1025-op2-mfma16_off | component | MEASURED | 1.0872 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 255x258x385-op0-mfma16_off | component | MEASURED | 0.6873 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 255x258x385-op1-mfma16_off | component | MEASURED | 1.1671 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 255x258x385-op2-mfma16_off | component | MEASURED | 1.0591 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 255x258x513-op0-mfma16_off | component | MEASURED | 0.9724 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 255x258x513-op1-mfma16_off | component | MEASURED | 1.1763 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 255x258x513-op2-mfma16_off | component | MEASURED | 1.0674 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 255x258x1025-op0-mfma16_off | component | MEASURED | 0.7607 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 255x258x1025-op1-mfma16_off | component | MEASURED | 1.0254 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 255x258x1025-op2-mfma16_off | component | MEASURED | 0.9189 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 4096x1024x1024-op0-mfma16_off | component | MEASURED | 1.0134 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 4096x1024x1024-op1-mfma16_off | component | MEASURED | 1.0097 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 4096x1024x1024-op2-mfma16_off | component | MEASURED | 0.9696 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 63x66x385-op0-band_off | component | MEASURED | 1.0491 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 63x66x385-op1-band_off | component | MEASURED | 1.0927 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 63x66x385-op2-band_off | component | MEASURED | 1.0918 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 63x66x513-op0-band_off | component | MEASURED | 1.1313 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 63x66x513-op1-band_off | component | MEASURED | 0.9929 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 63x66x513-op2-band_off | component | MEASURED | 1.0636 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 63x66x1025-op0-band_off | component | MEASURED | 0.9997 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 63x66x1025-op1-band_off | component | MEASURED | 1.0099 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 63x66x1025-op2-band_off | component | MEASURED | 1.0184 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 127x130x385-op0-band_off | component | MEASURED | 0.8655 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 127x130x385-op1-band_off | component | MEASURED | 0.9409 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 127x130x385-op2-band_off | component | MEASURED | 0.9576 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 127x130x513-op0-band_off | component | MEASURED | 1.0878 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 127x130x513-op1-band_off | component | MEASURED | 0.9204 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 127x130x513-op2-band_off | component | MEASURED | 1.0185 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 127x130x1025-op0-band_off | component | MEASURED | 1.0012 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 127x130x1025-op1-band_off | component | MEASURED | 0.8702 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 127x130x1025-op2-band_off | component | MEASURED | 0.9583 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 255x258x385-op0-band_off | component | MEASURED | 0.7906 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 255x258x385-op1-band_off | component | MEASURED | 1.0292 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 255x258x385-op2-band_off | component | MEASURED | 1.0020 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 255x258x513-op0-band_off | component | MEASURED | 1.2026 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 255x258x513-op1-band_off | component | MEASURED | 1.0686 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 255x258x513-op2-band_off | component | MEASURED | 1.0187 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 255x258x1025-op0-band_off | component | MEASURED | 1.0373 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 255x258x1025-op1-band_off | component | MEASURED | 1.0310 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 255x258x1025-op2-band_off | component | MEASURED | 1.0191 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 4096x1024x1024-op0-band_off | component | MEASURED | 1.2172 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 4096x1024x1024-op1-band_off | component | MEASURED | 2.1045 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A01 | amd/gfx942 | 4096x1024x1024-op2-band_off | component | MEASURED | 2.0671 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A02 | amd/gfx942 | 127x130x257-op0-one_page | component | MEASURED | 0.7562 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A02 | amd/gfx942 | 127x130x257-op1-one_page | component | MEASURED | 0.9553 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A02 | amd/gfx942 | 127x130x257-op2-one_page | component | MEASURED | 1.0787 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A02 | amd/gfx942 | 127x130x1025-op0-one_page | component | MEASURED | 0.9728 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A02 | amd/gfx942 | 127x130x1025-op1-one_page | component | MEASURED | 0.9940 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A02 | amd/gfx942 | 127x130x1025-op2-one_page | component | MEASURED | 0.9992 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A02 | amd/gfx942 | 127x130x4097-op0-one_page | component | MEASURED | 0.8588 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A02 | amd/gfx942 | 127x130x4097-op1-one_page | component | MEASURED | 0.9255 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A02 | amd/gfx942 | 127x130x4097-op2-one_page | component | MEASURED | 1.0773 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A02 | amd/gfx942 | 511x514x257-op0-one_page | component | MEASURED | 0.8681 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A02 | amd/gfx942 | 511x514x257-op1-one_page | component | MEASURED | 1.0040 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A02 | amd/gfx942 | 511x514x257-op2-one_page | component | MEASURED | 0.9955 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A02 | amd/gfx942 | 511x514x1025-op0-one_page | component | MEASURED | 0.9266 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A02 | amd/gfx942 | 511x514x1025-op1-one_page | component | MEASURED | 1.0078 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A02 | amd/gfx942 | 511x514x1025-op2-one_page | component | MEASURED | 1.0231 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A02 | amd/gfx942 | 511x514x4097-op0-one_page | component | MEASURED | 1.2175 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A02 | amd/gfx942 | 511x514x4097-op1-one_page | component | MEASURED | 0.9991 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A02 | amd/gfx942 | 511x514x4097-op2-one_page | component | MEASURED | 1.0059 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A02 | amd/gfx942 | 4096x1024x1024-op0-one_page | component | MEASURED | 0.8842 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A02 | amd/gfx942 | 4096x1024x1024-op1-one_page | component | MEASURED | 1.0977 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A02 | amd/gfx942 | 4096x1024x1024-op2-one_page | component | MEASURED | 1.1336 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 127x130x257-op0-pad0 | component | MEASURED | 1.2482 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 127x130x257-op1-pad0 | component | MEASURED | 1.0569 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 127x130x257-op2-pad0 | component | MEASURED | 1.0059 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 127x130x1025-op0-pad0 | component | MEASURED | 1.1741 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 127x130x1025-op1-pad0 | component | MEASURED | 1.0379 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 127x130x1025-op2-pad0 | component | MEASURED | 0.9757 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 127x130x4097-op0-pad0 | component | MEASURED | 0.9990 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 127x130x4097-op1-pad0 | component | MEASURED | 0.9603 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 127x130x4097-op2-pad0 | component | MEASURED | 1.1663 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 511x514x257-op0-pad0 | component | MEASURED | 1.1384 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 511x514x257-op1-pad0 | component | MEASURED | 0.9978 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 511x514x257-op2-pad0 | component | MEASURED | 1.0148 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 511x514x1025-op0-pad0 | component | MEASURED | 0.9893 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 511x514x1025-op1-pad0 | component | MEASURED | 1.0066 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 511x514x1025-op2-pad0 | component | MEASURED | 1.0222 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 511x514x4097-op0-pad0 | component | MEASURED | 1.0918 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 511x514x4097-op1-pad0 | component | MEASURED | 1.0345 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 511x514x4097-op2-pad0 | component | MEASURED | 0.9971 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 4096x1024x1024-op0-pad0 | component | MEASURED | 0.8842 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 4096x1024x1024-op1-pad0 | component | MEASURED | 0.9941 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 4096x1024x1024-op2-pad0 | component | MEASURED | 0.9841 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 127x130x257-op0-pad8 | component | MEASURED | 1.0385 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 127x130x257-op1-pad8 | component | MEASURED | 0.9894 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 127x130x257-op2-pad8 | component | MEASURED | 1.0582 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 127x130x1025-op0-pad8 | component | MEASURED | 1.0017 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 127x130x1025-op1-pad8 | component | MEASURED | 1.0137 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 127x130x1025-op2-pad8 | component | MEASURED | 1.0583 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 127x130x4097-op0-pad8 | component | MEASURED | 1.1161 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 127x130x4097-op1-pad8 | component | MEASURED | 0.8983 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 127x130x4097-op2-pad8 | component | MEASURED | 0.9664 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 511x514x257-op0-pad8 | component | MEASURED | 0.8686 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 511x514x257-op1-pad8 | component | MEASURED | 1.0032 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 511x514x257-op2-pad8 | component | MEASURED | 1.0141 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 511x514x1025-op0-pad8 | component | MEASURED | 1.1302 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 511x514x1025-op1-pad8 | component | MEASURED | 1.0217 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 511x514x1025-op2-pad8 | component | MEASURED | 1.0186 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 511x514x4097-op0-pad8 | component | MEASURED | 0.8911 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 511x514x4097-op1-pad8 | component | MEASURED | 0.9894 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 511x514x4097-op2-pad8 | component | MEASURED | 0.9959 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 4096x1024x1024-op0-pad8 | component | MEASURED | 0.8685 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 4096x1024x1024-op1-pad8 | component | MEASURED | 0.9807 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A03 | amd/gfx942 | 4096x1024x1024-op2-pad8 | component | MEASURED | 0.9299 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A04 | amd/gfx942 | {"AB_GROUPS": "4096"}/{"groups": 4096}/1 | component | MEASURED | 0.8017 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| A04 | amd/gfx942 | {"AB_GROUPS": "65537"}/{"groups": 65537}/1 | component | MEASURED | 0.7400 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| A04 | amd/gfx942 | {"AB_GROUPS": "1048576"}/{"groups": 1048576}/1 | component | MEASURED | 0.5378 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| A05 | amd/gfx942 | 127x130x257-op0-compact_rows | component | MEASURED | 2.5126 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A05 | amd/gfx942 | 127x130x257-op1-compact_rows | component | MEASURED | 3.0126 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A05 | amd/gfx942 | 127x130x257-op2-compact_rows | component | MEASURED | 3.0489 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A05 | amd/gfx942 | 127x130x1025-op0-compact_rows | component | MEASURED | 1.6996 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A05 | amd/gfx942 | 127x130x1025-op1-compact_rows | component | MEASURED | 10.2405 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A05 | amd/gfx942 | 127x130x1025-op2-compact_rows | component | MEASURED | 9.2968 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A05 | amd/gfx942 | 127x130x4097-op0-compact_rows | component | MEASURED | 4.1922 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A05 | amd/gfx942 | 127x130x4097-op1-compact_rows | component | MEASURED | 28.7112 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A05 | amd/gfx942 | 127x130x4097-op2-compact_rows | component | MEASURED | 25.7739 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A05 | amd/gfx942 | 511x514x257-op0-compact_rows | component | MEASURED | 1.0720 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A05 | amd/gfx942 | 511x514x257-op1-compact_rows | component | MEASURED | 1.6579 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A05 | amd/gfx942 | 511x514x257-op2-compact_rows | component | MEASURED | 1.4338 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A05 | amd/gfx942 | 511x514x1025-op0-compact_rows | component | MEASURED | 1.7229 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A05 | amd/gfx942 | 511x514x1025-op1-compact_rows | component | MEASURED | 7.4143 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A05 | amd/gfx942 | 511x514x1025-op2-compact_rows | component | MEASURED | 6.9426 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A05 | amd/gfx942 | 511x514x4097-op0-compact_rows | component | MEASURED | 4.3437 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A05 | amd/gfx942 | 511x514x4097-op1-compact_rows | component | MEASURED | 14.7744 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A05 | amd/gfx942 | 511x514x4097-op2-compact_rows | component | MEASURED | 15.1515 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A05 | amd/gfx942 | 4096x1024x1024-op0-compact_rows | component | MEASURED | 0.9811 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A05 | amd/gfx942 | 4096x1024x1024-op1-compact_rows | component | MEASURED | 1.6291 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A05 | amd/gfx942 | 4096x1024x1024-op2-compact_rows | component | MEASURED | 1.6246 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/capture/artifacts/summary.json |
| A06 | amd/gfx942 | {"AB_FEATURES": 2, "AB_K": 8, "AB_QUERIES": 2000, "AB_ROWS": 100000} | public_caller_component | NO_DISTINCT_RUNTIME_ARM | — | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repair-summary.json |
| A06 | amd/gfx942 | {"AB_FEATURES": 8, "AB_K": 8, "AB_QUERIES": 2000, "AB_ROWS": 100000} | public_caller_component | NO_DISTINCT_RUNTIME_ARM | — | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repair-summary.json |
| A06 | amd/gfx942 | {"AB_FEATURES": 17, "AB_K": 8, "AB_QUERIES": 2000, "AB_ROWS": 100000} | public_caller_component | NO_DISTINCT_RUNTIME_ARM | — | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repair-summary.json |
| A07 | amd/gfx942 | {"AB_FEATURES": 32, "AB_ROWS": 100000}-production_tasks256 | public_caller_component | MEASURED | 0.9929 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repair-summary.json |
| A07 | amd/gfx942 | {"AB_FEATURES": 33, "AB_ROWS": 100001}-production_tasks256 | public_caller_component | MEASURED | 0.9952 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repair-summary.json |
| A07 | amd/gfx942 | {"AB_FEATURES": 17, "AB_ROWS": 65537}-production_tasks256 | public_caller_component | MEASURED | 0.9978 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repair-summary.json |
| A08 | amd/gfx942 | {"AB_FEATURES": 32, "AB_ROWS": 100000}-old1024 | public_caller_component | MEASURED | 1.2049 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repair-summary.json |
| A08 | amd/gfx942 | {"AB_FEATURES": 33, "AB_ROWS": 100001}-old1024 | public_caller_component | MEASURED | 1.2096 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repair-summary.json |
| A08 | amd/gfx942 | {"AB_FEATURES": 17, "AB_ROWS": 65537}-old1024 | public_caller_component | MEASURED | 1.4259 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repair-summary.json |
| A08 | amd/gfx942 | {"AB_FEATURES": 32, "AB_ROWS": 100000}-wide4096 | public_caller_component | MEASURED | 2.8525 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repair-summary.json |
| A08 | amd/gfx942 | {"AB_FEATURES": 33, "AB_ROWS": 100001}-wide4096 | public_caller_component | MEASURED | 2.8428 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repair-summary.json |
| A08 | amd/gfx942 | {"AB_FEATURES": 17, "AB_ROWS": 65537}-wide4096 | public_caller_component | MEASURED | 4.4471 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repair-summary.json |
| A08 | amd/gfx942 | {"AB_FEATURES": 32, "AB_ROWS": 100000}-256_conv_poll1 | public_caller_component | MEASURED | 0.8329 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repair-summary.json |
| A08 | amd/gfx942 | {"AB_FEATURES": 33, "AB_ROWS": 100001}-256_conv_poll1 | public_caller_component | MEASURED | 0.8386 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repair-summary.json |
| A08 | amd/gfx942 | {"AB_FEATURES": 17, "AB_ROWS": 65537}-256_conv_poll1 | public_caller_component | MEASURED | 0.9885 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repair-summary.json |
| A08 | amd/gfx942 | {"AB_FEATURES": 32, "AB_ROWS": 100000}-1024_conv_poll1 | public_caller_component | MEASURED | 1.1433 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repair-summary.json |
| A08 | amd/gfx942 | {"AB_FEATURES": 33, "AB_ROWS": 100001}-1024_conv_poll1 | public_caller_component | MEASURED | 1.1635 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repair-summary.json |
| A08 | amd/gfx942 | {"AB_FEATURES": 17, "AB_ROWS": 65537}-1024_conv_poll1 | public_caller_component | MEASURED | 1.4608 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repair-summary.json |
| I02 | amd/gfx942 | {}/{"k": 2048, "m": 1024, "n": 1024}/1 | component | MEASURED | 0.3582 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I02 | amd/gfx942 | {"AB_K": "2049", "AB_M": "1023", "AB_N": "1025"}/{"k": 2049, "m": 1023, "n": 1025}/1 | component | MEASURED | 0.3993 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I03 | amd/gfx942 | {"AB_K": "1025", "AB_KIND": "2", "AB_M": "1025", "AB_N": "513"}/{"jobs": 3, "k": 1025, "kind": 2, "m": 1025, "n": 513, "op": 0, "version": 0} | component | MEASURED | 0.9876 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I03 | amd/gfx942 | {"AB_K": "1025", "AB_KIND": "2", "AB_M": "1025", "AB_N": "513"}/{"jobs": 3, "k": 1025, "kind": 2, "m": 1025, "n": 513, "op": 1, "version": 0} | component | MEASURED | 0.9805 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I03 | amd/gfx942 | {"AB_K": "1025", "AB_KIND": "2", "AB_M": "1025", "AB_N": "513"}/{"jobs": 3, "k": 1025, "kind": 2, "m": 1025, "n": 513, "op": 2, "version": 0} | component | MEASURED | 0.9676 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I03 | amd/gfx942 | {"AB_K": "1025", "AB_KIND": "2", "AB_M": "257", "AB_N": "259"}/{"jobs": 3, "k": 1025, "kind": 2, "m": 257, "n": 259, "op": 0, "version": 0} | component | MEASURED | 0.3523 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I03 | amd/gfx942 | {"AB_K": "1025", "AB_KIND": "2", "AB_M": "257", "AB_N": "259"}/{"jobs": 3, "k": 1025, "kind": 2, "m": 257, "n": 259, "op": 1, "version": 0} | component | MEASURED | 1.5606 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I03 | amd/gfx942 | {"AB_K": "1025", "AB_KIND": "2", "AB_M": "257", "AB_N": "259"}/{"jobs": 3, "k": 1025, "kind": 2, "m": 257, "n": 259, "op": 2, "version": 0} | component | MEASURED | 0.4003 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I03 | amd/gfx942 | {"AB_K": "2049", "AB_KIND": "2", "AB_M": "1023", "AB_N": "1025"}/{"jobs": 3, "k": 2049, "kind": 2, "m": 1023, "n": 1025, "op": 0, "version": 0} | component | MEASURED | 0.9309 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I03 | amd/gfx942 | {"AB_K": "2049", "AB_KIND": "2", "AB_M": "1023", "AB_N": "1025"}/{"jobs": 3, "k": 2049, "kind": 2, "m": 1023, "n": 1025, "op": 1, "version": 0} | component | MEASURED | 1.0717 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I03 | amd/gfx942 | {"AB_K": "2049", "AB_KIND": "2", "AB_M": "1023", "AB_N": "1025"}/{"jobs": 3, "k": 2049, "kind": 2, "m": 1023, "n": 1025, "op": 2, "version": 0} | component | MEASURED | 0.9754 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I04 | amd/gfx942 | {}/{"k": 2048, "m": 1024, "n": 1024}/candidate | component | MEASURED | 1.2681 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I04 | amd/gfx942 | {"AB_K": "2049", "AB_M": "1023", "AB_N": "1025"}/{"k": 2049, "m": 1023, "n": 1025}/candidate | component | MEASURED | 1.1568 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I05 | amd/gfx942 | {"AB_K": "1025", "AB_KIND": "0", "AB_M": "1025", "AB_N": "513"}/{"jobs": 3, "k": 1025, "kind": 0, "m": 1025, "n": 513, "op": 0, "version": 0} | component | MEASURED | 0.9580 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I05 | amd/gfx942 | {"AB_K": "1025", "AB_KIND": "0", "AB_M": "1025", "AB_N": "513"}/{"jobs": 3, "k": 1025, "kind": 0, "m": 1025, "n": 513, "op": 1, "version": 0} | component | MEASURED | 1.0256 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I05 | amd/gfx942 | {"AB_K": "1025", "AB_KIND": "0", "AB_M": "1025", "AB_N": "513"}/{"jobs": 3, "k": 1025, "kind": 0, "m": 1025, "n": 513, "op": 2, "version": 0} | component | MEASURED | 0.9840 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I05 | amd/gfx942 | {"AB_K": "1025", "AB_KIND": "0", "AB_M": "257", "AB_N": "259"}/{"jobs": 3, "k": 1025, "kind": 0, "m": 257, "n": 259, "op": 0, "version": 0} | component | MEASURED | 0.6908 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I05 | amd/gfx942 | {"AB_K": "1025", "AB_KIND": "0", "AB_M": "257", "AB_N": "259"}/{"jobs": 3, "k": 1025, "kind": 0, "m": 257, "n": 259, "op": 1, "version": 0} | component | MEASURED | 0.9902 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I05 | amd/gfx942 | {"AB_K": "1025", "AB_KIND": "0", "AB_M": "257", "AB_N": "259"}/{"jobs": 3, "k": 1025, "kind": 0, "m": 257, "n": 259, "op": 2, "version": 0} | component | MEASURED | 0.9673 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I05 | amd/gfx942 | {"AB_K": "2049", "AB_KIND": "0", "AB_M": "1023", "AB_N": "1025"}/{"jobs": 3, "k": 2049, "kind": 0, "m": 1023, "n": 1025, "op": 0, "version": 0} | component | MEASURED | 0.9844 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I05 | amd/gfx942 | {"AB_K": "2049", "AB_KIND": "0", "AB_M": "1023", "AB_N": "1025"}/{"jobs": 3, "k": 2049, "kind": 0, "m": 1023, "n": 1025, "op": 1, "version": 0} | component | MEASURED | 1.0668 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I05 | amd/gfx942 | {"AB_K": "2049", "AB_KIND": "0", "AB_M": "1023", "AB_N": "1025"}/{"jobs": 3, "k": 2049, "kind": 0, "m": 1023, "n": 1025, "op": 2, "version": 0} | component | MEASURED | 0.9941 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I06 | amd/gfx942 | amd-I06-baseline-e80a1d0a04-I06-shape-06d7de6f96d8 | component | NO_DISTINCT_RUNTIME_ARM | — | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I06 | amd/gfx942 | amd-I06-baseline-e80a1d0a04-I06-shape-87b919cf00ea | component | NO_DISTINCT_RUNTIME_ARM | — | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I06 | amd/gfx942 | {}/{"heads": 12, "kv_heads": 4, "length": 1024}/candidate | component | MEASURED | 0.1164 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I06 | amd/gfx942 | {"AB_HEADS": "12", "AB_KV_HEADS": "4", "AB_LENGTH": "1536"}/{"heads": 12, "kv_heads": 4, "length": 1536}/candidate | component | MEASURED | 0.1157 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I06 | amd/gfx942 | {"AB_HEADS": "8", "AB_KV_HEADS": "4", "AB_LENGTH": "1024"}/{"heads": 8, "kv_heads": 4, "length": 1024}/candidate | component | MEASURED | 1.2109 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I06 | amd/gfx942 | {"AB_HEADS": "8", "AB_KV_HEADS": "4", "AB_LENGTH": "1536"}/{"heads": 8, "kv_heads": 4, "length": 1536}/candidate | component | MEASURED | 1.2123 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I07 | amd/gfx942 | amd-I07-candidate-I07-0 | component | NO_DISTINCT_RUNTIME_ARM | — | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I07 | amd/gfx942 | amd-I07-candidate-I07-1 | component | NO_DISTINCT_RUNTIME_ARM | — | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I07 | amd/gfx942 | amd-I07-baseline-I07-0 | component | NO_DISTINCT_RUNTIME_ARM | — | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I07 | amd/gfx942 | amd-I07-baseline-I07-1 | component | NO_DISTINCT_RUNTIME_ARM | — | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I07 | amd/gfx942 | {"AB_HEADS": "12", "AB_KV_HEADS": "4", "AB_LENGTH": "1024"}/{"heads": 12, "kv_heads": 4, "length": 1024}/candidate | component | MEASURED | 0.5604 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I07 | amd/gfx942 | {"AB_HEADS": "12", "AB_KV_HEADS": "4", "AB_LENGTH": "1536"}/{"heads": 12, "kv_heads": 4, "length": 1536}/candidate | component | MEASURED | 0.5862 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I08 | amd/gfx942 | {"AB_CASE": "1", "AB_LENGTH": "513"}/{"features": 32, "length": 513}/candidate | public_caller_component | MEASURED | 0.9945 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I08 | amd/gfx942 | {"AB_CASE": "1", "AB_LENGTH": "1025"}/{"features": 32, "length": 1025}/candidate | public_caller_component | MEASURED | 1.0250 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I09 | amd/gfx942 | {"AB_FEATURES": "64", "AB_LENGTH": "512"}/{"features": 64, "length": 512}/candidate | public_caller_component | MEASURED | 0.8128 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I09 | amd/gfx942 | {"AB_FEATURES": "64", "AB_LENGTH": "1025"}/{"features": 64, "length": 1025}/candidate | public_caller_component | MEASURED | 0.8104 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I10 | amd/gfx942 | {}/{"parameters": 34944}/candidate | public_caller_component | MEASURED | 0.9930 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I11 | amd/gfx942 | {"AB_FEATURES": "64", "AB_ROWS": "65536", "AB_VOCAB": "4096"}/{"features": 64, "rows": 65536, "vocab": 4096}/candidate | public_caller_component | MEASURED | 0.9203 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I11 | amd/gfx942 | {"AB_FEATURES": "32", "AB_ROWS": "131071", "AB_VOCAB": "8192"}/{"features": 32, "rows": 131071, "vocab": 8192}/candidate | public_caller_component | MEASURED | 0.9585 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I12 | amd/gfx942 | {"AB_FEATURES": "32", "AB_ROWS": "100000"}/{"features": 32, "rows": 100000}/candidate | public_caller_component | MEASURED | 2.7844 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I12 | amd/gfx942 | {"AB_FEATURES": "17", "AB_ROWS": "131071"}/{"features": 17, "rows": 131071}/candidate | public_caller_component | MEASURED | 2.3187 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I13 | amd/gfx942 | {"AB_DEGREE": "32", "AB_ROWS": "100000"}/{"degree": 32, "rows": 100000}/candidate | public_caller_component | MEASURED | 3.6406 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I13 | amd/gfx942 | {"AB_DEGREE": "16", "AB_ROWS": "131071"}/{"degree": 16, "rows": 131071}/candidate | public_caller_component | MEASURED | 5.6492 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I14 | amd/gfx942 | {"AB_ROWS": "100000"}/{"passes": 2, "rows": 100000}/candidate | component | MEASURED | 1.1765 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I14 | amd/gfx942 | {"AB_ROWS": "131071"}/{"passes": 2, "rows": 131071}/candidate | component | MEASURED | 0.9785 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I15 | amd/gfx942 | amd-I15-baseline-I15-0 | public_caller_component | NO_DISTINCT_RUNTIME_ARM | — | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I15 | amd/gfx942 | amd-I15-baseline-I15-1 | public_caller_component | NO_DISTINCT_RUNTIME_ARM | — | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I15 | amd/gfx942 | amd-I15-baseline-I15-2 | public_caller_component | NO_DISTINCT_RUNTIME_ARM | — | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I15 | amd/gfx942 | amd-I15-candidate-I15-0 | public_caller_component | NO_DISTINCT_RUNTIME_ARM | — | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I15 | amd/gfx942 | amd-I15-candidate-I15-1 | public_caller_component | NO_DISTINCT_RUNTIME_ARM | — | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I15 | amd/gfx942 | amd-I15-candidate-I15-2 | public_caller_component | NO_DISTINCT_RUNTIME_ARM | — | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I16 | amd/gfx942 | {"AB_FEATURES": "7", "AB_QUERIES": "128", "AB_ROWS": "100000"}/{"features": 7, "occupancy": 0, "queries": 128, "rows": 100000}/1 | public_caller_component | MEASURED | 2.1066 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I16 | amd/gfx942 | {"AB_FEATURES": "7", "AB_QUERIES": "128", "AB_ROWS": "100000"}/{"features": 7, "occupancy": 0, "queries": 128, "rows": 100000}/2 | public_caller_component | MEASURED | 16.5098 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I16 | amd/gfx942 | {"AB_FEATURES": "7", "AB_QUERIES": "128", "AB_ROWS": "100000"}/{"features": 7, "occupancy": 1, "queries": 128, "rows": 100000}/1 | public_caller_component | MEASURED | 1.6972 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I16 | amd/gfx942 | {"AB_FEATURES": "7", "AB_QUERIES": "128", "AB_ROWS": "100000"}/{"features": 7, "occupancy": 1, "queries": 128, "rows": 100000}/2 | public_caller_component | MEASURED | 13.2682 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I16 | amd/gfx942 | {"AB_FEATURES": "7", "AB_QUERIES": "128", "AB_ROWS": "100000"}/{"features": 7, "occupancy": 2, "queries": 128, "rows": 100000}/1 | public_caller_component | MEASURED | 2.0218 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I16 | amd/gfx942 | {"AB_FEATURES": "7", "AB_QUERIES": "128", "AB_ROWS": "100000"}/{"features": 7, "occupancy": 2, "queries": 128, "rows": 100000}/2 | public_caller_component | MEASURED | 15.4145 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I16 | amd/gfx942 | {"AB_FEATURES": "33", "AB_QUERIES": "128", "AB_ROWS": "100000"}/{"features": 33, "occupancy": 0, "queries": 128, "rows": 100000}/1 | public_caller_component | MEASURED | 2.2001 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I16 | amd/gfx942 | {"AB_FEATURES": "33", "AB_QUERIES": "128", "AB_ROWS": "100000"}/{"features": 33, "occupancy": 0, "queries": 128, "rows": 100000}/2 | public_caller_component | MEASURED | 7.6709 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I16 | amd/gfx942 | {"AB_FEATURES": "33", "AB_QUERIES": "128", "AB_ROWS": "100000"}/{"features": 33, "occupancy": 1, "queries": 128, "rows": 100000}/1 | public_caller_component | MEASURED | 2.0173 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I16 | amd/gfx942 | {"AB_FEATURES": "33", "AB_QUERIES": "128", "AB_ROWS": "100000"}/{"features": 33, "occupancy": 1, "queries": 128, "rows": 100000}/2 | public_caller_component | MEASURED | 7.0466 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I16 | amd/gfx942 | {"AB_FEATURES": "33", "AB_QUERIES": "128", "AB_ROWS": "100000"}/{"features": 33, "occupancy": 2, "queries": 128, "rows": 100000}/1 | public_caller_component | MEASURED | 2.1375 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I16 | amd/gfx942 | {"AB_FEATURES": "33", "AB_QUERIES": "128", "AB_ROWS": "100000"}/{"features": 33, "occupancy": 2, "queries": 128, "rows": 100000}/2 | public_caller_component | MEASURED | 7.4041 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I16 | amd/gfx942 | {"AB_FEATURES": "65", "AB_QUERIES": "128", "AB_ROWS": "100000"}/{"features": 65, "occupancy": 0, "queries": 128, "rows": 100000}/1 | public_caller_component | MEASURED | 1.4048 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I16 | amd/gfx942 | {"AB_FEATURES": "65", "AB_QUERIES": "128", "AB_ROWS": "100000"}/{"features": 65, "occupancy": 0, "queries": 128, "rows": 100000}/2 | public_caller_component | MEASURED | 3.4741 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I16 | amd/gfx942 | {"AB_FEATURES": "65", "AB_QUERIES": "128", "AB_ROWS": "100000"}/{"features": 65, "occupancy": 1, "queries": 128, "rows": 100000}/1 | public_caller_component | MEASURED | 1.3810 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I16 | amd/gfx942 | {"AB_FEATURES": "65", "AB_QUERIES": "128", "AB_ROWS": "100000"}/{"features": 65, "occupancy": 1, "queries": 128, "rows": 100000}/2 | public_caller_component | MEASURED | 3.4144 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I16 | amd/gfx942 | {"AB_FEATURES": "65", "AB_QUERIES": "128", "AB_ROWS": "100000"}/{"features": 65, "occupancy": 2, "queries": 128, "rows": 100000}/1 | public_caller_component | MEASURED | 1.3959 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I16 | amd/gfx942 | {"AB_FEATURES": "65", "AB_QUERIES": "128", "AB_ROWS": "100000"}/{"features": 65, "occupancy": 2, "queries": 128, "rows": 100000}/2 | public_caller_component | MEASURED | 3.2526 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I17 | amd/gfx942 | {"AB_FEATURES": "17", "AB_ROWS": "10000"}/{"features": 17, "policy": "Depthwise", "rows": 10000}/candidate | public_caller_component | NO_DISTINCT_RUNTIME_ARM | — | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I17 | amd/gfx942 | {"AB_FEATURES": "17", "AB_ROWS": "10000"}/{"features": 17, "policy": "Lossguide", "rows": 10000}/candidate | public_caller_component | MEASURED | 1.0407 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I17 | amd/gfx942 | {"AB_FEATURES": "18", "AB_ROWS": "10001"}/{"features": 18, "policy": "Depthwise", "rows": 10001}/candidate | public_caller_component | NO_DISTINCT_RUNTIME_ARM | — | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I17 | amd/gfx942 | {"AB_FEATURES": "18", "AB_ROWS": "10001"}/{"features": 18, "policy": "Lossguide", "rows": 10001}/candidate | public_caller_component | MEASURED | 1.2920 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I17 | amd/gfx942 | {"AB_FEATURES": "9", "AB_ROWS": "32769"}/{"features": 9, "policy": "Depthwise", "rows": 32769}/candidate | public_caller_component | NO_DISTINCT_RUNTIME_ARM | — | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I17 | amd/gfx942 | {"AB_FEATURES": "9", "AB_ROWS": "32769"}/{"features": 9, "policy": "Lossguide", "rows": 32769}/candidate | public_caller_component | MEASURED | 1.0825 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I17 | amd/gfx942 | {"AB_FEATURES": "17", "AB_ROWS": "10000"}/{"features": 17, "policy": "Depthwise", "rows": 10000}/inherit_off | public_caller_component | MEASURED | 0.9113 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I17 | amd/gfx942 | {"AB_FEATURES": "17", "AB_ROWS": "10000"}/{"features": 17, "policy": "Lossguide", "rows": 10000}/inherit_off | public_caller_component | MEASURED | 0.8183 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I17 | amd/gfx942 | {"AB_FEATURES": "18", "AB_ROWS": "10001"}/{"features": 18, "policy": "Depthwise", "rows": 10001}/inherit_off | public_caller_component | MEASURED | 1.2276 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I17 | amd/gfx942 | {"AB_FEATURES": "18", "AB_ROWS": "10001"}/{"features": 18, "policy": "Lossguide", "rows": 10001}/inherit_off | public_caller_component | MEASURED | 1.1852 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I17 | amd/gfx942 | {"AB_FEATURES": "9", "AB_ROWS": "32769"}/{"features": 9, "policy": "Depthwise", "rows": 32769}/inherit_off | public_caller_component | MEASURED | 0.9474 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I17 | amd/gfx942 | {"AB_FEATURES": "9", "AB_ROWS": "32769"}/{"features": 9, "policy": "Lossguide", "rows": 32769}/inherit_off | public_caller_component | MEASURED | 0.8349 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I17 | amd/gfx942 | {"AB_FEATURES": "17", "AB_ROWS": "10000"}/{"features": 17, "policy": "Depthwise", "rows": 10000}/legacy_both_off | public_caller_component | MEASURED | 1.0915 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I17 | amd/gfx942 | {"AB_FEATURES": "17", "AB_ROWS": "10000"}/{"features": 17, "policy": "Lossguide", "rows": 10000}/legacy_both_off | public_caller_component | MEASURED | 2.0548 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I17 | amd/gfx942 | {"AB_FEATURES": "18", "AB_ROWS": "10001"}/{"features": 18, "policy": "Depthwise", "rows": 10001}/legacy_both_off | public_caller_component | MEASURED | 1.3059 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I17 | amd/gfx942 | {"AB_FEATURES": "18", "AB_ROWS": "10001"}/{"features": 18, "policy": "Lossguide", "rows": 10001}/legacy_both_off | public_caller_component | MEASURED | 2.5077 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I17 | amd/gfx942 | {"AB_FEATURES": "9", "AB_ROWS": "32769"}/{"features": 9, "policy": "Depthwise", "rows": 32769}/legacy_both_off | public_caller_component | MEASURED | 1.1496 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I17 | amd/gfx942 | {"AB_FEATURES": "9", "AB_ROWS": "32769"}/{"features": 9, "policy": "Lossguide", "rows": 32769}/legacy_both_off | public_caller_component | MEASURED | 2.1078 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I17 | amd/gfx942 | {"AB_FEATURES": "17", "AB_ROWS": "10000"}/{"features": 17, "policy": "Depthwise", "rows": 10000}/lg_exact_off | public_caller_component | NO_DISTINCT_RUNTIME_ARM | — | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I17 | amd/gfx942 | {"AB_FEATURES": "17", "AB_ROWS": "10000"}/{"features": 17, "policy": "Lossguide", "rows": 10000}/lg_exact_off | public_caller_component | MEASURED | 1.8939 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I17 | amd/gfx942 | {"AB_FEATURES": "18", "AB_ROWS": "10001"}/{"features": 18, "policy": "Depthwise", "rows": 10001}/lg_exact_off | public_caller_component | NO_DISTINCT_RUNTIME_ARM | — | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I17 | amd/gfx942 | {"AB_FEATURES": "18", "AB_ROWS": "10001"}/{"features": 18, "policy": "Lossguide", "rows": 10001}/lg_exact_off | public_caller_component | MEASURED | 2.3284 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I17 | amd/gfx942 | {"AB_FEATURES": "9", "AB_ROWS": "32769"}/{"features": 9, "policy": "Depthwise", "rows": 32769}/lg_exact_off | public_caller_component | NO_DISTINCT_RUNTIME_ARM | — | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I17 | amd/gfx942 | {"AB_FEATURES": "9", "AB_ROWS": "32769"}/{"features": 9, "policy": "Lossguide", "rows": 32769}/lg_exact_off | public_caller_component | MEASURED | 1.9891 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I18 | amd/gfx942 | {"AB_FEATURES": "32", "AB_ROWS": "100000"}/{}/candidate | public_caller_component | MEASURED | 1.0158 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I18 | amd/gfx942 | {"AB_FEATURES": "33", "AB_ROWS": "100001"}/{}/candidate | public_caller_component | MEASURED | 1.0164 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I18 | amd/gfx942 | {"AB_FEATURES": "17", "AB_ROWS": "65537"}/{}/candidate | public_caller_component | MEASURED | 1.0347 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I19 | amd/gfx942 | I19-amd-incumbent-I19-shape-ecf5aa990933 | public_caller_component | UNSUPPORTED_BASELINE_SHAPE | — | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I19 | amd/gfx942 | I19-amd-incumbent-I19-shape-5327ea2f296b | public_caller_component | UNSUPPORTED_BASELINE_SHAPE | — | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I19 | amd/gfx942 | I19-amd-incumbent-I19-shape-f80914fff0d4 | public_caller_component | UNSUPPORTED_BASELINE_SHAPE | — | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I19 | amd/gfx942 | {"AB_ROWS": "65537"}/{"rows": 65537, "segments": 32}/candidate | public_caller_component | MEASURED | 0.6830 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I19 | amd/gfx942 | {"AB_ROWS": "98305"}/{"rows": 98305, "segments": 32}/candidate | public_caller_component | MEASURED | 0.4647 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I19 | amd/gfx942 | {"AB_ROWS": "131071"}/{"rows": 131071, "segments": 32}/candidate | public_caller_component | MEASURED | 0.3561 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I19 | amd/gfx942 | {"AB_ROWS": "65537"}/{"rows": 65537, "segments": 32}/nibble_256 | public_caller_component | MEASURED | 1.3220 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I19 | amd/gfx942 | {"AB_ROWS": "98305"}/{"rows": 98305, "segments": 32}/nibble_256 | public_caller_component | MEASURED | 0.8998 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I19 | amd/gfx942 | {"AB_ROWS": "131071"}/{"rows": 131071, "segments": 32}/nibble_256 | public_caller_component | MEASURED | 0.6745 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I19 | amd/gfx942 | {"AB_ROWS": "65537"}/{"rows": 65537, "segments": 32}/nibble_128 | public_caller_component | MEASURED | 0.8870 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I19 | amd/gfx942 | {"AB_ROWS": "98305"}/{"rows": 98305, "segments": 32}/nibble_128 | public_caller_component | MEASURED | 0.5724 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I19 | amd/gfx942 | {"AB_ROWS": "131071"}/{"rows": 131071, "segments": 32}/nibble_128 | public_caller_component | MEASURED | 0.4311 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I20 | amd/gfx942 | {"AB_FEATURES": "8", "AB_ROWS": "100000"}/{"features": 8, "queries": 1024, "rows": 100000}/candidate | public_caller_component | MEASURED | 1.3915 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I20 | amd/gfx942 | {"AB_FEATURES": "9", "AB_ROWS": "100000"}/{"features": 9, "queries": 1024, "rows": 100000}/candidate | public_caller_component | MEASURED | 1.1601 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I20 | amd/gfx942 | {"AB_FEATURES": "17", "AB_ROWS": "100000"}/{"features": 17, "queries": 1024, "rows": 100000}/candidate | public_caller_component | MEASURED | 1.0696 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I20 | amd/gfx942 | {"AB_FEATURES": "17", "AB_QUERIES": "513", "AB_ROWS": "131071"}/{"features": 17, "queries": 513, "rows": 131071}/candidate | public_caller_component | MEASURED | 0.9339 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I21 | amd/gfx942 | {"AB_FEATURES": "17", "AB_ROWS": "10000"}/{"components": 9, "features": 17, "rows": 10000}/candidate | public_caller_component | MEASURED | 1.1384 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I21 | amd/gfx942 | {"AB_FEATURES": "18", "AB_ROWS": "10001"}/{"components": 9, "features": 18, "rows": 10001}/candidate | public_caller_component | MEASURED | 1.0823 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I21 | amd/gfx942 | {"AB_FEATURES": "9", "AB_ROWS": "32769"}/{"components": 9, "features": 9, "rows": 32769}/candidate | public_caller_component | MEASURED | 1.0477 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I22 | amd/gfx942 | {"AB_FEATURES": "33", "AB_ROWS": "65537"}/{"features": 33, "rows": 65537}/candidate | public_caller_component | MEASURED | 0.5641 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I22 | amd/gfx942 | {"AB_FEATURES": "34", "AB_ROWS": "65539"}/{"features": 34, "rows": 65539}/candidate | public_caller_component | MEASURED | 0.5639 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I22 | amd/gfx942 | {"AB_FEATURES": "17", "AB_ROWS": "131073"}/{"features": 17, "rows": 131073}/candidate | public_caller_component | MEASURED | 0.5407 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I23 | amd/gfx942 | {"AB_OBSERVATIONS": "4096"}/{"batch": 6, "case": "planted_ar1", "observations": 4096}/candidate | public_caller_component | NO_DISTINCT_RUNTIME_ARM | — | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I23 | amd/gfx942 | {"AB_OBSERVATIONS": "4096"}/{"batch": 6, "case": "planted_ma1", "observations": 4096}/candidate | public_caller_component | NO_DISTINCT_RUNTIME_ARM | — | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I23 | amd/gfx942 | {"AB_OBSERVATIONS": "4096"}/{"batch": 6, "case": "planted_arma11", "observations": 4096}/candidate | public_caller_component | NO_DISTINCT_RUNTIME_ARM | — | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I23 | amd/gfx942 | {"AB_OBSERVATIONS": "4097"}/{"batch": 6, "case": "planted_ar1", "observations": 4097}/candidate | public_caller_component | NO_DISTINCT_RUNTIME_ARM | — | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I23 | amd/gfx942 | {"AB_OBSERVATIONS": "4097"}/{"batch": 6, "case": "planted_ma1", "observations": 4097}/candidate | public_caller_component | NO_DISTINCT_RUNTIME_ARM | — | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I23 | amd/gfx942 | {"AB_OBSERVATIONS": "4097"}/{"batch": 6, "case": "planted_arma11", "observations": 4097}/candidate | public_caller_component | NO_DISTINCT_RUNTIME_ARM | — | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I23 | amd/gfx942 | {"AB_OBSERVATIONS": "8193"}/{"batch": 6, "case": "planted_ar1", "observations": 8193}/candidate | public_caller_component | NO_DISTINCT_RUNTIME_ARM | — | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I23 | amd/gfx942 | {"AB_OBSERVATIONS": "8193"}/{"batch": 6, "case": "planted_ma1", "observations": 8193}/candidate | public_caller_component | NO_DISTINCT_RUNTIME_ARM | — | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I23 | amd/gfx942 | {"AB_OBSERVATIONS": "8193"}/{"batch": 6, "case": "planted_arma11", "observations": 8193}/candidate | public_caller_component | NO_DISTINCT_RUNTIME_ARM | — | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I24 | amd/gfx942 | {"AB_ROWS": "1000000"}/{"classes": 33, "rows": 1000000}/candidate | public_caller_component | MEASURED | 0.8580 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I24 | amd/gfx942 | {"AB_ROWS": "1000001"}/{"classes": 33, "rows": 1000001}/candidate | public_caller_component | MEASURED | 0.8728 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I24 | amd/gfx942 | {"AB_ROWS": "1048577"}/{"classes": 33, "rows": 1048577}/candidate | public_caller_component | MEASURED | 0.8588 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| I24 | amd/gfx942 | {"AB_CLASSES": "17", "AB_ROWS": "1048573"}/{"classes": 17, "rows": 1048573}/candidate | public_caller_component | MEASURED | 0.9355 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| N05 | amd/gfx942 | {"AB_K": "1025", "AB_KIND": "3", "AB_M": "1025", "AB_N": "513"}/{"jobs": 3, "k": 1025, "kind": 3, "m": 1025, "n": 513, "op": 0, "version": 0} | component | MEASURED | 0.9558 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| N05 | amd/gfx942 | {"AB_K": "1025", "AB_KIND": "3", "AB_M": "1025", "AB_N": "513"}/{"jobs": 3, "k": 1025, "kind": 3, "m": 1025, "n": 513, "op": 1, "version": 0} | component | MEASURED | 0.9692 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| N05 | amd/gfx942 | {"AB_K": "1025", "AB_KIND": "3", "AB_M": "1025", "AB_N": "513"}/{"jobs": 3, "k": 1025, "kind": 3, "m": 1025, "n": 513, "op": 2, "version": 0} | component | MEASURED | 0.9646 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| N05 | amd/gfx942 | {"AB_K": "1025", "AB_KIND": "3", "AB_M": "1025", "AB_N": "513"}/{"jobs": 3, "k": 1025, "kind": 3, "m": 1025, "n": 513, "op": 0, "version": 1} | component | MEASURED | 1.0873 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| N05 | amd/gfx942 | {"AB_K": "1025", "AB_KIND": "3", "AB_M": "1025", "AB_N": "513"}/{"jobs": 3, "k": 1025, "kind": 3, "m": 1025, "n": 513, "op": 1, "version": 1} | component | MEASURED | 0.9914 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| N05 | amd/gfx942 | {"AB_K": "1025", "AB_KIND": "3", "AB_M": "1025", "AB_N": "513"}/{"jobs": 3, "k": 1025, "kind": 3, "m": 1025, "n": 513, "op": 2, "version": 1} | component | MEASURED | 0.9646 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| N05 | amd/gfx942 | {"AB_K": "1025", "AB_KIND": "3", "AB_M": "1025", "AB_N": "513"}/{"jobs": 3, "k": 1025, "kind": 3, "m": 1025, "n": 513, "op": 0, "version": 2} | component | MEASURED | 0.9816 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| N05 | amd/gfx942 | {"AB_K": "1025", "AB_KIND": "3", "AB_M": "1025", "AB_N": "513"}/{"jobs": 3, "k": 1025, "kind": 3, "m": 1025, "n": 513, "op": 1, "version": 2} | component | MEASURED | 0.9840 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| N05 | amd/gfx942 | {"AB_K": "1025", "AB_KIND": "3", "AB_M": "1025", "AB_N": "513"}/{"jobs": 3, "k": 1025, "kind": 3, "m": 1025, "n": 513, "op": 2, "version": 2} | component | MEASURED | 0.9752 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| N05 | amd/gfx942 | {"AB_K": "1025", "AB_KIND": "3", "AB_M": "257", "AB_N": "259"}/{"jobs": 3, "k": 1025, "kind": 3, "m": 257, "n": 259, "op": 0, "version": 0} | component | MEASURED | 0.3591 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| N05 | amd/gfx942 | {"AB_K": "1025", "AB_KIND": "3", "AB_M": "257", "AB_N": "259"}/{"jobs": 3, "k": 1025, "kind": 3, "m": 257, "n": 259, "op": 1, "version": 0} | component | MEASURED | 1.5720 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| N05 | amd/gfx942 | {"AB_K": "1025", "AB_KIND": "3", "AB_M": "257", "AB_N": "259"}/{"jobs": 3, "k": 1025, "kind": 3, "m": 257, "n": 259, "op": 2, "version": 0} | component | MEASURED | 0.3902 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| N05 | amd/gfx942 | {"AB_K": "1025", "AB_KIND": "3", "AB_M": "257", "AB_N": "259"}/{"jobs": 3, "k": 1025, "kind": 3, "m": 257, "n": 259, "op": 0, "version": 1} | component | MEASURED | 0.3544 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| N05 | amd/gfx942 | {"AB_K": "1025", "AB_KIND": "3", "AB_M": "257", "AB_N": "259"}/{"jobs": 3, "k": 1025, "kind": 3, "m": 257, "n": 259, "op": 1, "version": 1} | component | MEASURED | 1.5554 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| N05 | amd/gfx942 | {"AB_K": "1025", "AB_KIND": "3", "AB_M": "257", "AB_N": "259"}/{"jobs": 3, "k": 1025, "kind": 3, "m": 257, "n": 259, "op": 2, "version": 1} | component | MEASURED | 0.3947 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| N05 | amd/gfx942 | {"AB_K": "1025", "AB_KIND": "3", "AB_M": "257", "AB_N": "259"}/{"jobs": 3, "k": 1025, "kind": 3, "m": 257, "n": 259, "op": 0, "version": 2} | component | MEASURED | 0.3523 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| N05 | amd/gfx942 | {"AB_K": "1025", "AB_KIND": "3", "AB_M": "257", "AB_N": "259"}/{"jobs": 3, "k": 1025, "kind": 3, "m": 257, "n": 259, "op": 1, "version": 2} | component | MEASURED | 1.5437 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| N05 | amd/gfx942 | {"AB_K": "1025", "AB_KIND": "3", "AB_M": "257", "AB_N": "259"}/{"jobs": 3, "k": 1025, "kind": 3, "m": 257, "n": 259, "op": 2, "version": 2} | component | MEASURED | 0.3878 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| N05 | amd/gfx942 | {"AB_K": "2049", "AB_KIND": "3", "AB_M": "1023", "AB_N": "1025"}/{"jobs": 3, "k": 2049, "kind": 3, "m": 1023, "n": 1025, "op": 0, "version": 0} | component | MEASURED | 0.9263 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| N05 | amd/gfx942 | {"AB_K": "2049", "AB_KIND": "3", "AB_M": "1023", "AB_N": "1025"}/{"jobs": 3, "k": 2049, "kind": 3, "m": 1023, "n": 1025, "op": 1, "version": 0} | component | MEASURED | 1.0585 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| N05 | amd/gfx942 | {"AB_K": "2049", "AB_KIND": "3", "AB_M": "1023", "AB_N": "1025"}/{"jobs": 3, "k": 2049, "kind": 3, "m": 1023, "n": 1025, "op": 2, "version": 0} | component | MEASURED | 0.9797 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| N05 | amd/gfx942 | {"AB_K": "2049", "AB_KIND": "3", "AB_M": "1023", "AB_N": "1025"}/{"jobs": 3, "k": 2049, "kind": 3, "m": 1023, "n": 1025, "op": 0, "version": 1} | component | MEASURED | 0.9243 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| N05 | amd/gfx942 | {"AB_K": "2049", "AB_KIND": "3", "AB_M": "1023", "AB_N": "1025"}/{"jobs": 3, "k": 2049, "kind": 3, "m": 1023, "n": 1025, "op": 1, "version": 1} | component | MEASURED | 1.0551 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| N05 | amd/gfx942 | {"AB_K": "2049", "AB_KIND": "3", "AB_M": "1023", "AB_N": "1025"}/{"jobs": 3, "k": 2049, "kind": 3, "m": 1023, "n": 1025, "op": 2, "version": 1} | component | MEASURED | 0.9796 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| N05 | amd/gfx942 | {"AB_K": "2049", "AB_KIND": "3", "AB_M": "1023", "AB_N": "1025"}/{"jobs": 3, "k": 2049, "kind": 3, "m": 1023, "n": 1025, "op": 0, "version": 2} | component | MEASURED | 0.9216 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| N05 | amd/gfx942 | {"AB_K": "2049", "AB_KIND": "3", "AB_M": "1023", "AB_N": "1025"}/{"jobs": 3, "k": 2049, "kind": 3, "m": 1023, "n": 1025, "op": 1, "version": 2} | component | MEASURED | 1.0700 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| N05 | amd/gfx942 | {"AB_K": "2049", "AB_KIND": "3", "AB_M": "1023", "AB_N": "1025"}/{"jobs": 3, "k": 2049, "kind": 3, "m": 1023, "n": 1025, "op": 2, "version": 2} | component | MEASURED | 0.9696 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| N06 | amd/gfx942 | {}/{"heads": 4, "length": 512}/1 | public_caller_component | MEASURED | 0.7114 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| N07 | amd/gfx942 | {}/{}/candidate | public_caller_component | MEASURED | 1.0340 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |
| N08 | amd/gfx942 | {}/{"columns": 4096, "features": 8, "rows": 1024}/1 | public_caller_component | MEASURED | 0.9636 | /Users/andrewhendel/mojolearn-evidence/overnight-ab-20261006/amd/live/repairs/results.json |

## Campaign notes

- Candidate A/B first; missing GPU opponents after candidate queues and applicable repairs.
- Rentals delete after 1800 seconds genuinely idle, after captured evidence; M3 is retained.
- Existing identity and compilation accepted by owner; no separate validation passes.
- Reused native component executables warm up in a separate process; scored first calls may include JIT. These are not steady-state or full-workload promotion evidence.
- Winners and losers are recorded beside source toggles as sufficient measurements arrive; partial component screens leave defaults unchanged.

## Recorded source decisions

| Candidate / arm | Decision | Source commit | Evidence |
|---|---|---|---|
| F03 / resident | CONFIRMED EXISTING DEFAULT: B/A trajectory 0.5125–0.5565; already enabled | b953bc9a2 | M3 measured heldout and resume results; source _byte_lm_impl.py _is_resident |
| F08 / BWD_NOSYNC | PROMOTED: Apple FAST only; 12-step train B/A0.8766 with matching loss/resume; explicit OFF escape; one scored sample | b953bc9a2 | training/byte_lm_afn.mojo:53; M3 F08/default retained result |
| F08 / fused, views | RETAIN OFF: small or mixed gain; views cold call regressed | b953bc9a2 | M3 F08 independent-arm results; baseline manifests explicitly disable NOSYNC |
| F07, F10, F11 | RETAIN OFF for evaluated experimental arms: mixed/regressing measured cases; see source for each scoped outcome | b953bc9a2 | Inline toggle annotations retain case timing and quality counts |
| F12 / ordered-storage | CONFIRMED EXISTING DEFAULT: fit+first prediction B/A0.7802,0.8846; repeated0.9203,0.9955; same task error | 9ad011c2a | gbdt/methods/ordered_fast_switches.mojo ORD_ALL; 511x11 and1023x17 24-tree M3 tasks |
| F15 / HDBSCAN linkage | PROMOTED: Apple FAST only with OFF escape; all3 planned fits faster, matching task quality. Download combination pending; downloads stays OFF. | 429622bef | hdbscan/impl/detail/fast_apple.mojo:77; M3 cold B/A0.8085,0.6857,0.6613; repeated0.6957,0.6656,0.6698 |
| F18 / KDE DIRECT_PREP | PROMOTED: Apple FAST only with OFF escape; all3 planned shapes improve, equal captured task quality, refit/refusal recovery completed. Immutable-fit remains OFF. | 709f43abb | M3 cold B/A0.6349,0.9221,0.8119; repeated0.4917,0.7277,0.9234; shapes509x3,521x7,997x13 |
| F13 / SHAP row-pair | RETAIN OFF: repaired real x_trees binding reaches candidate but all3 cases regress, equal quality | dafcb7854 | M3 cold B/A1.0682,1.1598,1.7360; repeated1.0757,1.4591,1.6713 |
| F20 / multitensor, fused scan | RETAIN OFF: SGD-only multitensor gain leaves broader optimizer coverage pending; fused scan regresses | 709f43abb | M3 SGD12step B/A0.8528 with equal losses/refusal; fused-scan1.0514 |
| N03, I05, N07 | RETAIN OFF: component losses documented beside runtime candidate entrypoints | f9499bef8 / 7d8bf33f1 | Captured same-context component timing; no full-workload default claim |
| I03, N05, N06, N08, I20, I24 | RETAIN DEFAULTS: component wins, mixed results or small gains; wider applicable measurement contract pending | f9499bef8 / 7d8bf33f1 | Inline source annotations retain actual ratios, cases and limits |
