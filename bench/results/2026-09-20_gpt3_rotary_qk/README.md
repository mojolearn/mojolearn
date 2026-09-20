# GPT-3-small paired Q/K rotary screen

Apple M4, FP32 IDENTICAL, `B=16,L=2048,Hq=Hk=12,D=64`. The candidate
combined the two forward pointer domains in one grid and did the same for the
backward transposed rotation. Each cell retained the shipped cos/sin lookup,
loads, two pinned multiplies, FTZ seams, unfused add, and sign convention.

Every one of the 50,331,648 Q+K forward cells and every backward cell matched
the separate-launch baseline exactly. The full transformer forward gate also
passed 17 fixtures/30 stages and the backward gate passed 17 fixtures/37
stages against their scalar oracles.

Twelve-layer forward timings in milliseconds, separate Q/K launches followed
by the paired launch in each repetition:

```
separate 399.117 400.631 457.354 510.842 467.796 441.481
paired   416.519 443.863 515.652 503.949 466.439 457.299
```

Medians were about 454.6 ms separate and 461.9 ms paired. The saved launches
did not repay the combined-grid pointer/domain branch on Apple. The candidate
was rejected and all production source and benchmark-only kernels were
reverted. No vendor route is inferred without vendor evidence.
