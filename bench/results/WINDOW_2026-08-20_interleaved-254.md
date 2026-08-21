# 800k x 100, interleaved against CatBoost CPU, 2026-08-20 afternoon

    pixi run -e bench mojo run -I . \
        bench/interleaved/catboost_interleaved.mojo <dir> synth 800000

Same window, one process, arms alternating per rep, same rows and the same
CatBoost-built grid on both sides. M4 base, 10 GPU cores, AC power, load
1.29, nothing else on the GPU (the build lock was held for the duration).

    border  their arm      theirs       ours      speedup   our mse       their mse
    ------  ---------  ----------  ---------  -----------  -----------  -----------
    254     CPU        30.06 ms/t  35.35 ms/t  0.850x      0.141448232  0.141448238
    254     CPU        30.74       35.76       0.860x      0.141448232  0.141448238
    254     CPU        29.85       35.35       0.844x      0.141448232  0.141448238
    128     CPU        30.55       29.35       1.041x      0.151470605  0.151470604
    128     CPU        30.06       29.13       1.032x      0.151470605  0.151470604
    128     CPU        30.02       29.64       1.013x      0.151470605  0.151470604

`speedup` > 1 means ours is faster.

## What moved

Against the last recorded interleaved table in RESUME.md (after `482661e`):

    800k x 100 @ 254:   2.1x behind   ->  1.18x behind
    800k x 100 @ 128:   1.02-1.05x behind  ->  1.01-1.04x AHEAD

The 254 row is the whole story. `992aa86` gave the PASS family
(`TPointHistOneByte`, the 129-255-bin range) the same 2-warp-shared Int32
arm hist_2 already had, and that is a 1.9x cut on this dataset.

## The accuracy trap at 254 is closed

`992aa86` was recorded as tripping a dither trap: at 254 borders on synth
the blanket 2^28 scale flipped near-tied splits and train mse came out
**0.14375 against CatBoost's 0.14145**. Every oracle stayed green because
4096 rows never builds the tie density.

This run reads **0.141448232 vs their 0.141448238** -- agreement to eight
significant figures, on the exact dataset and border count that exposed it.
Whatever landed between then and now fixed it. That is a correctness result
and it is worth more than the speed row above it.

## What this is NOT

- **Three reps, one window, one machine.** The 128 rows span 1.013x to
  1.041x, which straddles nothing but is thin. "At parity, perhaps a hair
  ahead" is the honest read of 128; "we win" is not.
- **Our GPU against their CPU.** CatBoost's GPU arm cannot run on this box
  at all -- `task_type="GPU"` raises on Apple. That is the claim, not a
  handicap we chose, but it must be said beside the number every time.
  `tools/nvidia_bench.sh` exists to time their CUDA arm when we rent a box,
  and we expect to lose that column.
- **Not a full-loop audit.** The timed region is `fit`, which is the
  boosting loop -- gradients, tree, model update -- so the old "ours is tree
  growth alone" caveat does not apply to these numbers. Whether the two
  loops charge for exactly the same work has not been audited line by line.
- **Synthetic.** covtype was not re-run in this window. The last recorded
  covtype standing is 2.2x/2.7x behind and is now stale in the same
  direction; it should be re-run before either number is quoted.

## 2,000,000 x 100, independent replication of SCALING_2026-08-20_catboost.md

Same afternoon, this session, at `05f8fd0`. The peer's ladder was taken at
`028da27` in a different session. Medians of three reps:

    borders   theirs (CPU)   ours (GPU)   speedup   peer's reading
    -------   ------------   ----------   -------   --------------
    254          80.70          59.81      1.35x    80.72 / 60.22  -> 1.34x
    128          78.91          52.34      1.51x    80.31 / 53.75  -> 1.49x

Two sessions, two commits, agreement inside 3%. The 800k rows above
replicate the same file to under 1%. This ladder is not one run read twice.

**One outlier, reported rather than dropped.** At 128 borders rep 2 read
CatBoost at 92.80 ms/tree against 78.49 and 78.91 in the other two reps,
which alone would print 1.77x. Peer sessions were active on this box. The
median is the honest figure and 1.77x should not be quoted.

**The 254-border model defect reproduces BIT-FOR-BIT.** Ours
0.14395396875 against CatBoost's 0.1428514395525756 -- the same two values
the peer recorded, to every digit they printed, from a different session and
a different commit. So it is deterministic and reproducible, not drift, and
the 1.35x at 254 remains a win on a model CatBoost did not train. At 128 the
same run matches to eight significant figures (0.15073115625 vs
0.150731169852181) and that speedup stands.

Our MSE is also identical across all three reps at both border counts, which
is what the Apple integer-accumulation arm claims and is worth noting as
having been observed rather than assumed.
