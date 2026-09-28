# linear-apple: Apple (Metal) speed for the linear family

Brief: ~/mojolearn-evidence/apple_speed_brief.md (2026-09-28). Branch
`lane/linear-apple` off origin/main 5b622763d; not merged by this lane (the
merge-gate runners merge it after an NVIDIA + CPU gate). Home Mac for speed
jobs: m4pro-a (Apple M4 Pro). Every job below went through
`tools/apple_steward.py submit --kind speed --target m4pro-a`; before and
after are on the same Mac.

## Step 1: profile on Metal

### x_linear (the 21 expansion estimators), 100k rows, IDENTICAL, main 5b622763d

`bench/x_linear_speed.py --rows 100000 --column both` (steward
1790579978918-speed-linear-5b622763d3). gpu = the Metal binding, host = the
CPU host binding, one thread. Digests equal gpu vs host on every line.

| case | Metal fit s | host fit s | Metal / host |
|---|---|---|---|
| sgd-clf | 6.468 | 0.214 | 30x |
| sgd-reg | 3.893 | 0.123 | 32x |
| perceptron | 3.097 | 0.130 | 24x |
| pa-clf | 4.406 | 0.196 | 22x |
| pa-reg | 3.122 | 0.115 | 27x |
| sgd-ocsvm | 3.524 | 0.118 | 30x |
| poisson | 176.102 | 5.460 | 32x |
| gamma | 199.551 | 6.813 | 29x |
| tweedie | 6.611 | 0.123 | 54x |
| huber | 22.400 | 0.727 | 31x |
| bayes-ridge | 5.629 | 0.143 | 39x |
| ard | 5.848 | 0.159 | 37x |
| lars | 4.972 | 0.120 | 41x |
| lasso-lars | 4.970 | 0.119 | 42x |

Where Apple is slow: on main every x_linear fit is ONE device thread
(x_linear/device.mojo `fit_kernel`, pass 1's design), so the Metal column is
a single GPU thread against one CPU core: 22x to 54x slower. The same is true
on NVIDIA and AMD (it is the design, not Apple); the cure is the team-fit
schedule on lane/algos-linear (x_linear/team.mojo: one block of 256 threads,
independent outputs dealt across threads, every fold still one thread's
ascending loop, so bits do not move), which is owed its gates there (see
docs/lanes/progress/linear.md, "Speed phase"). This lane does not duplicate
it: it times that branch on Metal (below) so the linear lane has the Apple
column.
